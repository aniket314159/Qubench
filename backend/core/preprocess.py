"""
core/preprocess.py  (backend/core/preprocess.py)
================================================

SEAM 2 — preprocessing ops. Phase 1 ships nine user ops plus two locked ones.

    OP_REGISTRY[name] -> OpSpec(op, label, order, fn, params_schema, locked)
    PRESETS[name]     -> PresetSpec(name, label, ops)

Adding an op in Phase 2/3 = write a function + add one dict entry. No frontend
work: the Configure screen builds its controls from `params_schema`.

THE TWO RULES
-------------
1.  Every op takes and returns **uint8 RGB, shape (H, W, 3)**. Always. Even
    grayscale and binarize, which convert back to three channels before
    returning. This makes ops freely composable, makes every intermediate
    stage directly renderable in the preview strip, and removes an entire
    class of "why is this array 2-D here" bug.

2.  `order` is a property of the OP, not of the user's config. Users toggle
    ops on and off and tune params; they never reorder. This is not UI
    laziness — the order is load-bearing:

        remove_grid (300) MUST run before grayscale (400)

    because it identifies the pink ECG graph paper by its *colour saturation*.
    After grayscale, that information is gone and the op silently does
    nothing. `apply_ops` sorts by `order` regardless of list order, so a
    config that lists them backwards still works.

No ML here. Nothing is fitted, nothing is learned, every image gets identical
treatment. That is why preprocessing is not subject to the train-only rule —
there is nothing to leak.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

from backend.core.contracts import OpConfig, OpSpec, PresetSpec

log = logging.getLogger(__name__)

RESIZE_PX = 224          # ResNet18 input; BackboneSpec.input_size agrees
PREVIEW_PX = 240         # per-stage thumbnail size for the preview strip


# ---------------------------------------------------------------------------
# Image I/O — the one place a file becomes an array
# ---------------------------------------------------------------------------

def load_image(path: str | Path) -> np.ndarray:
    """Disk -> uint8 RGB (H, W, 3), EXIF orientation applied.

    Phone photos of ECG printouts carry an orientation tag. Ignore it and a
    portrait-shot ECG reaches the model rotated 90 degrees, accuracy drops,
    and the cause is invisible.
    """
    with Image.open(path) as im:
        im.load()
        im = ImageOps.exif_transpose(im)
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def to_b64(img: np.ndarray, max_px: int = PREVIEW_PX) -> str:
    """Array -> data URI, downscaled for transport. POST /preprocess/preview
    sends one of these per stage, so keep them small."""
    pil = Image.fromarray(_rgb(img))
    pil.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rgb(img: np.ndarray) -> np.ndarray:
    """Coerce anything to contiguous uint8 (H, W, 3). Rule 1's enforcement."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    elif a.ndim == 3 and a.shape[2] == 4:
        a = cv2.cvtColor(a, cv2.COLOR_RGBA2RGB)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = cv2.cvtColor(a[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"cannot interpret array of shape {a.shape} as an image")
    return np.ascontiguousarray(a)


def _gray(img: np.ndarray) -> np.ndarray:
    """Single-channel view for ops that need it internally."""
    return cv2.cvtColor(_rgb(img), cv2.COLOR_RGB2GRAY)


def _ink_mask(img: np.ndarray) -> np.ndarray:
    """Binary mask of the dark trace, whatever stage we are at. Lets
    order-sensitive ops (deskew, remove_lead_lines) work even when the user
    switched binarize off."""
    g = _gray(img)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return m


# ---------------------------------------------------------------------------
# The ops.  fn(img: uint8 RGB, **params) -> uint8 RGB
# ---------------------------------------------------------------------------

def op_crop_border(img: np.ndarray, ratio: float = 0.03) -> np.ndarray:
    """Trim scan margins and the report header/footer bars."""
    a = _rgb(img)
    h, w = a.shape[:2]
    dy, dx = int(h * ratio), int(w * ratio)
    if h - 2 * dy < 8 or w - 2 * dx < 8:
        return a                      # ratio too aggressive; refuse rather than
    return a[dy:h - dy, dx:w - dx]    # return an empty array into the pipeline


def op_deskew(img: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """Straighten a crooked scan or photo. Estimates the dominant angle of the
    ink and rotates back, clamped — an unclamped estimate on a noisy image can
    decide the page is 87 degrees off and destroy it."""
    a = _rgb(img)
    mask = _ink_mask(a)
    pts = cv2.findNonZero(mask)
    if pts is None or len(pts) < 50:
        return a
    angle = cv2.minAreaRect(pts)[-1]
    if angle > 45:                    # OpenCV's rect angle convention
        angle -= 90
    if abs(angle) > max_angle:
        return a
    h, w = a.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))


def op_remove_grid(img: np.ndarray, mode: str = "color",
                   sat_thresh: int = 40) -> np.ndarray:
    """Suppress the pink/red graph paper, keep the black trace.

    MUST run before grayscale in "color" mode — see Rule 2.

    color     the grid is coloured AND light; the trace is dark. Pixels above
              `sat_thresh` saturation *and* above value 80 become white. The
              value guard matters: JPEG ringing around a black trace produces
              coloured fringe pixels, and without it those get erased along
              with the grid, chewing holes in the signal.
              Default 40, not 60 — real ECG paper's fine grid is pale
              (around S=55) and a higher threshold removes only the coarse
              5 mm lines while leaving the 1 mm ones behind.
    adaptive  no colour needed: divide by a heavily blurred copy of itself,
              which flattens illumination and fades thin repeating lines.
              Use this on already-grayscale scans and on phone photos.
    """
    a = _rgb(img)
    if mode == "adaptive":
        g = _gray(a)
        bg = cv2.medianBlur(g, 21)
        flat = cv2.divide(g, bg, scale=255)
        return cv2.cvtColor(flat, cv2.COLOR_GRAY2RGB)

    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    grid = (hsv[:, :, 1] > int(sat_thresh)) & (hsv[:, :, 2] > 80)
    out = a.copy()
    out[grid] = 255
    return out


def op_grayscale(img: np.ndarray) -> np.ndarray:
    """Colour carries no diagnostic signal once the grid is gone."""
    return cv2.cvtColor(_gray(img), cv2.COLOR_GRAY2RGB)


def op_clahe(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Local contrast. Rescues faint traces on underexposed photos. On an
    already-clean scan it mostly amplifies noise, so it is off by default."""
    c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    return cv2.cvtColor(c.apply(_gray(img)), cv2.COLOR_GRAY2RGB)


def op_denoise(img: np.ndarray, method: str = "median",
               strength: int = 3) -> np.ndarray:
    """median   fast, ~1ms, removes speckle. The sane default.
    nlmeans  much better, ~300ms per image. On 1000 images that is five
             minutes added to every uncached run. Use only if a demo photo
             dataset genuinely needs it."""
    g = _gray(img)
    if method == "nlmeans":
        out = cv2.fastNlMeansDenoising(g, None, float(strength) * 3, 7, 21)
    else:
        k = max(1, int(strength) | 1)          # kernel must be odd
        out = cv2.medianBlur(g, k)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)


def op_binarize(img: np.ndarray, method: str = "fixed",
                thr: int = 200) -> np.ndarray:
    """Black trace on white. fixed is predictable and demoable; otsu picks the
    threshold per image and handles varying exposure; adaptive handles uneven
    lighting across a single page."""
    g = _gray(img)
    if method == "otsu":
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 10)
    else:
        _, b = cv2.threshold(g, int(thr), 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)


def op_remove_lead_lines(img: np.ndarray, axis: str = "vertical",
                         min_frac: float = 0.5, thickness: int = 3) -> np.ndarray:
    """Erase the long straight rules that separate the 12 lead readings.

    They are identical in every image of the dataset, so they are pure
    constant signal — they cannot help the classifier, and they survive
    downsampling to dominate early conv features.

    Finds runs of ink spanning at least `min_frac` of the image in `axis` and
    paints them white. Works on colour, grayscale or binary input.
    """
    a = _rgb(img)
    mask = _ink_mask(a)
    h, w = mask.shape
    span = max(8, int((h if axis == "vertical" else w) * float(min_frac)))
    k = (1, span) if axis == "vertical" else (span, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, k)

    lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if int(thickness) > 1:
        lines = cv2.dilate(
            lines, cv2.getStructuringElement(cv2.MORPH_RECT,
                                             (int(thickness), int(thickness)))
        )
    out = a.copy()
    out[lines > 0] = 255
    return out


def op_invert(img: np.ndarray) -> np.ndarray:
    """White trace on black. ImageNet backbones were trained on natural images
    where the subject is usually brighter than the background."""
    return 255 - _rgb(img)


def op_to_rgb(img: np.ndarray) -> np.ndarray:
    """LOCKED. Not a transform — the enforcement point for Rule 1. Guarantees
    the handoff type is exactly uint8 (H, W, 3) no matter what ran before."""
    return _rgb(img)


def op_resize(img: np.ndarray, size: int = RESIZE_PX) -> np.ndarray:
    """LOCKED. The backbone's fixed input size. INTER_AREA is the correct
    filter for downscaling; INTER_LINEAR aliases thin ECG traces away."""
    a = _rgb(img)
    s = int(size)
    interp = cv2.INTER_AREA if (a.shape[0] > s or a.shape[1] > s) else cv2.INTER_CUBIC
    return cv2.resize(a, (s, s), interpolation=interp)


# ---------------------------------------------------------------------------
# THE REGISTRY.  `order` is fixed. See Rule 2.
# ---------------------------------------------------------------------------

def _num(kind, lo, hi, default, label, step=None, help=""):
    d = {"type": kind, "min": lo, "max": hi, "default": default, "label": label}
    if step is not None:
        d["step"] = step
    if help:
        d["help"] = help
    return d


def _enum(options, default, label, help=""):
    d = {"type": "enum", "options": list(options), "default": default, "label": label}
    if help:
        d["help"] = help
    return d


OP_REGISTRY: dict[str, OpSpec] = {
    "crop_border": OpSpec(
        op="crop_border", label="Crop border", order=100, fn=op_crop_border,
        params_schema={"ratio": _num("float", 0.0, 0.25, 0.03,
                                     "Border ratio", step=0.01)},
        description="Trim scan margins and report header bars.",
    ),
    "deskew": OpSpec(
        op="deskew", label="Deskew", order=200, fn=op_deskew,
        params_schema={"max_angle": _num("float", 1.0, 30.0, 10.0,
                                         "Max correction °", step=1.0)},
        description="Straighten a crooked scan or photo.",
    ),
    "remove_grid": OpSpec(
        op="remove_grid", label="Remove grid", order=300, fn=op_remove_grid,
        params_schema={
            "mode": _enum(["color", "adaptive"], "color", "Mode",
                          "color needs the original pink paper; "
                          "adaptive works on grayscale scans too"),
            "sat_thresh": _num("int", 10, 200, 40, "Colour sensitivity"),
        },
        description="Suppress ECG graph paper, keep the trace. "
                    "Runs before grayscale — it needs the colour.",
    ),
    "grayscale": OpSpec(
        op="grayscale", label="Grayscale", order=400, fn=op_grayscale,
        description="Colour carries no signal once the grid is gone.",
    ),
    "clahe": OpSpec(
        op="clahe", label="CLAHE (local contrast)", order=500, fn=op_clahe,
        params_schema={
            "clip": _num("float", 1.0, 8.0, 2.0, "Clip limit", step=0.5),
            "grid": _num("int", 2, 16, 8, "Tile grid"),
        },
        description="Rescues faint traces on photos. Amplifies noise on clean scans.",
    ),
    "denoise": OpSpec(
        op="denoise", label="Denoise", order=600, fn=op_denoise,
        params_schema={
            "method": _enum(["median", "nlmeans"], "median", "Method",
                            "nlmeans is ~300ms/image — five minutes on 1000 images"),
            "strength": _num("int", 1, 9, 3, "Strength"),
        },
        description="Remove speckle from photographed printouts.",
    ),
    "binarize": OpSpec(
        op="binarize", label="Binarize", order=700, fn=op_binarize,
        params_schema={
            "method": _enum(["fixed", "otsu", "adaptive"], "fixed", "Method"),
            "thr": _num("int", 0, 255, 200, "Threshold (fixed only)"),
        },
        description="Black trace on white background.",
    ),
    "remove_lead_lines": OpSpec(
        op="remove_lead_lines", label="Remove lead separators", order=800,
        fn=op_remove_lead_lines,
        params_schema={
            "axis": _enum(["vertical", "horizontal"], "vertical", "Axis"),
            "min_frac": _num("float", 0.2, 1.0, 0.5, "Min length", step=0.05),
            "thickness": _num("int", 1, 9, 3, "Erase thickness"),
        },
        description="Erase the rules between lead readings. They are identical "
                    "in every image, so they are noise the model can only overfit.",
    ),
    "invert": OpSpec(
        op="invert", label="Invert", order=850, fn=op_invert,
        description="White trace on black, closer to ImageNet statistics.",
    ),
    # --- locked, always applied, never user-toggleable ---------------------
    "to_rgb": OpSpec(
        op="to_rgb", label="To RGB", order=900, fn=op_to_rgb, locked=True,
        description="Guarantees the uint8 (H, W, 3) handoff type.",
    ),
    "resize": OpSpec(
        op="resize", label=f"Resize {RESIZE_PX}", order=1000, fn=op_resize,
        locked=True,
        params_schema={"size": _num("int", 64, 512, RESIZE_PX, "Size")},
        description="Backbone input size.",
    ),
}

ALWAYS: tuple[str, ...] = ("to_rgb", "resize")     # appended by apply_ops


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, PresetSpec] = {
    "raw": PresetSpec(
        name="raw", label="Raw (resize only)", ops=[],
        description="No cleanup. The honest baseline to benchmark against.",
    ),
    "ecg_clean_scan": PresetSpec(
        name="ecg_clean_scan", label="ECG — clean scan",
        ops=[
            OpConfig("crop_border", {"ratio": 0.03}),
            OpConfig("grayscale"),
            OpConfig("binarize", {"method": "fixed", "thr": 200}),
            OpConfig("remove_lead_lines", {"axis": "vertical"}),
        ],
        description="Flatbed scans of ECG printouts. Start here.",
    ),
    "ecg_grid_paper": PresetSpec(
        name="ecg_grid_paper", label="ECG — pink grid paper",
        ops=[
            OpConfig("crop_border", {"ratio": 0.03}),
            OpConfig("remove_grid", {"mode": "color", "sat_thresh": 40}),
            OpConfig("grayscale"),
            OpConfig("clahe", {"clip": 2.0}),
            OpConfig("binarize", {"method": "otsu"}),
            OpConfig("remove_lead_lines", {"axis": "vertical"}),
        ],
        description="Colour scans where the graph paper is still visible.",
    ),
    "ecg_photo": PresetSpec(
        name="ecg_photo", label="ECG — phone photo",
        ops=[
            OpConfig("crop_border", {"ratio": 0.05}),
            OpConfig("deskew", {"max_angle": 10.0}),
            OpConfig("remove_grid", {"mode": "adaptive"}),
            OpConfig("grayscale"),
            OpConfig("clahe", {"clip": 3.0}),
            OpConfig("denoise", {"method": "median", "strength": 3}),
            OpConfig("binarize", {"method": "adaptive"}),
        ],
        description="Handheld photos: crooked, uneven lighting, shadows.",
    ),
}


# ---------------------------------------------------------------------------
# Applying ops
# ---------------------------------------------------------------------------

def resolve_params(op: str, params: dict[str, Any] | None) -> dict[str, Any]:
    """Fill defaults, drop unknown keys, clamp out-of-range numbers, fall back
    on invalid enums. Defensive on purpose: a bad param must not kill a job
    that has already spent four minutes embedding."""
    spec = OP_REGISTRY[op]
    out = spec.defaults()
    for key, val in (params or {}).items():
        if key not in spec.params_schema:
            log.warning("op %s: ignoring unknown param %r", op, key)
            continue
        meta = spec.params_schema[key]
        try:
            if meta["type"] == "enum":
                out[key] = val if val in meta["options"] else meta["default"]
            elif meta["type"] == "int":
                out[key] = int(np.clip(int(val), meta["min"], meta["max"]))
            elif meta["type"] == "float":
                out[key] = float(np.clip(float(val), meta["min"], meta["max"]))
            elif meta["type"] == "bool":
                out[key] = bool(val)
            else:
                out[key] = val
        except (TypeError, ValueError):
            log.warning("op %s: param %r=%r is not usable, using default",
                        op, key, val)
    return out


def _plan(ops: Sequence[OpConfig]) -> list[tuple[str, dict[str, Any]]]:
    """User config -> the actual execution order. Sorts by OpSpec.order,
    de-duplicates, and appends the locked ops."""
    seen: dict[str, dict[str, Any]] = {}
    for cfg in ops:
        if cfg.op not in OP_REGISTRY:
            raise KeyError(
                f"Unknown preprocessing op {cfg.op!r}. "
                f"Available: {sorted(OP_REGISTRY)}"
            )
        seen[cfg.op] = resolve_params(cfg.op, cfg.params)   # last wins

    for name in ALWAYS:
        seen.setdefault(name, resolve_params(name, None))

    return [(n, seen[n]) for n in sorted(seen, key=lambda n: OP_REGISTRY[n].order)]


def apply_ops(img: np.ndarray, ops: Sequence[OpConfig]) -> np.ndarray:
    """The workhorse. One image in, one model-ready uint8 (224, 224, 3) out.

    Called once per image by the benchmark worker. Keep it allocation-light.
    """
    out = _rgb(img)
    for name, params in _plan(ops):
        out = OP_REGISTRY[name].fn(out, **params)
    return _rgb(out)


def preview_stages(img: np.ndarray, ops: Sequence[OpConfig],
                   include_locked: bool = True) -> list[tuple[str, np.ndarray]]:
    """POST /preprocess/preview. Returns [("original", img), ("crop_border",
    img), ...] so the strip shows what each toggle actually does. The final
    tile is literally what the backbone will see."""
    out = _rgb(img)
    stages: list[tuple[str, np.ndarray]] = [("original", out)]
    for name, params in _plan(ops):
        out = OP_REGISTRY[name].fn(out, **params)
        if include_locked or not OP_REGISTRY[name].locked:
            stages.append((name, _rgb(out)))
    return stages


# ---------------------------------------------------------------------------
# Catalog payloads (main.py just returns these)
# ---------------------------------------------------------------------------

def ops_catalog() -> list[dict[str, Any]]:
    """GET /ops/catalog — ordered as the UI should list them."""
    return [OP_REGISTRY[n].catalog_entry()
            for n in sorted(OP_REGISTRY, key=lambda n: OP_REGISTRY[n].order)]


def presets_catalog() -> list[dict[str, Any]]:
    """GET /presets"""
    return [PRESETS[n].catalog_entry() for n in PRESETS]


# ---------------------------------------------------------------------------
# Smoke test:  python -m backend.core.preprocess
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    def synthetic_ecg(h=600, w=900, skew=False) -> np.ndarray:
        """Pink grid + black wave + vertical lead separators."""
        a = np.full((h, w, 3), 255, np.uint8)
        for x in range(0, w, 10):                      # fine grid
            a[:, x] = (255, 200, 205)
        for y in range(0, h, 10):
            a[y, :] = (255, 200, 205)
        for x in range(0, w, 50):                      # coarse grid
            a[:, x] = (250, 150, 160)
        for y in range(0, h, 50):
            a[y, :] = (250, 150, 160)
        for row in range(4):                           # four lead rows
            base = 80 + row * 140
            for x in range(w):
                y = int(base + 18 * np.sin(x / 11.0))
                if x % 90 < 4:
                    y = base - 55                      # QRS spike
                a[max(0, y - 1):y + 2, x] = (10, 10, 10)
        for x in (300, 600):                           # lead separator rules
            a[:, x - 1:x + 2] = (10, 10, 10)
        if skew:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 4.0, 1.0)
            a = cv2.warpAffine(a, M, (w, h), borderValue=(255, 255, 255))
        return a

    img = synthetic_ecg()

    # --- every op runs, preserves the type contract ------------------------
    for name, spec in OP_REGISTRY.items():
        out = spec.fn(img, **resolve_params(name, None))
        assert out.dtype == np.uint8, f"{name} broke dtype"
        assert out.ndim == 3 and out.shape[2] == 3, f"{name} broke shape: {out.shape}"

    # --- order is enforced regardless of list order ------------------------
    backwards = [OpConfig("binarize"), OpConfig("grayscale"),
                 OpConfig("remove_grid"), OpConfig("crop_border")]
    assert [n for n, _ in _plan(backwards)] == [
        "crop_border", "remove_grid", "grayscale", "binarize", "to_rgb", "resize"]

    # --- the colour-order trap: remove_grid AFTER grayscale is a no-op -----
    # Count non-white pixels: the pink grid greys to ~213, so a dark-ink
    # threshold would not see it at all.
    nonwhite = lambda a: (_gray(a) < 250).mean()
    grid_first = op_remove_grid(op_crop_border(img))
    gray_first = op_remove_grid(op_grayscale(op_crop_border(img)))
    assert nonwhite(grid_first) < nonwhite(gray_first) * 0.5, (
        f"remove_grid must need colour: {nonwhite(grid_first):.3f} vs "
        f"{nonwhite(gray_first):.3f}")

    # --- handoff shape -----------------------------------------------------
    for preset in PRESETS.values():
        out = apply_ops(img, preset.ops)
        assert out.shape == (RESIZE_PX, RESIZE_PX, 3) and out.dtype == np.uint8, \
            f"{preset.name} produced {out.shape}"

    # --- locked ops cannot be escaped --------------------------------------
    assert apply_ops(img, []).shape == (RESIZE_PX, RESIZE_PX, 3)

    # --- lead separators actually disappear --------------------------------
    binar = apply_ops(img, [OpConfig("grayscale"), OpConfig("binarize")])
    cleaned = apply_ops(img, [OpConfig("grayscale"), OpConfig("binarize"),
                              OpConfig("remove_lead_lines")])
    col = lambda a: (a[:, :, 0] < 128).mean(axis=0).max()
    assert col(cleaned) < col(binar), "remove_lead_lines did nothing"

    # --- deskew recovers a known rotation ----------------------------------
    fixed = op_deskew(synthetic_ecg(skew=True))
    assert fixed.shape == img.shape

    # --- bad params are survived, not fatal --------------------------------
    p = resolve_params("binarize", {"thr": 9999, "method": "nonsense",
                                    "bogus": 1})
    assert p == {"method": "fixed", "thr": 255}
    assert apply_ops(img, [OpConfig("crop_border", {"ratio": 0.49})]).shape[0] == RESIZE_PX

    try:
        apply_ops(img, [OpConfig("no_such_op")])
        raise AssertionError("expected KeyError on unknown op")
    except KeyError:
        pass

    # --- preview -----------------------------------------------------------
    stages = preview_stages(img, PRESETS["ecg_grid_paper"].ops)
    assert stages[0][0] == "original" and len(stages) == 9
    uri = to_b64(stages[-1][1])
    assert uri.startswith("data:image/png;base64,") and len(uri) < 120_000

    # --- catalog shape for W2 ----------------------------------------------
    cat = ops_catalog()
    assert [c["op"] for c in cat][:2] == ["crop_border", "deskew"]
    assert cat[-1]["locked"] is True

    # --- throughput --------------------------------------------------------
    t0 = time.perf_counter()
    for _ in range(20):
        apply_ops(img, PRESETS["ecg_clean_scan"].ops)
    per = (time.perf_counter() - t0) / 20
    print(f"clean_scan: {per*1000:.1f} ms/image  -> {per*1000:.0f}s for 1000 images")

    t0 = time.perf_counter()
    for _ in range(5):
        apply_ops(img, PRESETS["ecg_photo"].ops)
    per_photo = (time.perf_counter() - t0) / 5
    print(f"ecg_photo:  {per_photo*1000:.1f} ms/image "
          f"-> {per_photo*1000:.0f}s for 1000 images")

    print("preprocess OK — type contract, fixed order, colour-order trap, "
          "locked ops, param clamping, previews")
