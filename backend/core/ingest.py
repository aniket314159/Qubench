"""
core/ingest.py  (backend/core/ingest.py)
========================================

SEAM 1 — ingest adapters. Phase 1 ships exactly one: "image_folder".

    ADAPTERS["image_folder"].load(root) -> (paths, label_strings)

Phase 3 adds "dicom", "npz", "csv_waveform" as new dict entries. Nothing else
in the codebase changes, because everything downstream consumes the
`(paths, labels)` pair, never a directory layout.

The public entry point is `ingest_zip()`. It does the whole POST /datasets job:
extract, infer classes, subsample, copy into storage, render four thumbnails,
and hand back a `DatasetMeta`. It does NOT persist — `main.py` calls
`store.save_dataset()` with the result. Core knows nothing about the store.

Expected zip layout (a wrapper folder is fine, we descend through it):

    anything.zip
    └── ECG_Data/
        ├── Normal/          ├── HB/          ├── MI/          └── PMI/
        │   *.png, *.jpg     │   ...          │   ...              ...
"""

from __future__ import annotations

import base64
import io
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageOps, UnidentifiedImageError

from backend.core.contracts import AdapterSpec, DatasetMeta, new_id
from backend.core.paths import PREVIEW_DIRNAME, dataset_dir, preview_dir

# --- limits ----------------------------------------------------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MAX_SAMPLES = 1000          # stratified ceiling, per the spec
MIN_CLASSES = 2
MIN_PER_CLASS = 10          # never subsample a class below this; an 80:20
                            # split of 8 images gives a 2-image test set
MAX_ZIP_MEMBERS = 50_000
MAX_UNCOMPRESSED_BYTES = 4 * 1024 ** 3
N_PREVIEWS = 4
PREVIEW_PX = 320

# Junk that shows up in every zip anyone makes on a Mac or Windows machine.
_JUNK_NAMES = {"__MACOSX", ".DS_Store", "Thumbs.db", "desktop.ini", PREVIEW_DIRNAME}


class IngestError(ValueError):
    """Bad upload. main.py turns this into a 400 with the message verbatim,
    so every message here must be safe and useful to show a user."""


# ---------------------------------------------------------------------------
# Zip handling
# ---------------------------------------------------------------------------

def _is_junk(part: str) -> bool:
    return part in _JUNK_NAMES or part.startswith(".")


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Zip-slip and zip-bomb guards. A member named `../../etc/passwd` would
    otherwise be written outside dest by extractall()."""
    members = zf.infolist()
    if len(members) > MAX_ZIP_MEMBERS:
        raise IngestError(f"Zip has {len(members)} entries, limit is {MAX_ZIP_MEMBERS}.")

    total = 0
    for m in members:
        name = Path(m.filename)
        if name.is_absolute() or ".." in name.parts:
            raise IngestError(f"Zip contains an unsafe path: {m.filename!r}")
        total += m.file_size
        if total > MAX_UNCOMPRESSED_BYTES:
            raise IngestError("Zip expands to more than 4 GB; refusing to extract.")

    zf.extractall(dest)


def _find_dataset_root(extracted: Path) -> Path:
    """Descend through wrapper folders. `data.zip/ECG_Data/Normal/*.png` should
    behave identically to `data.zip/Normal/*.png`."""
    root = extracted
    for _ in range(4):                      # bounded; nobody nests deeper
        entries = [p for p in root.iterdir() if not _is_junk(p.name)]
        dirs = [p for p in entries if p.is_dir()]
        files = [p for p in entries if p.is_file()]
        if len(dirs) == 1 and not files:
            root = dirs[0]
        else:
            return root
    return root


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

def load_image_folder(root: Path) -> tuple[list[Path], list[str]]:
    """Immediate subdirectories are classes. Images may sit at any depth
    beneath a class folder — some published ECG datasets nest by patient.

    Returns parallel lists. Deterministic: classes sorted, files sorted.
    """
    root = Path(root)
    if not root.is_dir():
        raise IngestError(f"{root} is not a directory.")

    paths: list[Path] = []
    labels: list[str] = []

    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if _is_junk(class_dir.name):
            continue
        found = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
            and not any(_is_junk(part) for part in p.relative_to(class_dir).parts)
        )
        if not found:
            continue                         # empty folder is not a class
        paths.extend(found)
        labels.extend([class_dir.name] * len(found))

    if not paths:
        raise IngestError(
            "No images found. Expected one folder per class at the top level, "
            f"each containing {'/'.join(sorted(IMAGE_EXTS))} files."
        )
    return paths, labels


ADAPTERS: dict[str, AdapterSpec] = {
    "image_folder": AdapterSpec(
        name="image_folder",
        label="Image folder (one directory per class)",
        load=load_image_folder,
    ),
}


# ---------------------------------------------------------------------------
# Stratified subsample
# ---------------------------------------------------------------------------

def allocate(counts: dict[str, int], budget: int,
             min_per_class: int = MIN_PER_CLASS) -> dict[str, int]:
    """How many images to keep per class, capped at `budget` in total.

    Proportional with a floor, then largest-remainder for the leftovers. The
    floor matters: naive proportional allocation on an imbalanced dataset can
    reduce a small class to two or three images and the split then produces an
    empty test class, which makes macro-F1 silently meaningless.
    """
    total = sum(counts.values())
    if total <= budget:
        return dict(counts)

    base = {c: min(n, min_per_class) for c, n in counts.items()}
    if sum(base.values()) >= budget:
        # Pathological: more classes than budget allows even at the floor.
        # Fall back to pure proportional and let validate.py complain later.
        base = {c: 0 for c in counts}

    spare = budget - sum(base.values())
    headroom = {c: counts[c] - base[c] for c in counts}
    pool = sum(headroom.values()) or 1

    exact = {c: headroom[c] * spare / pool for c in counts}
    alloc = {c: base[c] + int(exact[c]) for c in counts}

    # largest remainder, ties broken by class name for determinism
    left = budget - sum(alloc.values())
    order = sorted(counts, key=lambda c: (-(exact[c] % 1), c))
    for c in order:
        if left <= 0:
            break
        if alloc[c] < counts[c]:
            alloc[c] += 1
            left -= 1
    return alloc


def _subsample(paths: Sequence[Path], labels: Sequence[str],
               budget: int, seed: int) -> tuple[list[Path], list[str]]:
    """Deterministic given the same seed and the same input ordering."""
    import random

    by_class: dict[str, list[Path]] = {}
    for p, lab in zip(paths, labels):
        by_class.setdefault(lab, []).append(p)

    counts = {c: len(v) for c, v in by_class.items()}
    alloc = allocate(counts, budget)

    rng = random.Random(seed)
    kept_paths: list[Path] = []
    kept_labels: list[str] = []
    for c in sorted(by_class):
        pool = sorted(by_class[c])
        chosen = pool if alloc[c] >= len(pool) else rng.sample(pool, alloc[c])
        for p in sorted(chosen):
            kept_paths.append(p)
            kept_labels.append(c)
    return kept_paths, kept_labels


# ---------------------------------------------------------------------------
# Writing into storage
# ---------------------------------------------------------------------------

def _open_upright(path: Path) -> Image.Image:
    """EXIF-aware open. The `ecg_photo` preset implies phone photos, and a
    portrait-shot ECG stored with an orientation tag will otherwise arrive
    sideways at the model and nobody will understand why accuracy collapsed."""
    img = Image.open(path)
    img.load()
    return ImageOps.exif_transpose(img)


def _store_images(ds_id: str, paths: Sequence[Path],
                  labels: Sequence[str]) -> tuple[list[Path], list[str]]:
    """Copy the kept images into storage/datasets/{ds_id}/{class}/.

    NOTE — deviation from the layout note in ARCHITECTURE.md, which writes
    `{class}/*.png`. We copy the original bytes and keep the original
    extension instead of re-encoding to PNG. Re-encoding 1000 JPEGs costs
    30-60s inside a synchronous upload request, and buys nothing: seam 2's
    `to_rgb` already normalises mode, and `_open_upright` handles EXIF at read
    time. To switch, this is the only function to change.

    Files that PIL cannot open are dropped here rather than at 20% of a job.
    """
    root = dataset_dir(ds_id)
    kept_paths: list[Path] = []
    kept_labels: list[str] = []

    for i, (src, lab) in enumerate(zip(paths, labels)):
        try:
            with Image.open(src) as probe:
                probe.verify()                  # cheap header check
        except (UnidentifiedImageError, OSError, ValueError):
            continue                            # silently drop the corrupt file

        out_dir = root / lab
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"{i:05d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        kept_paths.append(dst)
        kept_labels.append(lab)

    if not kept_paths:
        raise IngestError("Every file failed to open as an image.")
    return kept_paths, kept_labels


def _save_previews(ds_id: str, paths: Sequence[Path], labels: Sequence[str],
                   class_names: Sequence[str]) -> list[str]:
    """Four thumbnails, spread across classes so the user sees the variety.
    Written to disk AND returned as data URIs, so the upload response can show
    them with no second round trip."""
    out_dir = preview_dir(ds_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    first: dict[str, list[Path]] = {c: [] for c in class_names}
    for p, lab in zip(paths, labels):
        if len(first[lab]) < N_PREVIEWS:
            first[lab].append(p)

    picks: list[tuple[str, Path]] = []
    for rnd in range(N_PREVIEWS):                # one per class, then wrap
        for c in class_names:
            if len(picks) >= N_PREVIEWS:
                break
            if rnd < len(first[c]):
                picks.append((c, first[c][rnd]))
        if len(picks) >= N_PREVIEWS:
            break

    uris: list[str] = []
    for i, (cls, src) in enumerate(picks):
        try:
            img = _open_upright(src).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError):
            continue
        img.thumbnail((PREVIEW_PX, PREVIEW_PX))
        safe = "".join(ch if ch.isalnum() else "_" for ch in cls)[:32]
        img.save(out_dir / f"{i}_{safe}.png", format="PNG")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        uris.append("data:image/png;base64," +
                    base64.b64encode(buf.getvalue()).decode("ascii"))
    return uris


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ingest_zip(
    zip_source: Path | bytes | io.BytesIO,
    name: str,
    modality: str = "ecg",
    adapter: str = "image_folder",
    max_samples: int = MAX_SAMPLES,
    seed: int = 42,
) -> tuple[DatasetMeta, list[str]]:
    """Everything POST /datasets needs. Returns (meta, preview_data_uris).

    Does not persist the meta — that is store.save_dataset()'s job, called by
    the route. Core never imports the store.
    """
    if adapter not in ADAPTERS:
        raise IngestError(
            f"Unknown adapter {adapter!r}. Available: {sorted(ADAPTERS)}"
        )

    src: Path | io.BytesIO
    if isinstance(zip_source, bytes):
        src = io.BytesIO(zip_source)
    else:
        src = zip_source

    with tempfile.TemporaryDirectory(prefix="sih_ingest_") as tmp:
        tmpdir = Path(tmp)
        try:
            with zipfile.ZipFile(src) as zf:
                _safe_extract(zf, tmpdir)
        except zipfile.BadZipFile:
            raise IngestError("That file is not a readable .zip archive.")

        root = _find_dataset_root(tmpdir)
        paths, labels = ADAPTERS[adapter].load(root)

        class_names = sorted(set(labels))
        if len(class_names) < MIN_CLASSES:
            raise IngestError(
                f"Found {len(class_names)} class folder(s) ({class_names}). "
                f"At least {MIN_CLASSES} are required for classification."
            )

        paths, labels = _subsample(paths, labels, max_samples, seed)

        ds_id = new_id("d")
        while dataset_dir(ds_id).exists():
            ds_id = new_id("d")

        try:
            kept_paths, kept_labels = _store_images(ds_id, paths, labels)
            class_names = sorted(set(kept_labels))     # a class may have been
            counts = {c: kept_labels.count(c) for c in class_names}
            if len(class_names) < MIN_CLASSES:         # fully dropped as corrupt
                raise IngestError(
                    "After discarding unreadable files, fewer than "
                    f"{MIN_CLASSES} classes remain."
                )
            previews = _save_previews(ds_id, kept_paths, kept_labels, class_names)
        except Exception:
            shutil.rmtree(dataset_dir(ds_id), ignore_errors=True)
            raise

    meta = DatasetMeta(
        dataset_id=ds_id,
        name=(name or "Untitled dataset").strip()[:120],
        adapter=adapter,
        class_names=class_names,
        counts=counts,
        n_samples=sum(counts.values()),
        n_classes=len(class_names),
        modality=modality,
    )
    return meta, previews


def load_stored(meta: DatasetMeta) -> tuple[list[Path], list[int]]:
    """Read a previously ingested dataset back off disk for a run.

    Returns integer labels, already aligned to `meta.class_names`, because
    this is the last place in the pipeline that knows about folder names.
    Everything after this point sees `y : int[n]`.
    """
    root = dataset_dir(meta.dataset_id)
    paths: list[Path] = []
    y: list[int] = []
    for cls in meta.class_names:
        for p in sorted((root / cls).glob("*")):
            if p.suffix.lower() in IMAGE_EXTS:
                paths.append(p)
                y.append(meta.label_of(cls))
    if not paths:
        raise IngestError(
            f"Dataset {meta.dataset_id} has no images on disk at {root}."
        )
    return paths, y


# ---------------------------------------------------------------------------
# Smoke test:  python -m backend.core.ingest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random as _random

    def _fake_zip(spec: dict[str, int], wrapper: bool = True) -> bytes:
        """Build an in-memory zip with the given class -> count layout."""
        buf = io.BytesIO()
        rng = _random.Random(0)
        with zipfile.ZipFile(buf, "w") as zf:
            for cls, n in spec.items():
                for i in range(n):
                    img = Image.new("RGB", (64, 40),
                                    (rng.randrange(256), 200, 200))
                    b = io.BytesIO()
                    img.save(b, format="PNG")
                    prefix = "ECG_Data/" if wrapper else ""
                    zf.writestr(f"{prefix}{cls}/img_{i}.png", b.getvalue())
            zf.writestr("__MACOSX/._junk", b"junk")
            zf.writestr("ECG_Data/.DS_Store", b"junk")
        return buf.getvalue()

    # --- allocation maths, no I/O ------------------------------------------
    a = allocate({"HB": 233, "MI": 239, "Normal": 284, "PMI": 172}, 1000)
    assert sum(a.values()) == 928 or sum(a.values()) == 1000
    big = allocate({"A": 5000, "B": 40}, 1000)
    assert sum(big.values()) == 1000
    assert big["B"] >= MIN_PER_CLASS, "small class must survive the floor"
    assert big["A"] == 1000 - big["B"]

    # --- happy path --------------------------------------------------------
    meta, previews = ingest_zip(
        _fake_zip({"Normal": 30, "HB": 20, "MI": 25, "PMI": 12}),
        name="ECG Cardiac (synthetic)",
    )
    assert meta.n_classes == 4
    assert meta.class_names == ["HB", "MI", "Normal", "PMI"]   # sorted
    assert meta.counts["Normal"] == 30
    assert meta.n_samples == 87
    assert len(previews) == 4 and previews[0].startswith("data:image/png;base64,")
    assert len(list(preview_dir(meta.dataset_id).glob("*.png"))) == 4

    paths, y = load_stored(meta)
    assert len(paths) == len(y) == 87
    assert set(y) == {0, 1, 2, 3}
    assert y[0] == meta.label_of("HB") == 0

    # --- subsample is capped, stratified and deterministic -----------------
    m2, _ = ingest_zip(_fake_zip({"A": 400, "B": 60}), name="cap", max_samples=100)
    assert m2.n_samples == 100
    assert m2.counts["B"] >= MIN_PER_CLASS

    # --- rejections --------------------------------------------------------
    for bad, why in [
        (_fake_zip({"OnlyOne": 10}), "single class"),
        (b"not a zip at all", "not a zip"),
    ]:
        try:
            ingest_zip(bad, name="bad")
            raise AssertionError(f"expected rejection: {why}")
        except IngestError:
            pass

    evil = io.BytesIO()
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../../escaped.png", b"x")
    try:
        ingest_zip(evil.getvalue(), name="evil")
        raise AssertionError("expected a zip-slip rejection")
    except IngestError as e:
        assert "unsafe path" in str(e)

    # --- cleanup -----------------------------------------------------------
    for m in (meta, m2):
        shutil.rmtree(dataset_dir(m.dataset_id), ignore_errors=True)

    print("ingest OK — wrapper descent, junk skip, floor-aware stratified "
          "subsample, previews, zip-slip guard")
