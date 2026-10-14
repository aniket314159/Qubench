"""
backend/core/embed.py  --  SEAM 3
=================================

Owner: W4.  images -> float32[n, dim]

    uint8[n, 224, 224, 3]  (from W3's apply_ops)
            |
        backbone forward, batched
            |
    float32[n, dim]        (cached to disk, consumed by project.py)

This is the ONLY file in the model layer that imports torch, and it imports it
lazily, inside `build`. `import backend.core.embed` works on a machine with no
deep-learning stack; only calling a torch-backed backbone requires one. That
keeps `registry.py`, `encodings.py` and both model modules installable with
nothing but numpy, scikit-learn and pennylane.


THE CACHE IS THE REASON THE CONFIGURE SCREEN FEELS FAST
--------------------------------------------------------
`RunConfig.embed_key()` is `{dataset_id}__{backbone}__{ops_hash}.npy`. It
deliberately excludes `n_features` and `encoding`:

    change n_features or encoding  -> cache HIT,  run takes seconds
    change a preprocessing op      -> cache MISS, full forward pass

Users fiddle with the feature slider and the encoding dropdown constantly and
touch the op list rarely, so this is the boundary that matters. The key is
computed in contracts.py, not here, so W3 and W4 cannot disagree about it.


ON `resnet50_pool1`
-------------------
The existing notebook takes features from `pool1_pool` -- the output of the
FIRST ResNet block -- at 340x340 input, then flattens. That is 85*85*64 =
462,400 dimensions per image. For 928 images that is a 1.7 GB float32 matrix
before PCA even starts, which is why the notebook needed TruncatedSVD rather
than PCA and why it does not survive being turned into a web service.

It is kept here, globally average-pooled to 64 dims, as `resnet50_pool1`. That
preserves what is actually interesting about the choice (early-layer edge and
stroke features, which for a black-line-on-white ECG trace may genuinely beat
semantic features) without the 1.7 GB. It is NOT a bit-exact reproduction of
the notebook and should not be presented as one. If the 64-dim version scores
close to the notebook's number, the flatten was buying nothing and you have an
ablation row. If it scores much worse, that is worth knowing before Day 4.


ON `pixels`
-----------
Downsampled raw grayscale, no network at all. It exists for the same reason the
`raw` preprocessing preset exists: "with a pretrained CNN 96%, with raw pixels
71%" is a benchmark result that justifies the embedding stage on one slide.
It also means the entire pipeline is testable, and CI-runnable, without torch.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from .contracts import BackboneSpec, RunConfig
from .paths import artifacts_dir

__all__ = ["BACKBONES", "embed", "get_backbone", "catalog", "clear_cache"]


# ---------------------------------------------------------------------------
# ImageNet normalisation, shared by every torchvision backbone
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_tensor_batch(images: NDArray[np.uint8], size: int):
    """uint8[n, H, W, 3] -> torch float32[n, 3, size, size], ImageNet-normalised.

    W3 already resizes to 224 in the op pipeline, but a backbone may want a
    different input size, so resize defensively rather than trusting it.
    """
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(np.ascontiguousarray(images)).float().div_(255.0)
    x = x.permute(0, 3, 1, 2)                       # NHWC -> NCHW
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def _torchvision_forward(weights_name: str, truncate_after: str | None, size: int):
    """Return a callable batch -> float32[n, dim].

    `truncate_after` selects a named child of the ResNet to stop at; None means
    the standard globally-pooled penultimate features. Either way the output is
    globally average-pooled to a fixed width, so `dim` does not depend on input
    resolution.
    """

    def build() -> Callable[[NDArray[np.uint8]], NDArray[np.float32]]:
        import torch
        import torchvision

        ctor = getattr(torchvision.models, weights_name)
        # weights="DEFAULT" is the documented string every torchvision builder
        # accepts since 0.13; there is no need to resolve the enum by name.
        #
        # The name-mangled lookup this replaced was
        #     getattr(models, f"{weights_name.capitalize()}_Weights", None)
        # which produces "Resnet18_Weights". The real symbol is
        # "ResNet18_Weights" — capital N. getattr therefore returned None, the
        # builder was called with weights=None, and the backbone ran with
        # RANDOM weights. No error, no warning: just an ImageNet feature
        # extractor that had never seen ImageNet, and accuracy nobody could
        # explain.
        net = ctor(weights="DEFAULT")
        net.eval()

        if not any(p.requires_grad is not None and p.abs().sum() > 0
                   for p in net.parameters()):
            raise RuntimeError(f"{weights_name}: weights failed to load")

        if truncate_after is not None:
            keep, children = [], dict(net.named_children())
            for name, mod in children.items():
                keep.append(mod)
                if name == truncate_after:
                    break
            body = torch.nn.Sequential(*keep)
        else:
            body = torch.nn.Sequential(*list(net.children())[:-1])   # drop the fc

        @torch.inference_mode()
        def forward(batch: NDArray[np.uint8]) -> NDArray[np.float32]:
            x = _to_tensor_batch(batch, size)
            out = body(x)
            if out.ndim == 4:
                out = torch.nn.functional.adaptive_avg_pool2d(out, 1)
            return out.flatten(1).cpu().numpy().astype(np.float32)

        return forward

    return build


def _pixels_build() -> Callable[[NDArray[np.uint8]], NDArray[np.float32]]:
    """16x16 grayscale, flattened to 256. No network, no torch."""

    def forward(batch: NDArray[np.uint8]) -> NDArray[np.float32]:
        x = batch.astype(np.float32).mean(axis=-1)            # grayscale
        n, h, w = x.shape
        side = 16
        # Block-mean downsample. Trim to a multiple of `side` first so the
        # reshape is exact rather than silently dropping a ragged edge.
        hh, ww = (h // side) * side, (w // side) * side
        x = x[:, :hh, :ww]
        x = x.reshape(n, side, hh // side, side, ww // side).mean(axis=(2, 4))
        return (x.reshape(n, side * side) / 255.0).astype(np.float32)

    return forward


# ---------------------------------------------------------------------------
# THE REGISTRY
# ---------------------------------------------------------------------------

BACKBONES: dict[str, BackboneSpec] = {
    "resnet18": BackboneSpec(
        name="resnet18",
        label="ResNet18 (ImageNet)",
        dim=512,
        input_size=224,
        build=_torchvision_forward("resnet18", None, 224),
    ),
    "resnet50": BackboneSpec(
        name="resnet50",
        label="ResNet50 (ImageNet)",
        dim=2048,
        input_size=224,
        build=_torchvision_forward("resnet50", None, 224),
    ),
    "resnet50_pool1": BackboneSpec(
        name="resnet50_pool1",
        label="ResNet50 early block (edge features)",
        dim=64,
        input_size=224,
        build=_torchvision_forward("resnet50", "maxpool", 224),
    ),
    "pixels": BackboneSpec(
        name="pixels",
        label="Raw pixels 16x16 (no CNN)",
        dim=256,
        input_size=224,
        build=_pixels_build,
    ),
}

# Built backbones are cached per process: loading ResNet50 weights takes seconds
# and benchmark.py may embed twice in one job (train set, then a preview).
_BUILT: dict[str, Callable[[NDArray[np.uint8]], NDArray[np.float32]]] = {}


def get_backbone(name: str) -> BackboneSpec:
    try:
        return BACKBONES[name]
    except KeyError:
        raise KeyError(
            f"unknown backbone {name!r}; available: {sorted(BACKBONES)}"
        ) from None


def _forward_fn(name: str):
    if name not in _BUILT:
        try:
            _BUILT[name] = get_backbone(name).build()
        except ImportError as e:
            raise ImportError(
                f"backbone '{name}' needs torch and torchvision "
                f"(pip install torch torchvision). The 'pixels' backbone runs "
                f"without them. Original error: {e}"
            ) from e
    return _BUILT[name]


def catalog() -> list[dict]:
    """GET /backbones. Handed to W3, rendered by W2."""
    return [b.catalog_entry() for b in BACKBONES.values()]


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------

def embed(
    cfg: RunConfig,
    images: NDArray[np.uint8],
    on_progress: Callable[[int, str], None] | None = None,
    batch_size: int = 32,
    use_cache: bool = True,
) -> NDArray[np.float32]:
    """uint8[n, H, W, 3] -> float32[n, dim], via disk cache when possible."""
    spec = get_backbone(cfg.backbone)
    path = artifacts_dir() / cfg.embed_key()

    if use_cache and path.exists():
        E = np.load(path)
        # A stale cache entry is worse than no cache: it silently trains on the
        # wrong features. The key covers dataset/backbone/ops, so a mismatch
        # here means something outside the key changed, and the answer is to
        # recompute rather than to trust it.
        if E.shape == (len(images), spec.dim):
            if on_progress:
                on_progress(55, f"Embeddings loaded from cache ({spec.label})")
            return E.astype(np.float32, copy=False)
        path.unlink(missing_ok=True)

    forward = _forward_fn(cfg.backbone)
    n = len(images)
    out = np.empty((n, spec.dim), dtype=np.float32)

    for i in range(0, n, batch_size):
        chunk = forward(images[i : i + batch_size])
        if chunk.shape[1] != spec.dim:
            raise ValueError(
                f"backbone '{spec.name}' declares dim={spec.dim} but returned "
                f"{chunk.shape[1]}. Fix BackboneSpec.dim -- the catalog "
                f"endpoint and the cache validity check both rely on it."
            )
        out[i : i + batch_size] = chunk
        if on_progress:
            pct = 20 + int(35 * min(i + batch_size, n) / n)      # 20 -> 55
            on_progress(pct, f"Embedding {min(i + batch_size, n)}/{n}")

    if use_cache:
        np.save(path, out)
    return out


def clear_cache(dataset_id: str | None = None) -> int:
    """Delete cached embeddings. Returns how many files went.

    Needed on Day 4: if W3 changes an op's implementation without changing its
    name or params, `ops_hash` does not move and every cache entry silently
    becomes wrong. This is the escape hatch.
    """
    removed = 0
    for p in artifacts_dir().glob("*.npy"):
        if dataset_id is None or p.name.startswith(f"{dataset_id}__"):
            p.unlink()
            removed += 1
    return removed


# ---------------------------------------------------------------------------
# Smoke test:  python -m backend.core.embed
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["QUCARDIO_STORAGE"] = tmp
        rng = np.random.default_rng(0)
        imgs = rng.integers(0, 255, (12, 224, 224, 3), dtype=np.uint8)
        cfg = RunConfig(dataset_id="d_test", models=["svc"], backbone="pixels")

        E = embed(cfg, imgs)
        assert E.shape == (12, 256) and E.dtype == np.float32, E.shape
        assert (artifacts_dir() / cfg.embed_key()).exists(), "cache not written"

        E2 = embed(cfg, imgs)                      # must hit the cache
        assert np.array_equal(E, E2)

        # Changing n_features must NOT invalidate the cache.
        cfg_b = RunConfig(dataset_id="d_test", models=["svc"],
                          backbone="pixels", n_features=12)
        assert cfg_b.embed_key() == cfg.embed_key()

        # Changing ops MUST invalidate it.
        from .contracts import OpConfig
        cfg_c = RunConfig(dataset_id="d_test", models=["svc"], backbone="pixels",
                          ops=[OpConfig("grayscale")])
        assert cfg_c.embed_key() != cfg.embed_key()

        assert clear_cache("d_test") == 1
        print(f"backbones: {', '.join(BACKBONES)}")
        print("cache hit on n_features change, miss on ops change -- asserted")
        print("embed OK (pixels backbone; torch backbones untested here)")
