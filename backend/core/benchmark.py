"""
backend/core/benchmark.py
=========================

Owner: W4. The W3 -> W4 seam, and the only thing W3 imports from the model layer.

    run_benchmark(cfg, images, labels, class_names, on_progress) -> list[dict]
    predict_single(checkpoint_name, image_bgr)                   -> dict

W3 calls these two functions and nothing else. No other symbol in the model
layer is part of the contract, so W4 can restructure everything below this line
without a conversation.


ONE MODEL FAILING MUST NOT KILL THE RUN
---------------------------------------
A VQC that diverges, a qubit count that exhausts memory, a model that throws on
a 2-class dataset: each degrades to a `ModelResult.failed(...)` row and the
remaining models still train. The frontend gets a complete, renderable object
either way, with an `error` string on the row that broke.

This matters on Day 4 more than it looks like it does now. "QSVC and SVC
rendered, VQC shows an error badge" is a demo. An exception propagating out of
the job runner is a blank screen.


PROGRESS IS A CONTRACT, NOT A COURTESY
--------------------------------------
The frontend polls every 800ms and maps `pct` to a bar that must never go
backwards. The anchors come from contracts.STAGE_PCT and TRAINING_SPAN so that
W3's job runner and this loop cannot disagree about what 55% means.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from . import checkpoint as ckpt
from .contracts import (
    TRAINING_SPAN,
    CheckpointBundle,
    Metrics,
    ModelResult,
    PerClassMetrics,
    ProgressFn,
    RunConfig,
    Telemetry,
)
from .embed import embed, get_backbone
from .encodings import get_encoding
from .project import apply_projection, project
from .registry import get_model

__all__ = ["run_benchmark", "run_for_dataset", "predict_single", "evaluate"]


def _noop(pct: int, message: str) -> None:
    pass


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    class_names: list[str],
) -> tuple[Metrics, list[list[int]], dict[str, PerClassMetrics]]:
    """Macro-averaged, with the full label set pinned.

    `labels=range(n_classes)` is load-bearing. Without it sklearn infers the
    label set from what it observed, so a class the model never predicted
    silently vanishes from the confusion matrix and the matrix comes back
    3x3 for a 4-class problem. The frontend renders it against 4 class names
    and every cell is off by one.

    Macro rather than weighted because these datasets are imbalanced and
    weighted averaging lets a model score well by being good at the majority
    class alone.
    """
    n_classes = len(class_names)
    labels = list(range(n_classes))

    metrics = Metrics(
        accuracy=float((y_true == y_pred).mean()),
        f1_macro=float(f1_score(y_true, y_pred, average="macro",
                                labels=labels, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, average="macro",
                                        labels=labels, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average="macro",
                                  labels=labels, zero_division=0)),
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        class_names[i]: PerClassMetrics(
            precision=float(p[i]), recall=float(r[i]),
            f1=float(f[i]), support=int(s[i]),
        )
        for i in range(n_classes)
    }
    return metrics, cm.astype(int).tolist(), per_class


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------

def run_benchmark(
    cfg: RunConfig,
    images: NDArray[np.uint8],
    labels: NDArray[np.int64],
    class_names: list[str],
    on_progress: ProgressFn | None = None,
    run_id: str = "r_local",
) -> list[dict[str, Any]]:
    """Train every model in cfg.models and return the `results` array.

    `images` is uint8[n, H, W, 3] straight out of W3's apply_ops. `labels` is
    already integer-encoded against `class_names`. Checkpoints are saved here,
    not by the caller.
    """
    progress = on_progress or _noop
    labels = np.asarray(labels, dtype=np.int64)
    n_classes = len(class_names)

    if len(images) != len(labels):
        raise ValueError(f"{len(images)} images but {len(labels)} labels")
    if labels.size and labels.max() >= n_classes:
        raise ValueError(
            f"label {labels.max()} out of range for {n_classes} class names"
        )

    encoding = get_encoding(cfg.encoding)

    progress(20, f"Embedding with {get_backbone(cfg.backbone).label}")
    E = embed(cfg, images, on_progress=progress)

    progress(55, f"PCA to {cfg.n_features} features")
    proj = project(cfg, E, labels, encoding=encoding)
    progress(
        59,
        f"{proj.n_train} train / {proj.n_test} test, "
        f"{proj.explained_variance:.1%} variance retained",
    )

    lo, hi = TRAINING_SPAN
    results: list[ModelResult] = []

    for i, model_name in enumerate(cfg.models):
        pct = lo + int((hi - lo) * (i / max(len(cfg.models), 1)))
        label = model_name
        kind = "classical"
        try:
            cls = get_model(model_name)
            label, kind = cls.label, cls.kind
            progress(pct, f"Training {label} ({i + 1}/{len(cfg.models)})")

            model = cls(
                n_classes=n_classes,
                n_features=cfg.n_features,
                encoding=encoding,
                seed=cfg.seed,
                **cfg.params_for(model_name),
            )
            model.fit(proj.X_train, proj.y_train)
            y_pred = model.predict(proj.X_test)

            metrics, cm, per_class = evaluate(proj.y_test, y_pred, class_names)
            telemetry: Telemetry = model.telemetry()

            ck = ckpt.save(
                CheckpointBundle(
                    run_id=run_id,
                    model_name=model_name,
                    class_names=list(class_names),
                    ops=list(cfg.ops),
                    backbone=cfg.backbone,
                    encoding=cfg.encoding,
                    n_features=cfg.n_features,
                    estimator=model,
                    pca=proj.pca,
                    scaler=proj.scaler,
                )
            )

            results.append(
                ModelResult(
                    model=model_name, kind=kind, label=label,
                    metrics=metrics, confusion_matrix=cm,
                    telemetry=telemetry, per_class=per_class, checkpoint=ck,
                )
            )
        except Exception as e:                       # noqa: BLE001 -- deliberate
            # Degrade this row, keep the run. See the module docstring.
            results.append(
                ModelResult.failed(model_name, kind, label, f"{type(e).__name__}: {e}")
            )

    progress(95, "Finalising")
    return [r.to_dict() for r in results]


# ---------------------------------------------------------------------------
# The wrapper W3 calls
# ---------------------------------------------------------------------------

def run_for_dataset(
    cfg: RunConfig,
    ds: Any,
    on_progress: ProgressFn | None = None,
    run_id: str | None = None,
) -> Any:
    """DatasetMeta in, RunRecord out. The only symbol backend/main.py imports.

    `run_benchmark` above stays pure — arrays in, dicts out, no disk, no
    DatasetMeta — so it is testable without an upload. This wrapper is the
    part that touches storage: load the images, replay cfg.ops over them
    (progress 5 -> 20), then hand off.

    Preprocessing lives here rather than in main.py because a route body that
    loops over 928 images with numpy is not a route body.
    """
    from .contracts import ModelResult, RunRecord, STAGE_PCT, new_id
    from .ingest import load_stored
    from .preprocess import apply_ops, load_image

    progress = on_progress or _noop
    run_id = run_id or new_id("r")

    progress(STAGE_PCT["preprocessing"], "Loading images")
    paths, y = load_stored(ds)
    n = len(paths)

    # Size comes from the first processed image, not a hardcoded 224. The
    # locked `resize` op carries a tunable `size` param (64-512) that
    # validate.py accepts, so a config overriding it would otherwise raise
    # "could not broadcast input array" on the second line of the loop.
    first = apply_ops(load_image(paths[0]), cfg.ops)
    images = np.empty((n, *first.shape), dtype=np.uint8)
    images[0] = first
    progress(5, f"Preprocessing 1/{n}")

    for i in range(1, n):
        images[i] = apply_ops(load_image(paths[i]), cfg.ops)
        if i % 25 == 0 or i == n - 1:
            progress(5 + int(15 * (i + 1) / n), f"Preprocessing {i + 1}/{n}")

    results = run_benchmark(
        cfg, images, np.asarray(y, dtype=np.int64), list(ds.class_names),
        on_progress=progress, run_id=run_id,
    )

    return RunRecord(
        run_id=run_id,
        dataset_id=ds.dataset_id,
        dataset_name=ds.name,
        class_names=list(ds.class_names),
        config=cfg,
        results=[ModelResult.from_dict(r) for r in results],
    )


# ---------------------------------------------------------------------------
# Single-image inference (the Diagnose screen)
# ---------------------------------------------------------------------------

def predict_single(
    checkpoint_name: str,
    image: bytes | NDArray[np.uint8],
) -> dict[str, Any]:
    """Raw upload bytes (or a uint8 RGB array) -> {label, confidences, latency_ms}.

    Replays the bundled recipe, never the live config. The ops come from the
    checkpoint precisely so that a user who moved the feature slider after the
    run still gets predictions consistent with the model that was trained.

    THREE THINGS THIS SIGNATURE IS CAREFUL ABOUT
    --------------------------------------------
    1. It takes BYTES. The route has an UploadFile, not an array, and decoding
       belongs on this side of the seam so main.py stays free of image code.
    2. It decodes to RGB, not BGR. Seam 2's contract is uint8 RGB end to end,
       and `remove_grid` identifies the ECG paper by HUE — hand it BGR and the
       pink grid reads as blue, the mask misses, and the grid survives into the
       backbone. Predictions would be wrong with nothing raising.
    3. It passes OpConfig objects to apply_ops, not dicts. `_plan` reads
       `cfg.op` as an attribute; a dict raises AttributeError.
    """
    t0 = time.perf_counter()
    bundle = ckpt.load(checkpoint_name)

    # W3 owns preprocess.py. Import it here rather than at module scope so that
    # the model layer still imports, and pipeline.py still runs, before that
    # file exists.
    try:
        from .preprocess import apply_ops
    except ImportError as e:
        raise ImportError(
            "predict_single needs core/preprocess.py (W3). Until it lands, "
            "benchmark.run_benchmark works but the Diagnose route does not."
        ) from e

    if isinstance(image, (bytes, bytearray, memoryview)):
        import io as _io

        from PIL import Image, ImageOps
        with Image.open(_io.BytesIO(bytes(image))) as im:
            im.load()
            im = ImageOps.exif_transpose(im)          # phone photos of ECGs
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    else:
        rgb = np.asarray(image, dtype=np.uint8)

    img = apply_ops(rgb, bundle.ops)
    batch = img[None, ...].astype(np.uint8)

    cfg = RunConfig(
        dataset_id="_predict",
        models=[bundle.model_name],
        ops=list(bundle.ops),
        backbone=bundle.backbone,
        n_features=bundle.n_features,
        encoding=bundle.encoding,
    )
    # use_cache=False: a one-off image must never be written into the dataset
    # embedding cache, and it must never read a dataset's cached array either
    # (the shape check would reject it, but not writing is clearer).
    E = embed(cfg, batch, use_cache=False)

    X = apply_projection(bundle.pca, bundle.scaler, E)
    proba = bundle.estimator.predict_proba(X)[0]

    order = int(np.argmax(proba))

    # Where this patient's features physically land on each qubit. Uses the
    # model's OWN tuned bandwidth, not a default — the rotation a judge sees
    # has to be the rotation that actually ran.
    try:
        from .encodings import bloch_angles
        angles = bloch_angles(
            bundle.encoding, X[0],
            float(getattr(bundle.estimator, "bandwidth", 1.0) or 1.0),
        )
    except Exception:
        angles = None

    return {
        "label": bundle.class_names[order],
        "confidences": {
            name: float(proba[i]) for i, name in enumerate(bundle.class_names)
        },
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        "bloch_angles": angles,
    }
