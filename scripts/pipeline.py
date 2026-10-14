"""
scripts/pipeline.py
===================

Owner: W4. The Day-1 deliverable: the whole pipeline with no web layer.

    python -m scripts.pipeline --dataset data/ecg1 --features 8 \
                               --encoding zz --models svc,rf,qsvc,vqc

    python -m scripts.pipeline --synthetic 4 --models svc,qsvc     # no data needed

Everything this touches is importable by W3 as plain Python, so if the table
prints here it will print through the API too. When it does not, this is where
you debug, not in a FastAPI route.

Three sources, so the same table can be produced from anything:

    --dataset-id d_a1b2   an already-ingested dataset; uses the EXACT API path
                          (ingest.load_stored -> preprocess.apply_ops ->
                          benchmark.run_for_dataset)
    --dataset PATH        a raw folder of class subfolders, ingested inline
    --synthetic N         generated data, no files needed

All three apply a preprocessing preset (seam 2) before embedding, because the
claim at the top -- "if it prints here it prints through the API" -- is only
true if the CLI walks the same pipeline. It bypassed preprocessing entirely in
an earlier version, which made a CLI run and an API run produce different
features from identical images.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Run either way: `python -m scripts.pipeline` or `python scripts/pipeline.py`.
# The second form puts scripts/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.benchmark import run_benchmark, run_for_dataset
from backend.core.contracts import ModelResult, RunConfig
from backend.core.embed import BACKBONES
from backend.core.encodings import ENCODINGS
from backend.core.preprocess import PRESETS, apply_ops, load_image
from backend.core.registry import DEFAULT_MODELS, MODEL_REGISTRY

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_folder(root: Path, size: int = 224, limit: int | None = 400):
    """root/{class_name}/*.png -> (uint8[n,H,W,3] RGB, int[n], class_names).

    Uses preprocess.load_image, NOT cv2.imread. cv2 returns BGR; every other
    path in this project is RGB end to end. Feeding BGR to the backbone gives
    different embeddings for the same file, and `remove_grid` -- which keys on
    the pink paper's hue -- misses entirely. Nothing raises; the numbers are
    just quietly different from the API's.

    Per-class truncation to `limit` so a 3000-image folder does not turn a
    smoke test into a coffee break.
    """
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not classes:
        raise SystemExit(f"no class subdirectories under {root}")

    per_class = None if limit is None else max(limit // len(classes), 2)
    images: list[np.ndarray] = []
    labels: list[int] = []

    for idx, cname in enumerate(classes):
        files = sorted(
            p for p in (root / cname).iterdir()
            if p.suffix.lower() in IMAGE_SUFFIXES
        )
        if per_class is not None:
            files = files[:per_class]
        for p in files:
            try:
                images.append(load_image(p))          # RGB, EXIF-corrected
            except Exception:
                continue                              # unreadable file, skip
            labels.append(idx)

    if not images:
        raise SystemExit(f"no readable images under {root}")
    return images, np.array(labels, dtype=np.int64), classes


def make_synthetic(n_classes: int, n: int = 160, size: int = 224, seed: int = 0):
    """Class-correlated texture on noise. Not ECG, but separable enough that a
    working pipeline scores well above chance and a broken one does not."""
    rng = np.random.default_rng(seed)
    y = np.tile(np.arange(n_classes), n // n_classes + 1)[:n]
    rng.shuffle(y)
    imgs = rng.integers(60, 200, (n, size, size, 3), dtype=np.uint8)
    for i, c in enumerate(y):
        stripe = slice(c * 20, c * 20 + 14)
        imgs[i, stripe, :, :] = np.clip(
            imgs[i, stripe, :, :].astype(int) + 55, 0, 255
        ).astype(np.uint8)
    names = [f"class_{i}" for i in range(n_classes)]
    return list(imgs), y.astype(np.int64), names


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(rows: list[dict], class_names: list[str], wall: float) -> None:
    results = [ModelResult.from_dict(r) for r in rows]

    print(f"\n{'=' * 92}")
    print(f"{'model':24s} {'kind':10s} {'acc':>7s} {'f1':>7s} "
          f"{'prec':>7s} {'recall':>7s} {'fit s':>8s} {'pred s':>8s}")
    print("-" * 92)
    for r in results:
        if not r.ok:
            print(f"{r.label[:24]:24s} {r.kind:10s}   FAILED  {r.error[:44]}")
            continue
        m, t = r.metrics, r.telemetry
        print(f"{r.label[:24]:24s} {r.kind:10s} {m.accuracy:7.3f} {m.f1_macro:7.3f} "
              f"{m.precision:7.3f} {m.recall:7.3f} {t.fit_seconds:8.2f} "
              f"{t.predict_seconds:8.3f}")

    quantum = [r for r in results if r.kind == "quantum" and r.ok]
    if quantum:
        print(f"\n{'QUANTUM TELEMETRY':<24s} {'qubits':>7s} {'encoding':>12s} "
              f"{'depth':>7s} {'2q gates':>9s} {'bandwidth':>10s} "
              f"{'state MB':>9s} {'params':>8s}")
        print("-" * 92)
        for r in quantum:
            t = r.telemetry
            bw = "-" if t.bandwidth is None else f"{t.bandwidth:.3g}"
            print(f"{r.label[:24]:24s} {t.n_qubits:7d} {str(t.encoding):>12s} "
                  f"{t.circuit_depth:7d} {t.two_qubit_gates:9d} {bw:>10s} "
                  f"{t.state_memory_mb:9.2f} {str(t.n_params):>8s}")

    ok = [r for r in results if r.ok]
    best_c = max((r for r in ok if r.kind == "classical"),
                 key=lambda r: r.metrics.accuracy, default=None)
    best_q = max((r for r in ok if r.kind == "quantum"),
                 key=lambda r: r.metrics.accuracy, default=None)

    print(f"\n{'-' * 92}")
    if best_c:
        print(f"best classical  {best_c.label:32s} {best_c.metrics.accuracy:.4f}")
    if best_q:
        print(f"best quantum    {best_q.label:32s} {best_q.metrics.accuracy:.4f}")
    if best_c and best_q:
        d = best_q.metrics.accuracy - best_c.metrics.accuracy
        print(f"delta           {'quantum - classical':32s} {d:+.4f}")
        print()
        print("  This is the number the whole pitch rests on. It is only worth")
        print("  quoting if the classical arm was tuned -- check that 'svc' ran")
        print("  with tune=True, and check which bandwidth qsvc chose.")

    # Confusion matrix of the best overall model.
    best = max(ok, key=lambda r: r.metrics.accuracy, default=None)
    if best and best.confusion_matrix:
        w = max(len(c) for c in class_names)
        print(f"\nconfusion matrix -- {best.label}   (rows = true, cols = predicted)")
        print(" " * (w + 2) + "".join(f"{c[:6]:>7s}" for c in class_names))
        for name, row in zip(class_names, best.confusion_matrix):
            print(f"{name:>{w}s}  " + "".join(f"{v:7d}" for v in row))

    print(f"\ntotal wall time {wall:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="QuCardio pipeline, no web layer")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", type=Path, help="folder of class subfolders")
    src.add_argument("--synthetic", type=int, metavar="N_CLASSES",
                     help="generate a synthetic dataset with N classes")
    src.add_argument("--dataset-id", help="an already-ingested dataset "
                                          "(walks the exact API path)")

    ap.add_argument("--features", type=int, default=8)
    ap.add_argument("--encoding", default="angle_y", choices=sorted(ENCODINGS))
    ap.add_argument("--backbone", default="resnet18", choices=sorted(BACKBONES))
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help=f"comma-separated from {sorted(MODEL_REGISTRY)}")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None,
                    help="preprocessing preset; default ecg_clean_scan for "
                         "real images, raw for --synthetic")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=400,
                    help="max images (stratified). 0 = no limit")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODEL_REGISTRY]
    if unknown:
        ap.error(f"unknown model(s) {unknown}; available: {sorted(MODEL_REGISTRY)}")

    # The API gets this from validate.py before the Run button enables. The CLI
    # has no such gate, so check here rather than let project() raise mid-run.
    enc = ENCODINGS[a.encoding]
    if a.features > enc.max_features:
        ap.error(f"{enc.label} supports at most {enc.max_features} features; "
                 f"--features {a.features} given")
    if any(MODEL_REGISTRY[m].kind == "quantum" for m in models) \
            and enc.qubits_for(a.features) < 2:
        ap.error(f"{a.features} features on {a.encoding} is "
                 f"{enc.qubits_for(a.features)} qubit(s); quantum models need 2+")

    preset = a.preset or ("raw" if a.synthetic else "ecg_clean_scan")
    ops = list(PRESETS[preset].ops)

    stored = None
    if a.dataset_id:
        from backend import store
        stored = store.get_dataset(a.dataset_id)
        if stored is None:
            ap.error(f"no dataset {a.dataset_id!r} in storage. "
                     f"Available: {[d.dataset_id for d in store.list_datasets()]}")
        class_names, dataset_id = list(stored.class_names), stored.dataset_id
        labels = np.zeros(0, dtype=np.int64)          # run_for_dataset loads them
    elif a.synthetic:
        raw, labels, class_names = make_synthetic(a.synthetic, seed=a.seed)
        dataset_id = f"d_synth{a.synthetic}"
    else:
        raw, labels, class_names = load_folder(
            a.dataset, limit=None if a.limit == 0 else a.limit
        )
        dataset_id = f"d_{a.dataset.name}"

    if stored is not None:
        n_images = stored.n_samples
        counts = [stored.counts[c] for c in class_names]
    else:
        n_images = len(raw)
        counts = np.bincount(labels, minlength=len(class_names))

    print(f"dataset      {dataset_id}  {n_images} images, "
          f"{len(class_names)} classes")
    print(f"             " + "  ".join(
        f"{c}={n}" for c, n in zip(class_names, counts)))
    print(f"preprocess   {preset}  [{', '.join(o.op for o in ops) or 'locked ops only'}]")
    print(f"backbone     {a.backbone}")
    print(f"encoding     {a.encoding}  "
          f"{ENCODINGS[a.encoding].readout(a.features)}")
    print(f"models       {', '.join(models)}")

    cfg = RunConfig(
        dataset_id=dataset_id,
        models=models,
        ops=ops,
        preset_name=preset,
        backbone=a.backbone,
        n_features=a.features,
        encoding=a.encoding,
        test_size=a.test_size,
        seed=a.seed,
    )

    def on_progress(pct: int, msg: str) -> None:
        if not a.quiet:
            print(f"  [{pct:3d}%] {msg}")

    t0 = time.perf_counter()
    run_id = f"r_cli_{a.seed}"

    if stored is not None:
        # Identical code path to POST /runs — load, apply ops, embed, train.
        rows = [r.to_dict() for r in
                run_for_dataset(cfg, stored, on_progress, run_id).results]
    else:
        # Same ops, applied here because there is nothing on disk to load.
        on_progress(5, f"Preprocessing {len(raw)} images ({preset})")
        images = np.stack([apply_ops(im, ops) for im in raw])
        rows = run_benchmark(
            cfg, images, labels, class_names,
            on_progress=on_progress, run_id=run_id,
        )
    print_results(rows, class_names, time.perf_counter() - t0)

    return 0 if any(ModelResult.from_dict(r).ok for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
