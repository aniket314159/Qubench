"""
scripts/calibrate.py
====================

Owner: W4. Measures this machine and writes storage/calibration.json, which
estimate.py merges over its built-in defaults.

    python -m scripts.calibrate              # ~3 minutes, quantum + classical
    python -m scripts.calibrate --quick      # ~40 seconds, fewer points
    python -m scripts.calibrate --backbone resnet18   # also time a real CNN

RUN THIS ON THE DEMO MACHINE ON DAY 4. The shipped constants came from one
container; core count, cache size and BLAS build move them by 2-3x. An estimate
that says 15s for a 90s run is worse than showing no estimate at all.

Every fit is reported with its R^2 and the fit is REJECTED if R^2 falls below
MIN_R2, leaving estimate.py on its default for that term. A badly conditioned
fit is worse than a stale constant: it can come out negative, and a negative
coefficient means the cost strip tells users that more qubits are cheaper.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Callable

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Run either way: `python -m scripts.calibrate` or `python scripts/calibrate.py`.
# The second form puts scripts/ on sys.path, not the repo root, so `import
# backend` fails without this. Every other script in here does the same.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from backend.core.encodings import ENCODINGS, get_encoding
from backend.core.estimate import CALIBRATION_FILE, DEFAULT_COSTS, circuit_gates
from backend.core.models_classical import SVCModel
from backend.core.paths import storage_root

# Per-term R^2 floors. The gate exists to reject nonsense -- a negative slope
# would tell users more qubits are CHEAPER -- not to demand a tight fit
# everywhere. Quantum terms are held strictly because they dominate the wall
# clock and extrapolate across three orders of magnitude. Classical terms are
# held loosely: on realistic (separable) data they fit in well under a second,
# so absolute timing noise swamps the n^2 signal, and a 50%-explained fit is
# still far better than a constant calibrated on uniform noise.
MIN_R2_QUANTUM = 0.90
MIN_R2_CLASSICAL = 0.50


def timeit(fn: Callable[[], object], reps: int = 3) -> float:
    """Minimum of `reps`, not the mean. Timing noise on a shared machine is
    one-sided -- something else stealing a core only ever makes a run slower --
    so the minimum is the better estimator of the true cost."""
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def fit_affine(x: list[float], y: list[float], name: str,
               min_r2: float = MIN_R2_QUANTUM) -> tuple[float, float] | None:
    """Least squares for t = a + b*x, with an R^2 gate and a sign check."""
    A = np.column_stack([np.ones(len(x)), np.asarray(x, dtype=float)])
    b = np.asarray(y, dtype=float)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    ss_res = float(((b - A @ coef) ** 2).sum())
    ss_tot = float(((b - b.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    ok = r2 >= min_r2 and coef[1] > 0
    flag = "ok" if ok else ("REJECTED, keeping default" if r2 < min_r2 else
                            "REJECTED, negative slope")
    print(f"  {name:24s} a={coef[0]:11.4e}  b={coef[1]:11.4e}  "
          f"R2={r2:5.3f}  {flag}")
    return (float(coef[0]), float(coef[1])) if ok else None


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def calibrate_sim(quick: bool) -> dict:
    print("\nstatevector simulation   t = a + b * (n * gates * 2^q)")
    encs = ("angle_y", "zz") if quick else ("angle_y", "angle_x2", "zz")
    feats = (4, 8) if quick else (4, 6, 8, 10)
    sizes = (128, 512) if quick else (64, 256, 1024)

    x, y = [], []
    for enc_name in encs:
        spec = get_encoding(enc_name)
        for nf in feats:
            if nf > spec.max_features:
                continue
            q = spec.qubits_for(nf)
            gates = circuit_gates(enc_name, nf)
            dev = qml.device("default.qubit", wires=q)

            @qml.qnode(dev)
            def f(xx):
                spec.template(xx, wires=range(q))
                return qml.state()

            f(np.random.rand(4, nf))                     # warm the cache
            for n in sizes:
                X = np.random.rand(n, nf)
                y.append(timeit(lambda: f(X)))
                x.append(n * gates * (2 ** q))

    fit = fit_affine(x, y, "sim")
    return {"sim_fixed": fit[0], "sim_per_amp_gate": fit[1]} if fit else {}


def calibrate_gram(quick: bool) -> dict:
    print("\nGram matrix              t = a + b * (n^2 * 2^q)")
    x, y = [], []
    for q in ((6, 8) if quick else (4, 6, 8, 10)):
        for n in ((256, 768) if quick else (128, 512, 1024)):
            S = (np.random.rand(n, 2 ** q)
                 + 1j * np.random.rand(n, 2 ** q)).astype(np.complex64)
            y.append(timeit(lambda: np.abs(S @ S.conj().T) ** 2))
            x.append((n ** 2) * (2 ** q))
    fit = fit_affine(x, y, "gram")
    return {"gram_fixed": fit[0], "gram_per": fit[1]} if fit else {}


def calibrate_svc_precomputed(quick: bool) -> dict:
    print("\nprecomputed SVC          t = a + b * n^2")
    x, y = [], []
    for n in ((150, 500) if quick else (100, 250, 500, 800)):
        K = np.random.rand(n, n)
        K = (K + K.T) / 2
        np.fill_diagonal(K, 1.0)
        yy = np.random.randint(0, 4, n)
        y.append(timeit(lambda: SVC(kernel="precomputed", probability=True).fit(K, yy),
                        reps=2))
        x.append(n ** 2)
    fit = fit_affine(x, y, "svc_precomputed", MIN_R2_CLASSICAL)
    return {"svc_pre_fixed": fit[0], "svc_pre_n2": fit[1]} if fit else {}


def calibrate_svc_grid(quick: bool) -> dict:
    """The badly conditioned one.

    Grid search cost is dominated by a fixed grid of ~22 candidates x cv folds
    spread over however many cores sklearn got, so n^2 explains much less of
    the variance than it does elsewhere. Feature count is deliberately held
    FIXED here rather than varied: including it as a second regressor made the
    earlier fit worse, not better, because the real driver is the thread pool.
    """
    print("\nclassical SVC grid       t = a + b * n^2")
    enc = ENCODINGS["angle_y"]
    x, y = [], []
    for n in ((150, 400) if quick else (100, 250, 400, 650)):
        # Separable data, NOT uniform noise. An SVM on random labels makes
        # almost every point a support vector, which is the worst case and
        # roughly 4x slower than real data -- calibrating on it made the
        # estimate 3.8x too high on a real run.
        X, yy = _realistic(n, 8, seed=n)
        m = SVCModel(n_classes=4, n_features=8, encoding=enc)
        y.append(timeit(lambda: m.fit(X, yy), reps=1))
        x.append(n ** 2)
    fit = fit_affine(x, y, "svc_grid", MIN_R2_CLASSICAL)
    return {"svc_grid_fixed": fit[0], "svc_grid_n2": fit[1]} if fit else {}


def calibrate_rf(quick: bool) -> dict:
    print("\nrandom forest            t = a + b * (trees * n * log2 n * N)")
    x, y = [], []
    for n in ((200, 600) if quick else (200, 500, 900)):
        for trees in ((150, 300) if quick else (100, 300, 500)):
            X, yy = _realistic(n, 8, seed=n + trees)
            y.append(timeit(
                lambda: RandomForestClassifier(n_estimators=trees, n_jobs=-1).fit(X, yy),
                reps=1))
            x.append(trees * n * np.log2(n) * 8)
    fit = fit_affine(x, y, "rf", MIN_R2_CLASSICAL)
    return {"rf_fixed": fit[0], "rf_per": fit[1]} if fit else {}


def _realistic(n: int, n_features: int, seed: int = 0):
    """Separable-ish data, the way real PCA output looks."""
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n, n_features=n_features,
        n_informative=max(2, n_features - 2), n_redundant=0,
        n_classes=4, n_clusters_per_class=1, class_sep=2.0, random_state=seed,
    )
    return X.astype(np.float32), y.astype(np.int64)


def calibrate_vqc(quick: bool) -> dict:
    """Cost of ONE optimiser step:  t = a + b * (batch * gates * 2^q).

    Modelled per STEP rather than per sample, which an earlier version got
    wrong by a factor of 3.5. A VQC trains in minibatches, so a 40-epoch run
    over 160 samples at batch 24 is 280 separate QNode invocations, each paying
    fixed autograd tracing overhead. Charging only for total samples simulated
    ignores 280 copies of that overhead, and the overhead is most of the cost
    at realistic batch sizes.
    """
    print("\nVQC optimiser step       t = a + b * (batch * gates * 2^q)")
    x, y = [], []
    combos = (("angle_y", 6, 24),) if quick else (
        ("angle_y", 6, 16), ("angle_y", 8, 32), ("zz", 6, 24), ("zz", 8, 24),
    )
    for enc_name, nf, batch in combos:
        spec = get_encoding(enc_name)
        q = spec.qubits_for(nf)
        layers = 3
        gates = circuit_gates(enc_name, nf) + layers * q * 4
        dev = qml.device("default.qubit", wires=q)

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def c(xx, w):
            spec.template(xx, wires=range(q))
            qml.StronglyEntanglingLayers(w, wires=range(q))
            return [qml.expval(qml.PauliZ(i)) for i in range(q)]

        shape = qml.StronglyEntanglingLayers.shape(layers, q)
        w = pnp.array(np.random.normal(0, 0.1, shape), requires_grad=True)
        W = pnp.array(np.random.normal(0, 0.5, (q, 4)), requires_grad=True)
        bvec = pnp.zeros(4, requires_grad=True)
        Xb = pnp.array(np.random.rand(batch, nf), requires_grad=False)
        yb = pnp.array(np.random.randint(0, 4, batch), requires_grad=False)

        def cost(w_, W_, b_):
            f = qml.math.stack(c(Xb, w_), axis=-1)
            z = f @ W_ + b_
            z = z - qml.math.max(z, axis=1, keepdims=True)
            p = pnp.exp(z) / pnp.sum(pnp.exp(z), axis=1, keepdims=True)
            return -pnp.mean(pnp.log(p[pnp.arange(batch), yb] + 1e-12))

        opt = qml.AdamOptimizer(0.05)
        opt.step_and_cost(cost, w, W, bvec)                  # warm
        y.append(timeit(lambda: opt.step_and_cost(cost, w, W, bvec), reps=2))
        x.append(batch * gates * (2 ** q))

    fit = fit_affine(x, y, "vqc_step")
    return {"vqc_step_fixed": fit[0], "vqc_step_per": fit[1]} if fit else {}


def calibrate_preprocess(dataset_id: str | None, n: int = 24) -> dict:
    """Seconds per image for each preset, on REAL images.

    Skipped without --dataset-id, because synthetic 224x224 arrays are useless
    here: the cost is dominated by JPEG decode and the first resize, so it
    tracks source resolution. A 2200x1700 clinical scan is an order of
    magnitude slower than anything this script could generate.
    """
    if not dataset_id:
        print("\npreprocessing            skipped (pass --dataset-id to measure)")
        return {}

    print(f"\npreprocessing            seconds per image ({n} real images)")
    from backend import store
    from backend.core.ingest import load_stored
    from backend.core.preprocess import PRESETS, apply_ops, load_image

    ds = store.get_dataset(dataset_id)
    if ds is None:
        print(f"  no dataset {dataset_id!r}; skipped")
        return {}

    paths = load_stored(ds)[0][:n]
    raws = [load_image(p) for p in paths]          # decode once, time the ops
    decode = timeit(lambda: [load_image(p) for p in paths[:4]], reps=2) / 4

    out: dict[str, float] = {}
    for name, preset in PRESETS.items():
        dt = timeit(lambda: [apply_ops(im, preset.ops) for im in raws], reps=1)
        out[name] = float(dt / len(raws) + decode)
        print(f"  {name:24s} {out[name] * 1000:8.1f} ms/image")
    print(f"  {'(of which decode)':24s} {decode * 1000:8.1f} ms/image")
    return {"preprocess_per_image": out} if out else {}


def calibrate_embed(backbones: list[str], n: int = 48) -> dict:
    print(f"\nembedding                seconds per image ({n} images)")
    from backend.core.contracts import RunConfig
    from backend.core.embed import embed

    out: dict[str, float] = {}
    imgs = np.random.default_rng(0).integers(
        0, 255, (n, 224, 224, 3), dtype=np.uint8
    )
    for name in backbones:
        cfg = RunConfig(dataset_id="_calib", models=["svc"], backbone=name)
        try:
            embed(cfg, imgs[:4], use_cache=False)         # load weights first
            dt = timeit(lambda: embed(cfg, imgs, use_cache=False), reps=1)
            out[name] = float(dt / n)
            print(f"  {name:24s} {dt / n * 1000:8.2f} ms/image")
        except Exception as e:
            print(f"  {name:24s} skipped: {type(e).__name__}")
    return {"embed_per_image": out} if out else {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fit estimate.py's cost model here")
    ap.add_argument("--quick", action="store_true", help="fewer points, ~40s")
    ap.add_argument("--backbone", action="append", default=None,
                    help="also time this backbone (repeatable)")
    ap.add_argument("--dataset-id", default=None,
                    help="measure preprocessing on this dataset's real images")
    ap.add_argument("--dry-run", action="store_true", help="do not write the file")
    a = ap.parse_args(argv)

    print("=" * 70)
    print("calibrating estimate.py on this machine")
    print("=" * 70)

    measured: dict = {}
    t0 = time.perf_counter()
    measured.update(calibrate_sim(a.quick))
    measured.update(calibrate_gram(a.quick))
    measured.update(calibrate_svc_precomputed(a.quick))
    measured.update(calibrate_svc_grid(a.quick))
    measured.update(calibrate_rf(a.quick))
    measured.update(calibrate_vqc(a.quick))
    measured.update(calibrate_preprocess(a.dataset_id))
    measured.update(calibrate_embed(a.backbone or ["pixels"]))

    print(f"\nmeasured in {time.perf_counter() - t0:.0f}s")
    kept = {k: v for k, v in measured.items()
            if k not in ("embed_per_image", "preprocess_per_image")}
    missing = [k for k in DEFAULT_COSTS if k not in measured
               and k not in ("embed_per_image", "embed_default",
                             "preprocess_per_image", "preprocess_default")]
    print(f"{len(kept)} constants fitted, {len(missing)} left on defaults")
    if missing:
        print(f"  still default: {', '.join(missing)}")

    if a.dry_run:
        print("\n--dry-run, nothing written")
        print(json.dumps(measured, indent=2))
        return 0

    path = storage_root() / CALIBRATION_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(measured, indent=2))
    print(f"\nwrote {path}")
    print("estimate.py picks this up automatically; nothing else to change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
