"""
backend/core/estimate.py
========================

Owner: W4. Powers POST /runs/estimate -- the cost strip above the Run button.

    8 features -> 8 qubits -> 256-dim Hilbert space
    2,368 simulations, 12 MB peak, ~14s

This is the file that turns the feature slider from a number into a decision.
Without it a user drags to 12 qubits, clicks Run, and stares at a progress bar
for four minutes with no idea whether that is normal. It is also the mechanism
that makes exponential scaling VISIBLE, which is the honest version of the
quantum story: the estimate climbing from 14s to 200s as the slider moves is a
better demonstration of why qubit count matters than any slide.


ACCURACY, HONESTLY
------------------
The constants below were fitted by scripts/calibrate.py on ONE machine. The
quantum terms fit well because statevector simulation is arithmetic with a
clean cost model:

    statevector sims    t = a + b*(n * gates * 2^q)        R2 = 0.996
    Gram matrix         t = a + b*(n^2 * 2^q)              R2 = 0.999
    precomputed SVC     t = a + b*n^2                      R2 = 1.000

    classical SVC grid  t = a + b*n^2                      R2 = 0.999
    random forest       t = a + b*(trees*n*log2 n*N)        R2 = 0.895

The grid-search term only fits once feature count is held FIXED. An earlier
attempt used n^2*N as the regressor and managed R2 = 0.46, because the real
driver is the size of the parameter grid and the thread pool, not the feature
count. scripts/calibrate.py holds N fixed for that reason, and there is a
comment there saying so. Do not quote these constants as a performance result.

Re-run `python -m scripts.calibrate` on the demo machine on Day 4. A laptop with
different core count and cache size will be off by 2-3x otherwise, and an
estimate that says 15s when the run takes 90s is worse than no estimate.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .contracts import Estimate, RunConfig
from .encodings import get_encoding
from .paths import storage_root
from .registry import get_model

__all__ = ["estimate", "breakdown", "load_costs", "DEFAULT_COSTS", "CALIBRATION_FILE"]

CALIBRATION_FILE = "calibration.json"


# ---------------------------------------------------------------------------
# Fitted constants, all seconds
# ---------------------------------------------------------------------------

DEFAULT_COSTS: dict[str, Any] = {
    # statevector simulation:  t = fixed + per_amp_gate * (n * gates * 2^q)
    "sim_fixed": 4.51e-3,
    "sim_per_amp_gate": 5.96e-9,

    # Gram matrix:  t = fixed + per * (n^2 * 2^q)
    "gram_fixed": 3.47e-4,
    "gram_per": 4.47e-11,

    # SVC on a precomputed kernel, probability=True:  t = fixed + per * n^2
    "svc_pre_fixed": 1.66e-3,
    "svc_pre_n2": 3.41e-8,

    # classical SVC grid search:  t = fixed + per * n^2.
    # Calibrated on SEPARABLE data. An earlier version fitted this on uniform
    # noise, where almost every point becomes a support vector; that inflated
    # the coefficient 12x and made the estimate 3.8x too high on a real run.
    # Fits can still come out with a negative intercept on some machines,
    # which is why every estimator clamps at zero.
    "svc_grid_fixed": 0.528,
    "svc_grid_n2": 3.67e-6,

    # random forest:  t = fixed + per * (trees * n * log2 n * n_features)
    "rf_fixed": 1.47e-1,
    "rf_per": 1.70e-8,

    # one autograd step costs this multiple of a forward pass (measured 2.89)
    # one VQC optimiser step:  t = fixed + per * (batch * gates * 2^q).
    # Per STEP, not per sample: a 40-epoch run at batch 24 over 160 samples is
    # 280 separate QNode invocations, and the fixed autograd tracing overhead
    # (~31 ms) dominates the arithmetic at realistic batch sizes. Modelling
    # this per-sample underestimated a real run by 3.5x.
    "vqc_step_fixed": 3.10e-2,
    "vqc_step_per": 3.73e-8,

    # seconds per image, per backbone. 'pixels' measured; the torch backbones
    # are placeholders until calibrate.py runs on a machine that has torch.
    "embed_per_image": {
        "pixels": 1.5e-3,
        "resnet18": 8.0e-3,
        "resnet50": 2.2e-2,
        "resnet50_pool1": 6.0e-3,
    },
    "embed_default": 1.0e-2,

    # Seconds per image for preprocess.apply_ops, per preset. Measured on
    # 2200x1700 ECG scans: the cost is dominated by decode + the first resize,
    # so it scales with SOURCE resolution, not with how many ops are ticked.
    #
    # This term was missing entirely in the first version, and it was the
    # single largest error in the model: on a 296-image run it accounted for
    # ~27s of a 45s wall time while the estimate said 9s. Preprocessing is not
    # a rounding error on real scans — it is usually the biggest line item
    # after embedding.
    "preprocess_per_image": {
        "raw": 6.0e-2,
        "ecg_clean_scan": 9.0e-2,
        "ecg_grid_paper": 1.1e-1,
        "ecg_photo": 1.4e-1,
    },
    "preprocess_default": 9.0e-2,
}


def load_costs() -> dict[str, Any]:
    """Merge storage/calibration.json over the built-in defaults.

    Shallow-merged deliberately: a calibration run that only measured the
    `pixels` backbone should not wipe out the estimates for the others.
    """
    costs = dict(DEFAULT_COSTS)
    path = storage_root() / CALIBRATION_FILE
    if path.exists():
        try:
            measured = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return costs
        per_image = dict(costs["embed_per_image"])
        per_image.update(measured.pop("embed_per_image", {}) or {})
        pre_image = dict(costs["preprocess_per_image"])
        pre_image.update(measured.pop("preprocess_per_image", {}) or {})
        costs.update(measured)
        costs["embed_per_image"] = per_image
        costs["preprocess_per_image"] = pre_image
    return costs


# ---------------------------------------------------------------------------
# Circuit size
# ---------------------------------------------------------------------------

@lru_cache(maxsize=128)
def circuit_gates(encoding_name: str, n_features: int) -> int:
    """Decomposed gate count for one encoding at one feature count.

    Measured off the real circuit rather than duplicated as a formula here.
    A gate-count formula in this file would be a second source of truth that
    silently goes stale the moment someone edits a template in encodings.py.
    Memoised because the Configure screen calls estimate on every slider tick.
    """
    import pennylane as qml

    spec = get_encoding(encoding_name)
    q = spec.qubits_for(n_features)
    x = np.zeros(2 ** q if encoding_name == "amplitude" else n_features)
    if encoding_name == "amplitude":
        x[0] = 1.0

    def probe(xx):
        spec.template(xx, wires=range(q))
        return qml.state()

    try:
        res = qml.specs(
            qml.QNode(probe, qml.device("default.qubit", wires=q)), level="device"
        )(x).resources
        # Amplitude embedding decomposes to a single StatePrep whose real cost
        # is O(2^q), not O(1). Charge it q gates so the estimate does not claim
        # a 10-qubit amplitude run is free.
        return max(int(res.num_gates), q)
    except Exception:
        return max(n_features, 1)


# ---------------------------------------------------------------------------
# Per-model estimates
# ---------------------------------------------------------------------------

@dataclass
class ModelEstimate:
    model: str
    kind: str
    label: str
    sims: int
    seconds: float
    mem_mb: float
    note: str = ""

    def __post_init__(self) -> None:
        # Some fitted intercepts are negative (see svc_grid_fixed), so a small
        # dataset can drive the raw model below zero. A cost strip reading
        # "-0.3s" destroys confidence in every other number on screen.
        self.seconds = max(float(self.seconds), 0.0)


def _quantum_common(cfg: RunConfig, n_train: int, n_test: int) -> tuple[int, int, int]:
    spec = get_encoding(cfg.encoding)
    q = spec.qubits_for(cfg.n_features)
    return q, 2 ** q, circuit_gates(cfg.encoding, cfg.n_features)


def _sim_seconds(c: dict, n: int, gates: int, dim: int) -> float:
    return c["sim_fixed"] + c["sim_per_amp_gate"] * n * gates * dim


def _estimate_qsvc(cfg: RunConfig, n_train: int, n_test: int, c: dict) -> ModelEstimate:
    q, dim, gates = _quantum_common(cfg, n_train, n_test)
    params = cfg.params_for("qsvc")
    tune = bool(params.get("tune", True))
    amplitude = cfg.encoding == "amplitude"

    from .models_quantum import QSVCModel
    n_bw = 1 if (amplitude or not tune) else len(QSVCModel.BANDWIDTH_GRID)
    n_c = 4 if tune else 1
    cv = int(params.get("cv", 5)) if tune else 0

    # Tuning re-simulates the whole training set once per bandwidth candidate,
    # then once more for the final fit, then the test set.
    sims = n_bw * n_train + n_train + n_test
    if not tune and not amplitude:
        sims += n_bw * min(n_train, 128)      # the auto-bandwidth probe

    seconds = _sim_seconds(c, sims, gates, dim)
    seconds += (n_bw + 1) * (c["gram_fixed"] + c["gram_per"] * n_train ** 2 * dim)
    if tune:
        # CV fits are on folds of size n_train*(1 - 1/cv), without Platt scaling.
        fold = n_train * (1 - 1 / max(cv, 2))
        seconds += n_bw * n_c * cv * (c["svc_pre_fixed"] + c["svc_pre_n2"] * fold ** 2)
    seconds += c["svc_pre_fixed"] + c["svc_pre_n2"] * n_train ** 2

    # complex64 statevectors + the float64 Gram matrix, which for large n is
    # the bigger of the two and is the thing that actually OOMs.
    mem = (n_train * dim * 8 + n_train ** 2 * 8) / 1024 ** 2

    note = ""
    if tune and not amplitude:
        note = f"tunes {n_bw} bandwidths x {n_c} C over {cv} folds"
    return ModelEstimate("qsvc", "quantum", "Quantum Kernel SVC",
                         sims, seconds, mem, note)


def _estimate_vqc(cfg: RunConfig, n_train: int, n_test: int, c: dict) -> ModelEstimate:
    q, dim, gates = _quantum_common(cfg, n_train, n_test)
    params = cfg.params_for("vqc")
    epochs = int(params.get("epochs", 40))
    layers = int(params.get("n_layers", 3))
    amplitude = cfg.encoding == "amplitude"

    # StronglyEntanglingLayers is 3 rotations + 1 CNOT per wire per layer.
    total_gates = gates + layers * q * 4

    from .models_quantum import QSVCModel
    probe = 0 if amplitude else len(QSVCModel.BANDWIDTH_GRID) * min(n_train, 128)

    # Per optimiser STEP. Batches smaller than 2 are skipped by the training
    # loop, so a ragged final batch of 1 costs nothing.
    batch = int(params.get("batch_size", 24))
    steps_per_epoch = max(n_train // batch, 1)
    steps = epochs * steps_per_epoch
    train_sims = steps * batch
    sims = probe + train_sims + n_test

    seconds = _sim_seconds(c, probe, gates, dim)
    seconds += steps * (
        c["vqc_step_fixed"] + c["vqc_step_per"] * batch * total_gates * dim
    )
    seconds += _sim_seconds(c, n_test, total_gates, dim)

    mem = (batch * dim * 16 * 4) / 1024 ** 2
    return ModelEstimate("vqc", "quantum", "Variational Quantum Classifier",
                         sims, seconds, mem,
                         f"{epochs} epochs x {layers} layers, {steps} steps")


def _estimate_svc(cfg: RunConfig, n_train: int, n_test: int, c: dict) -> ModelEstimate:
    tune = bool(cfg.params_for("svc").get("tune", True))
    if tune:
        seconds = c["svc_grid_fixed"] + c["svc_grid_n2"] * n_train ** 2
    else:
        seconds = c["svc_pre_fixed"] + c["svc_pre_n2"] * n_train ** 2
    return ModelEstimate("svc", "classical", "SVC (RBF, tuned)", 0, seconds,
                         n_train ** 2 * 8 / 1024 ** 2,
                         "grid search" if tune else "single fit")


def _estimate_rf(cfg: RunConfig, n_train: int, n_test: int, c: dict) -> ModelEstimate:
    trees = int(cfg.params_for("rf").get("n_estimators", 300))
    work = trees * n_train * math.log2(max(n_train, 2)) * cfg.n_features
    return ModelEstimate("rf", "classical", "Random Forest", 0,
                         c["rf_fixed"] + c["rf_per"] * work, 8.0,
                         f"{trees} trees")


_ESTIMATORS = {
    "qsvc": _estimate_qsvc,
    "vqc": _estimate_vqc,
    "svc": _estimate_svc,
    "rf": _estimate_rf,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def breakdown(
    cfg: RunConfig, n_samples: int, costs: dict[str, Any] | None = None
) -> list[ModelEstimate]:
    """Per-model rows. Renders as the tooltip behind the cost strip, and tells
    a user WHICH model is the expensive one, which the flat Estimate cannot."""
    c = costs or load_costs()
    n_test = max(int(round(n_samples * cfg.test_size)), 1)
    n_train = max(n_samples - n_test, 1)

    rows: list[ModelEstimate] = []
    for name in cfg.models:
        fn = _ESTIMATORS.get(name)
        if fn is None:
            # An unknown model must not break the estimate endpoint. The run
            # itself will report it properly as a failed row.
            try:
                cls = get_model(name)
                label, kind = cls.label, cls.kind
            except KeyError:
                label, kind = name, "classical"
            rows.append(ModelEstimate(name, kind, label, 0, 0.0, 0.0, "not modelled"))
            continue
        rows.append(fn(cfg, n_train, n_test, c))
    return rows


def estimate(
    cfg: RunConfig, n_samples: int, costs: dict[str, Any] | None = None
) -> Estimate:
    """POST /runs/estimate.

    `sims` and `est_seconds` sum across models; `mem_mb` takes the max, because
    models train one after another and peak memory is the worst single model,
    not the total. Embedding is added once and only when the cache will miss --
    but this function cannot see the cache, so it always charges for it. An
    estimate that is too high on a cache hit is a pleasant surprise; the reverse
    is not.
    """
    c = costs or load_costs()
    spec = get_encoding(cfg.encoding)
    q = spec.qubits_for(cfg.n_features)
    rows = breakdown(cfg, n_samples, c)

    embed_s = n_samples * c["embed_per_image"].get(cfg.backbone, c["embed_default"])

    # Charged even on a cache hit: the embedding cache stores features, not
    # preprocessed images, so apply_ops runs on every single run.
    pre_s = n_samples * c["preprocess_per_image"].get(
        cfg.preset_name or "", c["preprocess_default"]
    )

    return Estimate(
        n_qubits=q,
        hilbert_dim=2 ** q,
        sims=sum(r.sims for r in rows),
        mem_mb=round(max([r.mem_mb for r in rows] + [0.0]), 2),
        est_seconds=round(pre_s + embed_s + sum(r.seconds for r in rows), 1),
    )


# ---------------------------------------------------------------------------
# Smoke test:  python -m backend.core.estimate
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Every model must be estimable from DEFAULT_COSTS alone. Without this, a
    # renamed constant only fails on a machine that has never run calibrate.py
    # -- which is every machine except the one that developed it.
    for _name in _ESTIMATORS:
        _e = estimate(RunConfig("d", [_name], backbone="pixels"), 300,
                      costs=dict(DEFAULT_COSTS))
        assert _e.est_seconds >= 0, _name
    print("every model estimable from DEFAULT_COSTS alone\n")

    n = 928                       # the ECG dataset, all four classes

    print(f"n_samples = {n}, backbone = pixels, models = svc,rf,qsvc\n")
    print(f"{'enc':10s} {'N':>3s} {'q':>3s} {'dim':>7s} {'sims':>8s} "
          f"{'mem MB':>8s} {'est s':>8s}")
    print("-" * 56)
    prev = {}
    for enc in ("angle_y", "zz", "amplitude"):
        for nf in (4, 8, 10, 12):
            if nf > get_encoding(enc).max_features:
                continue
            cfg = RunConfig(dataset_id="d", models=["svc", "rf", "qsvc"],
                            backbone="pixels", n_features=nf, encoding=enc)
            e = estimate(cfg, n)
            print(f"{enc:10s} {nf:3d} {e.n_qubits:3d} {e.hilbert_dim:7d} "
                  f"{e.sims:8d} {e.mem_mb:8.1f} {e.est_seconds:8.1f}")
            prev[(enc, nf)] = e

    # Cost must climb with qubit count, or the strip is not teaching anything.
    assert prev[("angle_y", 12)].est_seconds > prev[("angle_y", 4)].est_seconds
    assert prev[("amplitude", 12)].n_qubits < prev[("angle_y", 12)].n_qubits

    # More models can only cost more.
    base = estimate(RunConfig("d", ["qsvc"], backbone="pixels"), n)
    more = estimate(RunConfig("d", ["qsvc", "vqc"], backbone="pixels"), n)
    assert more.est_seconds > base.est_seconds and more.sims > base.sims

    # Turning tuning off must be cheaper, and must be reflected.
    tuned = estimate(RunConfig("d", ["qsvc"], backbone="pixels"), n)
    plain = estimate(
        RunConfig("d", ["qsvc"], backbone="pixels",
                  model_params={"qsvc": {"tune": False}}), n
    )
    assert plain.est_seconds < tuned.est_seconds, (plain, tuned)

    print()
    for r in breakdown(RunConfig("d", ["svc", "rf", "qsvc", "vqc"],
                                 backbone="pixels", n_features=8,
                                 encoding="zz"), n):
        print(f"  {r.label[:30]:30s} {r.seconds:7.1f}s  {r.sims:7d} sims  "
              f"{r.mem_mb:7.1f} MB  {r.note}")

    print("\nestimate OK")
