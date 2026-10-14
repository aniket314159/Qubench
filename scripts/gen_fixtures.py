#!/usr/bin/env python3
"""
scripts/gen_fixtures.py
=======================

Generates fixtures/*.json — one file per endpoint, for W2's MOCK=true mode.

    python scripts/gen_fixtures.py

WHY GENERATE RATHER THAN HAND-WRITE
-----------------------------------
Hand-written fixtures drift from the API within a day, and the drift shows up
as a frontend that works against mocks and breaks against the server. Here:

  * `ops_catalog` and `presets` come straight out of the real OP_REGISTRY.
  * `preview` runs the real `preview_stages()` on a real image, so the base64
    strings are genuine PNGs at the real size.
  * `run_full` builds real dataclasses and serialises them with `.to_dict()`.
  * EVERY fixture is validated against its Pydantic response model from
    `backend/schemas.py` before being written. A fixture that would fail
    FastAPI's own response validation aborts this script.

So after any change to contracts, schemas or the op registry, re-run this and
the mocks stay honest.

Two fixtures are invented rather than derived — `encodings` and
`models_catalog` — because they describe W4's registries, which do not exist
yet. Treat them as the spec W4 implements against: the names, qubit formulas,
max_features and param schemas here are what the frontend will be built for.

Writes nothing to storage/. The dataset and preview fixtures are produced from
a synthetic ECG image in memory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "fixtures"

from backend.core.contracts import (                                # noqa: E402
    DatasetMeta, Estimate, JobState, Metrics, ModelResult, OpConfig,
    PerClassMetrics, Prediction, RunConfig, RunRecord, Telemetry,
    ValidationResult,
)
from backend.core.preprocess import (                               # noqa: E402
    PRESETS, ops_catalog, presets_catalog, preview_stages, to_b64,
)
from backend import schemas                                        # noqa: E402

# The real dataset this project is built around.
CLASS_NAMES = ["HB", "MI", "Normal", "PMI"]
COUNTS = {"HB": 233, "MI": 239, "Normal": 284, "PMI": 172}
DATASET_ID = "d_a1b2c3"
RUN_ID = "r_8f2a91"
TEST_SIZE = 0.2


# ---------------------------------------------------------------------------
# A synthetic ECG, so previews contain real PNG bytes
# ---------------------------------------------------------------------------

def synthetic_ecg(seed: int = 0, h: int = 600, w: int = 900) -> np.ndarray:
    """Pink graph paper, four lead rows, vertical lead separators."""
    rng = np.random.default_rng(seed)
    a = np.full((h, w, 3), 255, np.uint8)
    for x in range(0, w, 10):
        a[:, x] = (255, 200, 205)
    for y in range(0, h, 10):
        a[y, :] = (255, 200, 205)
    for x in range(0, w, 50):
        a[:, x] = (250, 150, 160)
    for y in range(0, h, 50):
        a[y, :] = (250, 150, 160)
    phase = rng.uniform(0, 6.28)
    for row in range(4):
        base = 80 + row * 140
        for x in range(w):
            y = int(base + 18 * np.sin(x / 11.0 + phase))
            if x % 90 < 4:
                y = base - 55
            a[max(0, y - 1):y + 2, x] = (10, 10, 10)
    for x in (300, 600):
        a[:, x - 1:x + 2] = (10, 10, 10)
    return a


# ---------------------------------------------------------------------------
# Confusion matrices that actually agree with their metrics
# ---------------------------------------------------------------------------

def confusion(support: list[int], accuracy: float, seed: int) -> list[list[int]]:
    """Build a [true][pred] matrix whose row sums are exactly `support` and
    whose accuracy is close to `accuracy`.

    Every model is scored on the SAME test set, so every matrix here must sum
    to the same number. Row i sums to support[i], always.

    Misclassification is not uniform — it follows the pattern QuCardio reports
    on this dataset: MI is never confused with anything, and Normal, when it
    is wrong, goes to HB.
    """
    n = len(support)
    cm = [[0] * n for _ in range(n)]
    leaks = {0: [3, 2], 1: [], 2: [0, 3], 3: [0, 2]}    # MI (1) never leaks

    leaking = [i for i in range(n) if leaks[i]]
    total_wrong = int(round(sum(support) * (1 - accuracy)))
    pool = sum(support[i] for i in leaking)

    alloc, left = {}, total_wrong
    for pos, i in enumerate(leaking):
        share = left if pos == len(leaking) - 1 else \
            int(round(total_wrong * support[i] / pool))
        alloc[i] = min(share, support[i])
        left -= alloc[i]

    for i in range(n):
        wrong = alloc.get(i, 0)
        cm[i][i] = support[i] - wrong
        for k in range(wrong):
            cm[i][leaks[i][k % len(leaks[i])]] += 1
    return cm


def metrics_from(cm: list[list[int]]) -> tuple[Metrics, dict[str, PerClassMetrics]]:
    """Derive every number FROM the matrix, so the table and the heatmap in
    the UI can never disagree."""
    m = np.array(cm, dtype=float)
    n = m.shape[0]
    correct, total = float(np.trace(m)), float(m.sum())

    per: dict[str, PerClassMetrics] = {}
    precs, recs, f1s = [], [], []
    for i in range(n):
        tp = m[i, i]
        p = tp / m[:, i].sum() if m[:, i].sum() else 0.0
        r = tp / m[i, :].sum() if m[i, :].sum() else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        precs.append(p); recs.append(r); f1s.append(f)
        per[CLASS_NAMES[i]] = PerClassMetrics(round(p, 4), round(r, 4),
                                              round(f, 4), int(m[i, :].sum()))
    return Metrics(
        accuracy=round(correct / total, 4),
        f1_macro=round(float(np.mean(f1s)), 4),
        precision=round(float(np.mean(precs)), 4),
        recall=round(float(np.mean(recs)), 4),
    ), per


# ---------------------------------------------------------------------------
# Shared objects
# ---------------------------------------------------------------------------

SUPPORT = [int(COUNTS[c] * TEST_SIZE) for c in CLASS_NAMES]      # 46,47,56,34

DATASET = DatasetMeta(
    dataset_id=DATASET_ID,
    name="ECG Cardiac (Multan)",
    adapter="image_folder",
    class_names=CLASS_NAMES,
    counts=COUNTS,
    n_samples=sum(COUNTS.values()),
    n_classes=4,
    modality="ecg",
    created_at="2026-09-11T09:14:00+00:00",
)

CONFIG = RunConfig(
    dataset_id=DATASET_ID,
    models=["svc", "rf", "qsvc", "vqc"],
    ops=PRESETS["ecg_clean_scan"].ops,
    preset_name="ecg_clean_scan",
    backbone="resnet18",
    n_features=8,
    encoding="angle_y",
    test_size=TEST_SIZE,
    seed=42,
)


def model_result(name: str, kind: str, label: str, acc: float, seed: int,
                 *, fit_s: float, pred_s: float, backend: str,
                 n_params: int | None = None, quantum: dict | None = None
                 ) -> ModelResult:
    cm = confusion(SUPPORT, acc, seed)
    metrics, per = metrics_from(cm)
    tel = Telemetry(fit_seconds=fit_s, predict_seconds=pred_s, backend=backend,
                    n_params=n_params, **(quantum or {}))
    return ModelResult(model=name, kind=kind, label=label, metrics=metrics,
                       confusion_matrix=cm, telemetry=tel, per_class=per,
                       checkpoint=f"{RUN_ID}__{name}.pkl")


Q8 = {"n_qubits": 8, "encoding": "angle_y", "circuit_depth": 14,
      "two_qubit_gates": 16, "state_memory_mb": 0.002, "bandwidth": 0.25}

RESULTS = [
    model_result("svc", "classical", "SVC (RBF, tuned)", 0.8333, 1,
                 fit_s=0.31, pred_s=0.01, backend="sklearn", n_params=214),
    model_result("rf", "classical", "Random Forest", 0.8656, 2,
                 fit_s=0.88, pred_s=0.02, backend="sklearn", n_params=300),
    model_result("qsvc", "quantum", "Quantum Kernel SVC", 0.9409, 3,
                 fit_s=18.42, pred_s=4.61, backend="pennylane.default.qubit",
                 n_params=198, quantum=Q8),
    model_result("vqc", "quantum", "Variational Quantum Classifier", 0.9139, 4,
                 fit_s=64.20, pred_s=1.15, backend="pennylane.default.qubit",
                 n_params=132,
                 quantum={**Q8, "circuit_depth": 22, "two_qubit_gates": 24}),
]

RUN = RunRecord(run_id=RUN_ID, dataset_id=DATASET_ID,
                dataset_name=DATASET.name, class_names=CLASS_NAMES,
                config=CONFIG, results=RESULTS,
                created_at="2026-09-11T09:31:00+00:00")


# ---------------------------------------------------------------------------
# W4 registries: use the real ones when they exist, fall back to the spec
# ---------------------------------------------------------------------------

def _w4_encodings() -> tuple[list[dict], str]:
    """Enriched with the real circuit at CONFIG.n_features, so MOCK mode can
    draw the circuit diagram. Without this the fixture is a catalog with no
    ops and the panel renders empty in mock — which would look like a bug in
    the component rather than a gap in the fixture."""
    try:
        from backend.core.encodings import bloch_demo, catalog, circuit_shape
        entries = catalog()
        for e in entries:
            if CONFIG.n_features <= e["max_features"]:
                try:
                    e.update(circuit_shape(e["name"], CONFIG.n_features))
                    e["bloch_demo"] = bloch_demo(e["name"], CONFIG.n_features)
                except Exception:
                    pass
        return entries, "live registry"
    except Exception:
        return SPEC_ENCODINGS, "SPEC (W4 not written)"


def _w4_models() -> tuple[list[dict], str]:
    try:
        from backend.core.registry import catalog
        return catalog(), "live registry"
    except Exception:
        return SPEC_MODELS, "SPEC (W4 not written)"


def _w4_backbones() -> tuple[list[dict], str]:
    try:
        from backend.core.embed import BACKBONES
        return [s.catalog_entry() for s in BACKBONES.values()], "live registry"
    except Exception:
        return SPEC_BACKBONES, "SPEC (W4 not written)"


# The fallbacks below are the SPEC W4 implements against, used only while the
# real registries do not exist. Once they do, the live ones win automatically
# and these stop mattering — do not maintain them in parallel.

SPEC_ENCODINGS = [
    {"name": "angle_y", "label": "Angle (RY)", "max_features": 12,
     "qubit_formula": "N",
     "description": "One qubit per feature. RY rotation, MinMax scaled to "
                    "[0, pi]. The simple baseline."},
    {"name": "zz", "label": "ZZ feature map", "max_features": 10,
     "qubit_formula": "N",
     "description": "Entangling map with ZZ interactions, MinMax scaled to "
                    "[0, 2pi]. Harder to simulate classically."},
    {"name": "amplitude", "label": "Amplitude", "max_features": 64,
     "qubit_formula": "ceil(log2 N)",
     "description": "Packs N features into log2(N) qubits. 16 features fit in "
                    "4 qubits."},
]

SPEC_MODELS = [
    {"name": "svc", "kind": "classical", "label": "SVC (RBF, tuned)",
     "param_schema": {
         "C": {"type": "float", "min": 0.01, "max": 1000.0, "default": 10.0,
               "label": "C (regularisation)"}}},
    {"name": "rf", "kind": "classical", "label": "Random Forest",
     "param_schema": {
         "n_estimators": {"type": "int", "min": 10, "max": 500, "default": 300,
                          "label": "Trees"}}},
    {"name": "qsvc", "kind": "quantum", "label": "Quantum Kernel SVC",
     "param_schema": {}},
    {"name": "vqc", "kind": "quantum", "label": "Variational Quantum Classifier",
     "param_schema": {}},
]

SPEC_BACKBONES = [{"name": "resnet18", "label": "ResNet18 (ImageNet)",
                   "dim": 512, "input_size": 224}]

ENCODINGS, ENC_SOURCE = _w4_encodings()
MODELS, MODEL_SOURCE = _w4_models()
BACKBONES, BB_SOURCE = _w4_backbones()


# ---------------------------------------------------------------------------
# Build every fixture
# ---------------------------------------------------------------------------

def _live_estimate() -> dict[str, Any]:
    """Run the real cost model when it exists, so the mock cost strip shows
    numbers the server would actually produce."""
    try:
        from backend.core.estimate import estimate as real_estimate
        return real_estimate(CONFIG, DATASET.n_samples).to_dict()
    except Exception:
        return Estimate(n_qubits=8, hilbert_dim=256, sims=742,
                        mem_mb=1.5, est_seconds=96.0).to_dict()


def build() -> dict[str, tuple[object, object | None]]:
    """name -> (payload, pydantic model or None). None means a bare shape
    with no response_model, e.g. /health."""
    img = synthetic_ecg(0)
    previews = [to_b64(synthetic_ecg(i), max_px=320) for i in range(4)]
    stages = preview_stages(img, PRESETS["ecg_clean_scan"].ops)

    dataset_out = {**DATASET.to_dict(), "preview_b64": previews}

    # a poll sequence, so MOCK mode can animate the progress bar
    progress = [
        JobState("queued", 0, "Queued"),
        JobState("preprocessing", 5, "Loading 928 images"),
        JobState("preprocessing", 14, "Applying ops (412/928)"),
        JobState("embedding", 20, "ResNet18 forward (0/928)"),
        JobState("embedding", 41, "ResNet18 forward (512/928)"),
        JobState("projecting", 55, "PCA to 8 components"),
        JobState("training", 62, "Training svc (1/4)"),
        JobState("training", 70, "Training rf (2/4)"),
        JobState("training", 79, "Training qsvc (3/4)"),
        JobState("training", 91, "Training vqc (4/4)"),
        JobState("done", 100, "Done", run_id=RUN_ID),
    ]

    return {
        "health": ({"ok": True, "version": "0.1.0", "wired": {
            "encodings": True, "models": True, "backbones": True,
            "estimate": True, "benchmark": True, "predict": True}}, None),

        "dataset": (dataset_out, schemas.DatasetOut),
        "datasets": ([DATASET.to_dict()], schemas.DatasetOut),

        "ops_catalog": (ops_catalog(), schemas.OpCatalogOut),
        "presets": (presets_catalog(), schemas.PresetOut),
        "backbones": (BACKBONES, schemas.BackboneOut),
        "encodings": (ENCODINGS, schemas.EncodingOut),
        "models_catalog": (MODELS, schemas.ModelCatalogOut),

        "preview": ({
            "original_b64": to_b64(stages[0][1]),
            "stages": [{"op": op, "image_b64": to_b64(im)}
                       for op, im in stages[1:]],
        }, schemas.PreviewOut),

        "validate": (ValidationResult(
            errors=[],
            warnings=["Classes are imbalanced (284:172). Compare macro F1 "
                      "rather than accuracy — accuracy flatters the majority "
                      "class."],
        ).to_dict(), schemas.ValidationOut),

        "estimate": (_live_estimate(), schemas.EstimateOut),

        "job_submit": ({"job_id": "j_4c7e21"}, schemas.JobSubmitOut),
        "job_progress": ([s.to_dict() for s in progress], schemas.JobOut),

        "runs": ([RUN.summary()], schemas.RunSummaryOut),
        "run_full": (RUN.to_dict(), schemas.RunOut),

        "leaderboard": ([{
            "dataset_id": DATASET_ID,
            "dataset_name": DATASET.name,
            "n_runs": 3,
            "best_classical": {
                "model": "rf", "label": "Random Forest",
                "accuracy": RESULTS[1].metrics.accuracy,
                "f1_macro": RESULTS[1].metrics.f1_macro,
                "run_id": RUN_ID, "encoding": "angle_y", "n_features": 8},
            "best_quantum": {
                "model": "qsvc", "label": "Quantum Kernel SVC",
                "accuracy": RESULTS[2].metrics.accuracy,
                "f1_macro": RESULTS[2].metrics.f1_macro,
                "run_id": RUN_ID, "encoding": "angle_y", "n_features": 8},
            "delta": round(RESULTS[2].metrics.accuracy
                           - RESULTS[1].metrics.accuracy, 4),
            "last_run_at": RUN.created_at,
        }], schemas.LeaderboardRowOut),

        "predict": (Prediction(
            label="MI",
            confidences={"HB": 0.0214, "MI": 0.9139, "Normal": 0.0403,
                         "PMI": 0.0244},
            latency_ms=142.7,
            # angle_y at 8 features: theta is the scaled feature times the
            # tuned bandwidth, phi is zero. Real shape, illustrative numbers.
            bloch_angles=[{"theta": t, "phi": 0.0} for t in
                          (0.42, 1.86, 2.71, 0.95, 2.28, 1.13, 0.67, 2.94)],
        ).to_dict(), schemas.PredictionOut),
    }


def main() -> int:
    OUT.mkdir(exist_ok=True)
    fixtures = build()
    failures = 0

    print(f"writing {len(fixtures)} fixtures to {OUT.relative_to(ROOT)}/")
    print(f"  encodings      <- {ENC_SOURCE}")
    print(f"  models_catalog <- {MODEL_SOURCE}")
    print(f"  backbones      <- {BB_SOURCE}\n")
    for name, (payload, model) in sorted(fixtures.items()):
        # Validate exactly as FastAPI would on the way out.
        if model is not None:
            try:
                for item in (payload if isinstance(payload, list) else [payload]):
                    model(**item)
            except Exception as exc:
                print(f"  FAIL  {name:<16} {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0][:70]}")
                failures += 1
                continue

        path = OUT / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        kind = "list" if isinstance(payload, list) else "object"
        check = model.__name__ if model else "—"
        print(f"  ok    {name:<16} {path.stat().st_size/1024:7.1f}K  "
              f"{kind:<6} {check}")

    if failures:
        print(f"\n{failures} fixture(s) failed response-model validation.")
        return 1

    print("\nEvery fixture validates against its schemas.py response model.")
    print("Sanity check on run_full.json:")
    for r in RESULTS:
        print(f"  {r.label:<32} acc {r.metrics.accuracy:.4f}  "
              f"f1 {r.metrics.f1_macro:.4f}  "
              f"cm diag {sum(r.confusion_matrix[i][i] for i in range(4))}/"
              f"{sum(map(sum, r.confusion_matrix))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
