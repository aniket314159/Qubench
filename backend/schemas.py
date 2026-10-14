"""
backend/schemas.py
==================

The HTTP boundary. Pydantic mirrors of `core/contracts.py`.

Division of labour, and it matters:

    schemas.py    SHAPE and TYPE.  "n_features must be an integer."
                  Violations produce a 422 from FastAPI before any code runs.

    validate.py   RULES and MEANING.  "n_features must be <= 10 for zz."
                  Violations produce a friendly message in the amber banner.

Keep bounds here deliberately loose. A 422 is a raw framework error with a
field path in it; the Configure screen's ValidationBanner is where a user
should learn that 12 features is too many for ZZ. Only reject here what would
crash parsing.

Routes return `dataclass.to_dict()` and declare `response_model=`. FastAPI
validates the dict on the way out, so a response model that has drifted from
its dataclass shows up immediately in /docs instead of as a missing column in
W2's table three days later.

THE DRIFT GUARD
---------------
Every mirrored model is checked field-for-field against its dataclass in the
smoke test at the bottom. Add a field to `Telemetry` and forget it here, and
`python -m backend.schemas` fails. That test is the only reason this file is
safe to have.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.core.contracts import RunConfig as RunConfigDC

# Requests forbid unknown fields. During Day 4 integration a typo'd key from
# the frontend becomes an immediate 422 naming the field, instead of being
# silently dropped and debugged as "why is my seed always 42".
_REQ = ConfigDict(extra="forbid")


# ===========================================================================
# Requests
# ===========================================================================

class OpConfigIn(BaseModel):
    model_config = _REQ
    op: str
    params: dict[str, Any] = Field(default_factory=dict)


class RunConfigIn(BaseModel):
    """POST /runs, /runs/validate, /runs/estimate — all three take this."""
    model_config = _REQ

    dataset_id: str
    models: list[str] = Field(default_factory=list)
    ops: list[OpConfigIn] = Field(default_factory=list)
    preset_name: str | None = None
    backbone: str = "resnet18"
    n_features: int = Field(default=8, ge=1)
    encoding: str = "angle_y"
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    seed: int = 42
    model_params: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def to_contract(self) -> RunConfigDC:
        """The one conversion into core. Everything below this line is
        dataclasses; Pydantic does not cross into core/."""
        return RunConfigDC.from_dict(self.model_dump())


class PreviewIn(BaseModel):
    """POST /preprocess/preview"""
    model_config = _REQ
    dataset_id: str
    ops: list[OpConfigIn] = Field(default_factory=list)
    image_index: int = Field(default=0, ge=0)


class DatasetCreateIn(BaseModel):
    """Multipart companion fields for POST /datasets. The zip itself arrives
    as an UploadFile, not through this model."""
    model_config = _REQ
    name: str | None = None
    modality: str = "ecg"
    adapter: str = "image_folder"


# ===========================================================================
# Responses — dataclass mirrors
# ===========================================================================

class DatasetOut(BaseModel):
    dataset_id: str
    name: str
    adapter: str
    class_names: list[str]
    counts: dict[str, int]
    n_samples: int
    n_classes: int
    modality: str
    created_at: str
    # Response-only. Populated on POST /datasets, empty on GET /datasets —
    # the list endpoint must not ship four base64 images per row.
    preview_b64: list[str] = Field(default_factory=list)


class OpConfigOut(BaseModel):
    op: str
    params: dict[str, Any] = Field(default_factory=dict)


class RunConfigOut(BaseModel):
    dataset_id: str
    models: list[str]
    ops: list[OpConfigOut]
    preset_name: str | None
    backbone: str
    n_features: int
    encoding: str
    test_size: float
    seed: int
    model_params: dict[str, dict[str, Any]]


class MetricsOut(BaseModel):
    accuracy: float
    f1_macro: float
    precision: float
    recall: float


class PerClassOut(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class TelemetryOut(BaseModel):
    """One key set for both kinds. The quantum fields are null for classical
    rows — they are NOT optional-and-absent. W2 renders one table off this."""
    fit_seconds: float
    predict_seconds: float
    backend: str
    n_params: int | None = None
    n_qubits: int | None = None
    encoding: str | None = None
    circuit_depth: int | None = None
    two_qubit_gates: int | None = None
    state_memory_mb: float | None = None
    bandwidth: float | None = None


class ModelResultOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())   # allow a `model` field

    model: str
    kind: Literal["classical", "quantum"]
    label: str
    metrics: MetricsOut
    confusion_matrix: list[list[int]]
    telemetry: TelemetryOut
    per_class: dict[str, PerClassOut] = Field(default_factory=dict)
    checkpoint: str | None = None
    error: str | None = None


class RunOut(BaseModel):
    """GET /runs/{run_id}"""
    run_id: str
    dataset_id: str
    dataset_name: str
    class_names: list[str]
    config: RunConfigOut
    results: list[ModelResultOut] = Field(default_factory=list)
    created_at: str


class BestOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model: str
    label: str
    accuracy: float


class RunSummaryOut(BaseModel):
    """GET /runs — the list view. Deliberately not the full RunOut; a list of
    twenty runs with every confusion matrix is a slow screen."""
    run_id: str
    dataset_id: str
    dataset_name: str
    created_at: str
    n_features: int
    encoding: str
    models: list[str]
    best_classical: BestOut | None = None
    best_quantum: BestOut | None = None
    delta: float | None = None


class JobOut(BaseModel):
    """GET /jobs/{job_id} — polled every 800ms, so keep it small."""
    status: Literal["queued", "preprocessing", "embedding", "projecting",
                    "training", "done", "failed"]
    pct: int
    message: str
    run_id: str | None = None
    error: str | None = None


class JobSubmitOut(BaseModel):
    """POST /runs"""
    job_id: str


class ValidationOut(BaseModel):
    """POST /runs/validate. `errors` non-empty disables the Run button."""
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class EstimateOut(BaseModel):
    """POST /runs/estimate — the cost strip above the Run button."""
    n_qubits: int
    hilbert_dim: int
    sims: int
    mem_mb: float
    est_seconds: float


class PredictionOut(BaseModel):
    """POST /predict/{run_id}/{model}"""
    label: str
    confidences: dict[str, float]
    latency_ms: float
    bloch_angles: list[BlochAngleOut] | None = None


# ===========================================================================
# Responses — catalogs and previews (no dataclass mirror)
# ===========================================================================

class OpCatalogOut(BaseModel):
    op: str
    label: str
    order: int
    params_schema: dict[str, dict[str, Any]] = Field(default_factory=dict)
    locked: bool = False
    description: str = ""


class PresetOut(BaseModel):
    name: str
    label: str
    ops: list[OpConfigOut]
    description: str = ""


class BackboneOut(BaseModel):
    name: str
    label: str
    dim: int
    input_size: int


class BlochAngleOut(BaseModel):
    theta: float
    phi: float


class CircuitOpOut(BaseModel):
    name: str
    wires: list[int]


class EncodingOut(BaseModel):
    name: str
    label: str
    max_features: int
    qubit_formula: str
    description: str = ""
    # Present only when GET /encodings is called with ?n_features=N. The op
    # list is the REAL decomposed circuit, so the frontend can draw any
    # encoding without knowing its name — including ones added in Phase 2.
    qubits: int | None = None
    depth: int | None = None
    two_qubit_gates: int | None = None
    ops: list[CircuitOpOut] | None = None
    ops_truncated: bool | None = None
    # One feature swept across its scaled range — the encoding's geometry,
    # shown before any patient exists.
    bloch_demo: list[BlochAngleOut] | None = None


class ModelCatalogOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    name: str
    kind: Literal["classical", "quantum"]
    label: str
    param_schema: dict[str, dict[str, Any]] = Field(default_factory=dict)


class StageOut(BaseModel):
    op: str
    image_b64: str


class PreviewOut(BaseModel):
    """POST /preprocess/preview — the strip under the op list."""
    original_b64: str
    stages: list[StageOut]


class LeaderboardBestOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model: str
    label: str
    accuracy: float
    f1_macro: float
    run_id: str
    encoding: str
    n_features: int


class LeaderboardRowOut(BaseModel):
    """GET /leaderboard — best classical vs best quantum per dataset, across
    every run of that dataset."""
    dataset_id: str
    dataset_name: str
    n_runs: int
    best_classical: LeaderboardBestOut | None = None
    best_quantum: LeaderboardBestOut | None = None
    delta: float | None = None
    last_run_at: str


class ErrorOut(BaseModel):
    """Every 4xx/5xx body. main.py must never leak a traceback."""
    detail: str


# ===========================================================================
# Smoke test:  python -m backend.schemas
# ===========================================================================

if __name__ == "__main__":
    from dataclasses import fields as dc_fields

    from backend.core.contracts import (
        DatasetMeta, Estimate, JobState, Metrics, ModelResult, OpConfig,
        PerClassMetrics, Prediction, RunConfig, RunRecord, Telemetry,
        ValidationResult,
    )

    # --- THE DRIFT GUARD ---------------------------------------------------
    # Every mirrored model must match its dataclass field-for-field.
    PAIRS: list[tuple[type, type, set[str]]] = [
        (DatasetOut, DatasetMeta, {"preview_b64"}),     # response-only extra
        (OpConfigOut, OpConfig, set()),
        (OpConfigIn, OpConfig, set()),
        (RunConfigOut, RunConfig, set()),
        (RunConfigIn, RunConfig, set()),
        (MetricsOut, Metrics, set()),
        (PerClassOut, PerClassMetrics, set()),
        (TelemetryOut, Telemetry, set()),
        (ModelResultOut, ModelResult, set()),
        (RunOut, RunRecord, set()),
        (JobOut, JobState, set()),
        (EstimateOut, Estimate, set()),
        (PredictionOut, Prediction, set()),
        (ValidationOut, ValidationResult, set()),
    ]
    for model, dc, extra in PAIRS:
        want = {f.name for f in dc_fields(dc)}
        have = set(model.model_fields) - extra
        assert have == want, (
            f"{model.__name__} has drifted from {dc.__name__}:\n"
            f"  only in schema:    {sorted(have - want)}\n"
            f"  only in dataclass: {sorted(want - have)}"
        )

    # --- RunConfig survives the round trip through HTTP ---------------------
    cfg = RunConfig(
        dataset_id="d_a1b2", models=["svc", "qsvc"],
        ops=[OpConfig("crop_border", {"ratio": 0.03}), OpConfig("binarize")],
        preset_name="ecg_clean_scan", n_features=8, encoding="angle_y",
        model_params={"svc": {"C": 10}},
    )
    back = RunConfigIn(**cfg.to_dict()).to_contract()
    assert back.to_dict() == cfg.to_dict()
    assert back.ops_hash() == cfg.ops_hash(), "cache key must survive HTTP"

    # --- responses validate against real dataclass output ------------------
    counts = {"HB": 233, "MI": 239, "Normal": 284, "PMI": 172}
    ds = DatasetMeta("d_a1b2", "ECG", "image_folder", sorted(counts), counts,
                     928, 4)
    out = DatasetOut(**ds.to_dict(), preview_b64=["data:image/png;base64,x"])
    assert out.n_classes == 4

    tel = Telemetry(0.31, 0.01, "pennylane.default.qubit", n_params=None,
                    n_qubits=8, encoding="angle_y", circuit_depth=12,
                    two_qubit_gates=16, state_memory_mb=0.002)
    res = ModelResult("qsvc", "quantum", "Quantum Kernel SVC",
                      Metrics(.94, .94, .94, .94), [[1, 0], [0, 1]], tel,
                      {"HB": PerClassMetrics(.9, .9, .9, 47)}, "r__qsvc.pkl")
    run = RunRecord("r_8f2a", "d_a1b2", "ECG", sorted(counts), cfg,
                    [res, ModelResult.failed("vqc", "quantum", "VQC", "boom")])

    ro = RunOut(**run.to_dict())
    assert ro.results[0].telemetry.n_qubits == 8
    assert ro.results[1].telemetry.n_qubits is None     # classical-shaped nulls
    assert ro.results[1].error == "boom"
    assert set(ro.results[0].telemetry.model_dump()) == \
           set(ro.results[1].telemetry.model_dump()), "telemetry keys must match"

    RunSummaryOut(**run.summary())
    JobOut(**JobState("training", 74, "Training qsvc (2/3)").to_dict())
    EstimateOut(**Estimate(8, 256, 742, 1.5, 12.0).to_dict())
    PredictionOut(**Prediction("MI", {"MI": 0.91, "HB": 0.09}, 42.0).to_dict())
    ValidationOut(**ValidationResult(["bad"], ["meh"]).to_dict())

    # --- catalogs match what the registries actually emit -------------------
    from backend.core.preprocess import ops_catalog, presets_catalog
    for entry in ops_catalog():
        OpCatalogOut(**entry)
    for entry in presets_catalog():
        PresetOut(**entry)

    # --- requests reject junk ----------------------------------------------
    import pydantic
    for bad, why in [
        ({"dataset_id": "d", "n_features": "eight"}, "non-numeric features"),
        ({"dataset_id": "d", "test_size": 1.5}, "test_size out of range"),
        ({"dataset_id": "d", "nfeatures": 8}, "typo'd field name"),
        ({}, "missing dataset_id"),
    ]:
        try:
            RunConfigIn(**bad)
            raise AssertionError(f"expected rejection: {why}")
        except pydantic.ValidationError:
            pass

    # loose on purpose — validate.py owns this message, not a 422
    assert RunConfigIn(dataset_id="d", n_features=999).n_features == 999

    print(f"schemas OK — {len(PAIRS)} models match their dataclasses, "
          "RunConfig round-trips, cache key survives HTTP")
