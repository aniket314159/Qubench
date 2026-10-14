"""
core/contracts.py  (backend/core/contracts.py)
==============================================

The frozen contract layer. Written first, frozen Day 1 noon.

Rules this file enforces by construction:

  * No web stack. `import fastapi` must never appear at or below core/.
    backend/schemas.py mirrors these shapes in Pydantic; it imports FROM here.
  * No ML framework. No torch, no sklearn, no pennylane. Only numpy, for typing.
  * `n_classes` is a constructor argument on every model. Never hardcoded, and
    BaseModel.fit() actively rejects labels outside [0, n_classes).
  * Telemetry has ONE key set for classical and quantum. Quantum fields are
    nullable. One frontend table renders both kinds.
  * The embed cache key lives here (RunConfig.ops_hash / embed_key) so W3 and
    W4 cannot compute it differently.

THE INVARIANT
-------------
    ingest -> preprocess -> embed -> project -> encode -> train -> evaluate

Everything upstream of the model layer collapses to exactly:

    X : float32[n, N]     y : int[n]     class_names : list[str]

Models never see images. Models never see file paths. Models never see a
DatasetMeta.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    # aliases
    "FeatureMatrix", "LabelVector", "Kind", "JobStatus", "ProgressFn",
    # constants
    "STAGE_PCT", "TRAINING_SPAN",
    # data
    "DatasetMeta", "OpConfig", "RunConfig", "Metrics", "PerClassMetrics",
    "Telemetry", "ModelResult", "RunRecord", "CheckpointBundle",
    # transport
    "ValidationResult", "Estimate", "JobState", "Prediction",
    # registry specs
    "AdapterSpec", "OpSpec", "PresetSpec", "BackboneSpec", "EncodingSpec",
    # model layer
    "BaseModel", "ModelFactory",
    # helpers
    "utc_now_iso", "new_id",
]


# ---------------------------------------------------------------------------
# Type aliases and shared constants
# ---------------------------------------------------------------------------

FeatureMatrix = NDArray[np.float32]   # shape (n, N)
LabelVector = NDArray[np.int64]       # shape (n,), values in [0, n_classes)

Kind = Literal["classical", "quantum"]

JobStatus = Literal[
    "queued", "preprocessing", "embedding", "projecting",
    "training", "done", "failed",
]

# benchmark.py (W4) calls this; jobs.py (W3) supplies it. Crosses the boundary,
# so the signature is pinned here: (pct, message) -> None
ProgressFn = Callable[[int, str], None]

# Percentage anchors per stage. Shared so the bar never jumps backwards.
# TOTAL over JobStatus on purpose: jobs.py may do STAGE_PCT[status] without
# a guard. "failed" reads 100 because the job is over; the frontend colours
# off `status`, not `pct`, so a failed bar renders red and full.
STAGE_PCT: dict[str, int] = {
    "queued": 0,
    "preprocessing": 5,
    "embedding": 20,
    "projecting": 55,
    "training": 60,
    "done": 100,
    "failed": 100,
}

# benchmark.py divides this span across the selected models, so progress during
# training is computed the same way no matter who calls it:
#     lo, hi = TRAINING_SPAN
#     pct = lo + int((hi - lo) * (i / len(cfg.models)))
TRAINING_SPAN: tuple[int, int] = (60, 95)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    """`new_id("d") -> "d_3f9a2c"`. Short, readable, collision-safe enough."""
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


def _known(cls: type, d: dict[str, Any]) -> dict[str, Any]:
    """Drop unknown keys so old JSON on disk never crashes a newer dataclass."""
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in allowed}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class DatasetMeta:
    """One uploaded dataset. Element of storage/datasets.json.

    Images live at storage/datasets/{dataset_id}/{class_name}/*.png
    Preview thumbs at storage/datasets/{dataset_id}/_preview/*.png
    `preview_b64` is a RESPONSE-only field on schemas.py, never persisted here.
    """
    dataset_id: str
    name: str
    adapter: str                       # seam 1 key, e.g. "image_folder"
    class_names: list[str]             # sorted; index == integer label
    counts: dict[str, int]             # class_name -> n images kept
    n_samples: int
    n_classes: int
    modality: str = "ecg"
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if self.n_classes != len(self.class_names):
            raise ValueError(
                f"n_classes={self.n_classes} but {len(self.class_names)} class_names"
            )
        missing = set(self.class_names) - set(self.counts)
        if missing:
            raise ValueError(f"counts is missing classes: {sorted(missing)}")
        total = sum(self.counts.values())
        if total != self.n_samples:
            # Catches the classic ingest bug: counts recorded before the
            # stratified subsample, n_samples recorded after.
            raise ValueError(
                f"counts sum to {total} but n_samples={self.n_samples}"
            )

    def label_of(self, class_name: str) -> int:
        """The integer label for a class. The only sanctioned mapping."""
        return self.class_names.index(class_name)

    def is_binary(self) -> bool:
        return self.n_classes == 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetMeta":
        return cls(**_known(cls, d))


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

@dataclass
class OpConfig:
    """One preprocessing op the user switched on, plus its tuned params.

    There is no `order` field here on purpose. Order belongs to the op itself
    (OpSpec.order, seam 2), not to the user's config. Users toggle and tune,
    never reorder.
    """
    op: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OpConfig":
        return cls(op=d["op"], params=dict(d.get("params") or {}))


@dataclass
class RunConfig:
    """Exactly what POST /runs accepts. The single input to run_benchmark()."""
    dataset_id: str
    models: list[str]                                  # keys into MODEL_REGISTRY
    ops: list[OpConfig] = field(default_factory=list)
    preset_name: str | None = None                     # provenance only
    backbone: str = "resnet18"
    n_features: int = 8
    encoding: str = "angle_y"
    test_size: float = 0.2
    seed: int = 42
    model_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    # -- cache key: defined once, here, for both W3 and W4 -------------------
    def ops_hash(self) -> str:
        """Order-insensitive. apply_ops() sorts by OpSpec.order anyway, so two
        configs that differ only in the order the user clicked the checkboxes
        MUST hit the same cache entry."""
        payload = json.dumps(
            sorted((o.to_dict() for o in self.ops), key=lambda o: o["op"]),
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:8]

    def embed_key(self) -> str:
        """Filename under storage/artifacts/. Changing n_features or encoding
        HITS the cache; changing ops recomputes. That boundary is deliberate."""
        return f"{self.dataset_id}__{self.backbone}__{self.ops_hash()}.npy"

    def params_for(self, model_name: str) -> dict[str, Any]:
        return dict(self.model_params.get(model_name, {}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunConfig":
        d = _known(cls, dict(d))
        d["ops"] = [OpConfig.from_dict(o) for o in (d.get("ops") or [])]
        return cls(**d)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    accuracy: float
    f1_macro: float
    precision: float            # macro
    recall: float               # macro

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Metrics":
        return cls(**_known(cls, d))

    @classmethod
    def zero(cls) -> "Metrics":
        return cls(0.0, 0.0, 0.0, 0.0)


@dataclass
class PerClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PerClassMetrics":
        return cls(**_known(cls, d))


@dataclass
class Telemetry:
    """ONE key set for both kinds. Quantum fields are None for classical models.
    This is what lets a single frontend table render every row. Adding a field
    here means adding it for BOTH kinds, or the table breaks."""
    fit_seconds: float
    predict_seconds: float
    backend: str                        # "sklearn" | "pennylane.default.qubit" | ...
    n_params: int | None = None         # trainable params, or support vectors
    n_qubits: int | None = None
    encoding: str | None = None
    circuit_depth: int | None = None
    two_qubit_gates: int | None = None
    state_memory_mb: float | None = None
    # The kernel bandwidth a quantum model actually used. Nullable like the
    # rest: classical models have no such thing. Worth a column of its own
    # because it is the single largest effect we have measured — the same zz
    # kernel scores 0.774 at bandwidth 1.0 and 0.903 cross-validated, and a
    # results table that hides it invites "did you just get lucky?".
    bandwidth: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Telemetry":
        return cls(**_known(cls, d))


@dataclass
class ModelResult:
    """One trained model's row in the benchmark table."""
    model: str
    kind: Kind
    label: str
    metrics: Metrics
    confusion_matrix: list[list[int]]                    # [true][pred]
    telemetry: Telemetry
    per_class: dict[str, PerClassMetrics] = field(default_factory=dict)
    checkpoint: str | None = None                        # filename, not a path
    error: str | None = None                             # set if this model failed

    @property
    def ok(self) -> bool:
        return self.error is None

    @classmethod
    def failed(cls, model: str, kind: Kind, label: str, error: str) -> "ModelResult":
        """One model blowing up degrades its row, it does not kill the run.
        The frontend still gets a complete, renderable object."""
        return cls(
            model=model, kind=kind, label=label,
            metrics=Metrics.zero(), confusion_matrix=[],
            telemetry=Telemetry(0.0, 0.0, "n/a"),
            error=error[:500],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelResult":
        d = _known(cls, dict(d))
        d["metrics"] = Metrics.from_dict(d["metrics"])
        d["telemetry"] = Telemetry.from_dict(d["telemetry"])
        d["per_class"] = {
            k: PerClassMetrics.from_dict(v)
            for k, v in (d.get("per_class") or {}).items()
        }
        return cls(**d)


@dataclass
class RunRecord:
    """One completed benchmark. Element of storage/runs.json."""
    run_id: str
    dataset_id: str
    dataset_name: str
    class_names: list[str]
    config: RunConfig
    results: list[ModelResult] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)

    def best(self, kind: Kind) -> ModelResult | None:
        rows = [r for r in self.results if r.kind == kind and r.ok]
        return max(rows, key=lambda r: r.metrics.accuracy) if rows else None

    def result_for(self, model: str) -> ModelResult | None:
        """POST /predict/{run_id}/{model} needs the checkpoint filename."""
        return next((r for r in self.results if r.model == model), None)

    def summary(self) -> dict[str, Any]:
        """Lightweight payload for GET /runs and GET /leaderboard."""
        c, q = self.best("classical"), self.best("quantum")
        return {
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "created_at": self.created_at,
            "n_features": self.config.n_features,
            "encoding": self.config.encoding,
            "models": [r.model for r in self.results],
            "best_classical": (
                {"model": c.model, "label": c.label, "accuracy": c.metrics.accuracy}
                if c else None
            ),
            "best_quantum": (
                {"model": q.model, "label": q.label, "accuracy": q.metrics.accuracy}
                if q else None
            ),
            "delta": (
                round(q.metrics.accuracy - c.metrics.accuracy, 4) if (c and q) else None
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunRecord":
        d = _known(cls, dict(d))
        d["config"] = RunConfig.from_dict(d["config"])
        d["results"] = [ModelResult.from_dict(r) for r in (d.get("results") or [])]
        return cls(**d)


# ---------------------------------------------------------------------------
# Transport-only shapes (no persistence)
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Estimate:
    """POST /runs/estimate. Rendered as the cost strip above the Run button."""
    n_qubits: int
    hilbert_dim: int
    sims: int
    mem_mb: float
    est_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JobState:
    """Seam 5's value type. GET /jobs/{id} returns exactly this."""
    status: JobStatus = "queued"
    pct: int = 0
    message: str = "Queued"
    run_id: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Prediction:
    """POST /predict/{run_id}/{model}."""
    label: str
    confidences: dict[str, float]        # class_name -> probability, sums to 1
    latency_ms: float
    # Where THIS patient's features land on each qubit, after the encoding's
    # first data layer and the model's tuned bandwidth. None for classical
    # models and for encodings with no per-qubit reading.
    bloch_angles: list[dict[str, float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Checkpoint bundle
# ---------------------------------------------------------------------------

@dataclass
class CheckpointBundle:
    """What checkpoint.save() pickles and checkpoint.load() returns.

    Not JSON. Pickled to storage/checkpoints/{run_id}__{model}.pkl.

    It carries `ops`, `backbone` and `n_features` because the Diagnose screen
    must replay the EXACT training pipeline on the uploaded image:

        ops -> embed(backbone) -> pca -> scaler -> estimator.predict_proba

    If Diagnose applies different preprocessing than training did, predictions
    are quietly wrong and nothing raises. Binding the recipe to the weights is
    what prevents that. Never reconstruct these from the live RunConfig; the
    user may have changed the UI since the run.
    """
    run_id: str
    model_name: str
    class_names: list[str]
    ops: list[OpConfig]
    backbone: str
    encoding: str
    n_features: int
    estimator: "BaseModel"
    pca: Any                             # fitted sklearn PCA (train only)
    scaler: Any                          # fitted encoding scaler (train only)
    created_at: str = field(default_factory=utc_now_iso)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


# ---------------------------------------------------------------------------
# Registry specs -- the value type of each seam
# ---------------------------------------------------------------------------

@dataclass
class AdapterSpec:
    """Seam 1. ADAPTERS[name] -> AdapterSpec."""
    name: str
    label: str
    load: Callable[..., Any]             # (root: Path) -> (paths, label_strings)


@dataclass
class OpSpec:
    """Seam 2. OP_REGISTRY[op] -> OpSpec.

    `order` is fixed by the op, not the user. apply_ops() sorts the enabled ops
    by this field. to_rgb (900) and resize (1000) are always last and are not
    user-toggleable.

    params_schema drives the frontend ParamControl generically:
        {"thr": {"type": "int", "min": 0, "max": 255, "default": 200,
                 "label": "Threshold"}}
    """
    op: str
    label: str
    order: int
    fn: Callable[..., np.ndarray]        # (img: uint8[H,W,C], **params) -> uint8
    params_schema: dict[str, dict[str, Any]] = field(default_factory=dict)
    locked: bool = False                 # True -> frontend shows a padlock
    description: str = ""

    def defaults(self) -> dict[str, Any]:
        return {k: v.get("default") for k, v in self.params_schema.items()}

    def catalog_entry(self) -> dict[str, Any]:
        """GET /ops/catalog"""
        return {
            "op": self.op,
            "label": self.label,
            "order": self.order,
            "params_schema": self.params_schema,
            "locked": self.locked,
            "description": self.description,
        }


@dataclass
class PresetSpec:
    """Seam 2's companion. PRESETS[name] -> PresetSpec."""
    name: str
    label: str
    ops: list[OpConfig]
    description: str = ""

    def catalog_entry(self) -> dict[str, Any]:
        """GET /presets"""
        return {
            "name": self.name,
            "label": self.label,
            "ops": [o.to_dict() for o in self.ops],
            "description": self.description,
        }


@dataclass
class BackboneSpec:
    """Seam 3. BACKBONES[name] -> BackboneSpec."""
    name: str
    label: str
    dim: int                             # embedding width, e.g. 512
    input_size: int                      # e.g. 224
    build: Callable[[], Any]             # () -> callable batch -> float32[n, dim]

    def catalog_entry(self) -> dict[str, Any]:
        """GET /backbones"""
        return {
            "name": self.name, "label": self.label,
            "dim": self.dim, "input_size": self.input_size,
        }


@dataclass
class EncodingSpec:
    """Seam 3.5. ENCODINGS[name] -> EncodingSpec.

    Binds qubit count, scaler and circuit together so they cannot be mismatched.
    PennyLane only. No Qiskit anywhere below this line.
    """
    name: str
    label: str
    max_features: int
    qubit_formula: str                   # display string, e.g. "N", "ceil(log2 N)"
    qubits_for: Callable[[int], int]     # n_features -> n_qubits
    make_scaler: Callable[[], Any]       # () -> fresh sklearn transformer
    template: Callable[..., None]        # (x, wires) -> applies PennyLane ops
    two_qubit_gates_for: Callable[[int], int] | None = None
    # (x, bandwidth) -> [{"theta","phi"}] for one patient, or None when the
    # encoding has no per-qubit rotation reading. Amplitude returns None: its
    # qubits hold a joint superposition, so drawing eight independent arrows
    # would be a picture of something that is not happening.
    bloch_for: Callable[..., list[dict] | None] | None = None
    description: str = ""

    def hilbert_dim(self, n_features: int) -> int:
        return 2 ** self.qubits_for(n_features)

    def state_memory_mb(self, n_features: int) -> float:
        """One complex64 statevector."""
        return self.hilbert_dim(n_features) * 8 / (1024 ** 2)

    def readout(self, n_features: int) -> str:
        """The live string under the feature slider on the Configure screen.
        Generated server-side so W2 never hardcodes a qubit formula."""
        q = self.qubits_for(n_features)
        parts = [f"{n_features} features", f"{q} qubits",
                 f"{self.hilbert_dim(n_features)}-dim Hilbert space"]
        if self.two_qubit_gates_for is not None:
            parts.append(f"~{self.two_qubit_gates_for(n_features)} two-qubit gates")
        return " \u2192 ".join(parts)

    def catalog_entry(self) -> dict[str, Any]:
        """GET /encodings"""
        return {
            "name": self.name,
            "label": self.label,
            "max_features": self.max_features,
            "qubit_formula": self.qubit_formula,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Seam 4 -- the model ABC
# ---------------------------------------------------------------------------

class BaseModel(ABC):
    """Every entry in MODEL_REGISTRY is a subclass of this.

    Uniform constructor, so benchmark.py builds any model without branching:

        MODEL_REGISTRY[name](
            n_classes=ds.n_classes,
            n_features=cfg.n_features,
            encoding=ENCODINGS[cfg.encoding],
            seed=cfg.seed,
            **cfg.params_for(name),
        )

    Classical models accept `encoding` and ignore it. That is the price of one
    call site and it is worth paying.

    Subclasses implement `_fit` and `_predict_proba` only. The public `fit`,
    `predict` and `predict_proba` are concrete here so timing is measured
    identically for every model and nobody can forget to record it.
    """

    # -- class-level identity; feeds GET /models/catalog --------------------
    name: ClassVar[str] = ""
    kind: ClassVar[Kind] = "classical"
    label: ClassVar[str] = ""
    backend_name: ClassVar[str] = "sklearn"
    param_schema: ClassVar[dict[str, dict[str, Any]]] = {}

    def __init__(
        self,
        n_classes: int,
        n_features: int,
        encoding: EncodingSpec | None = None,
        seed: int = 42,
        **params: Any,
    ) -> None:
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        if (self.kind == "quantum" and encoding is not None
                and n_features > encoding.max_features):
            # The encoding's own declared ceiling, not a policy rule.
            # validate.py catches this in the UI; this catches pipeline.py.
            raise ValueError(
                f"{self.name}: encoding '{encoding.name}' supports at most "
                f"{encoding.max_features} features, got {n_features}"
            )
        self.n_classes = int(n_classes)
        self.n_features = int(n_features)
        self.encoding = encoding
        self.seed = int(seed)
        self.params: dict[str, Any] = dict(params)
        self._fit_seconds: float = 0.0
        self._predict_seconds: float = 0.0
        self._fitted: bool = False

    # -- subclasses implement these two ------------------------------------
    @abstractmethod
    def _fit(self, X: FeatureMatrix, y: LabelVector) -> None: ...

    @abstractmethod
    def _predict_proba(self, X: FeatureMatrix) -> NDArray[np.float64]:
        """Return float64[n, n_classes], rows summing to 1."""

    # -- concrete public surface -------------------------------------------
    def fit(self, X: FeatureMatrix, y: LabelVector) -> "BaseModel":
        self._check_X(X)
        self._check_y(y, len(X))
        t0 = perf_counter()
        self._fit(X, y)
        self._fit_seconds = perf_counter() - t0
        self._fitted = True
        return self

    def predict_proba(self, X: FeatureMatrix) -> NDArray[np.float64]:
        self._require_fitted()
        self._check_X(X)
        t0 = perf_counter()
        p = np.asarray(self._predict_proba(X), dtype=np.float64)
        self._predict_seconds = perf_counter() - t0

        if p.ndim != 2 or p.shape != (len(X), self.n_classes):
            raise ValueError(
                f"{self.name}: predict_proba returned {p.shape}, "
                f"expected ({len(X)}, {self.n_classes})"
            )
        if not np.isfinite(p).all() or (p < 0).any():
            raise ValueError(f"{self.name}: predict_proba returned NaN/inf/negative")
        rows = p.sum(axis=1, keepdims=True)
        if (rows == 0).any():
            raise ValueError(f"{self.name}: predict_proba returned an all-zero row")
        return p / rows          # tolerate float drift; argmax is unaffected

    def predict(self, X: FeatureMatrix) -> LabelVector:
        return self.predict_proba(X).argmax(axis=1).astype(np.int64)

    # -- telemetry ---------------------------------------------------------
    def n_trainable_params(self) -> int | None:
        """Override where meaningful. None is a legitimate answer."""
        return None

    def quantum_telemetry(self) -> dict[str, Any]:
        """Quantum subclasses override and return real values. The key set is
        fixed here so classical and quantum rows always agree."""
        return {
            "n_qubits": None,
            "encoding": None,
            "circuit_depth": None,
            "two_qubit_gates": None,
            "state_memory_mb": None,
            "bandwidth": None,
        }

    def telemetry(self) -> Telemetry:
        return Telemetry(
            fit_seconds=round(self._fit_seconds, 4),
            predict_seconds=round(self._predict_seconds, 4),
            backend=self.backend_name,
            n_params=self.n_trainable_params(),
            **self.quantum_telemetry(),
        )

    # -- catalog -----------------------------------------------------------
    @classmethod
    def catalog_entry(cls) -> dict[str, Any]:
        """GET /models/catalog"""
        return {
            "name": cls.name,
            "kind": cls.kind,
            "label": cls.label,
            "param_schema": cls.param_schema,
        }

    # -- internals ---------------------------------------------------------
    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: predict called before fit")

    def _check_X(self, X: FeatureMatrix) -> None:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"{self.name}: X must be 2-D, got shape {X.shape}")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"{self.name}: X has {X.shape[1]} features, "
                f"model was built for {self.n_features}"
            )
        if not np.issubdtype(X.dtype, np.floating):
            raise ValueError(f"{self.name}: X must be floating, got {X.dtype}")

    def _check_y(self, y: LabelVector, n: int) -> None:
        """The n_classes hardcode-hunt, enforced at runtime. A model built for
        4 classes and handed a 2-class dataset dies here, loudly, not silently
        at metric time."""
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != n:
            raise ValueError(f"{self.name}: y must be 1-D of length {n}, got {y.shape}")
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError(f"{self.name}: y must be integer labels, got {y.dtype}")
        if len(y) and (y.min() < 0 or y.max() >= self.n_classes):
            raise ValueError(
                f"{self.name}: y has labels in [{y.min()}, {y.max()}] but model "
                f"was built for n_classes={self.n_classes}"
            )

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(n_classes={self.n_classes}, "
                f"n_features={self.n_features}, fitted={self._fitted})")


ModelFactory = Callable[..., BaseModel]


# ---------------------------------------------------------------------------
# Smoke test:  python -m backend.core.contracts
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    class _Dummy(BaseModel):
        name, kind, label = "dummy", "classical", "Dummy"

        def _fit(self, X, y):
            self._prior = np.bincount(y, minlength=self.n_classes) / len(y)

        def _predict_proba(self, X):
            return np.tile(self._prior, (len(X), 1))

    rng = np.random.default_rng(0)
    X = rng.random((40, 8)).astype(np.float32)
    y = rng.integers(0, 3, 40).astype(np.int64)

    m = _Dummy(n_classes=3, n_features=8).fit(X, y)
    assert m.predict(X).shape == (40,)
    assert m.telemetry().n_qubits is None
    assert set(m.telemetry().to_dict()) == {
        "fit_seconds", "predict_seconds", "backend", "n_params", "n_qubits",
        "encoding", "circuit_depth", "two_qubit_gates", "state_memory_mb",
        "bandwidth",
    }

    # n_classes mismatch must fail loudly
    try:
        _Dummy(n_classes=2, n_features=8).fit(X, y)
        raise AssertionError("expected a label-range failure")
    except ValueError as e:
        assert "n_classes=2" in str(e)

    # ops_hash is order-insensitive
    a = RunConfig("d_a1b2", ["svc"], [OpConfig("crop_border", {"ratio": .03}),
                                      OpConfig("grayscale")])
    b = RunConfig("d_a1b2", ["svc"], [OpConfig("grayscale"),
                                      OpConfig("crop_border", {"ratio": .03})])
    assert a.ops_hash() == b.ops_hash()
    assert RunConfig.from_dict(a.to_dict()).embed_key() == a.embed_key()

    run = RunRecord(
        run_id="r_8f2a", dataset_id="d_a1b2", dataset_name="ECG Cardiac (Multan)",
        class_names=["HB", "MI", "Normal", "PMI"], config=a,
        results=[
            ModelResult(
                model="svc", kind="classical", label="SVC (RBF, tuned)",
                metrics=Metrics(0.912, 0.908, 0.914, 0.912),
                confusion_matrix=[[1]], telemetry=m.telemetry(),
                per_class={"HB": PerClassMetrics(.93, .96, .94, 47)},
            ),
            ModelResult.failed("vqc", "quantum", "VQC", "CUDA out of memory"),
        ],
    )
    rt = RunRecord.from_dict(json.loads(json.dumps(run.to_dict())))
    assert rt.config.n_features == 8
    assert rt.best("quantum") is None          # failed rows never win
    assert rt.summary()["delta"] is None

    assert rt.result_for("vqc").error.startswith("CUDA")
    assert rt.result_for("nope") is None

    counts = {"HB": 233, "MI": 239, "Normal": 284, "PMI": 172}
    ds = DatasetMeta("d_a1b2", "ECG", "image_folder",
                     list(counts), counts, sum(counts.values()), 4)
    assert ds.label_of("Normal") == 2

    try:
        DatasetMeta("d_x", "ECG", "image_folder", list(counts), counts, 999, 4)
        raise AssertionError("expected an n_samples mismatch")
    except ValueError as e:
        assert "n_samples" in str(e)

    assert set(STAGE_PCT) >= {"queued", "preprocessing", "embedding",
                              "projecting", "training", "done", "failed"}

    print("contracts OK \u2014 round-trip, telemetry keys, ops_hash, label guard")
