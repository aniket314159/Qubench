"""
backend/validate.py
===================

Config rules. `validate(cfg, ds) -> ValidationResult(errors, warnings)`.

    errors    disable the Run button. The config cannot produce a valid run.
    warnings  render in amber. The run will work but you may not like it.

Called twice, on purpose:

    POST /runs/validate   on every change to the Configure screen
    POST /runs            again, server-side, before submitting the job

The second call is not redundant. Never trust the client, and `curl` is a
client. It is also the only defence for `scripts/pipeline.py` users, who never
touch the UI at all.

DEPENDENCY NOTE
---------------
Several rules need W4's registries (which encodings exist, what each one's
`max_features` is, which models exist). Those files may not be written yet, so
`collect_facts()` imports them defensively: rules that need a missing registry
are skipped rather than crashing or producing false errors. When W4 lands, the
rules light up with no change to this file. `facts.complete` reports whether
the full rule set is active — useful in a test, not shown to users.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from backend.core.contracts import DatasetMeta, RunConfig, ValidationResult
from backend.core.preprocess import OP_REGISTRY

log = logging.getLogger(__name__)

# --- thresholds ------------------------------------------------------------
MIN_FEATURES = 2
MIN_QUBITS = 2                  # a 1-qubit circuit has no entangling ansatz
MIN_TEST_SIZE = 0.05
MAX_TEST_SIZE = 0.50
MIN_PER_CLASS_SPLIT = 2         # sklearn's stratify floor
MIN_PER_CLASS_COMFORT = 10      # below this, per-class metrics are noise
MIN_TEST_PER_CLASS = 3
QUBIT_WARN = 10                 # simulator cost climbs steeply past here
IMBALANCE_WARN = 3.0            # majority:minority ratio
SMALL_DATASET_WARN = 60
DEFAULT_EMBED_DIM = 512         # resnet18, if BACKBONES is not importable yet


# ---------------------------------------------------------------------------
# Facts gathered from W4's registries (or not, if they do not exist yet)
# ---------------------------------------------------------------------------

@dataclass
class Facts:
    """Only what the rules need, extracted from the registries. Keeping this
    a plain-data snapshot means validate.py does not depend on W4's types and
    the tests can construct one by hand."""
    encodings: dict[str, dict[str, Any]] | None = None   # name -> {max_features, qubits_for}
    models: dict[str, str] | None = None                 # name -> kind
    backbones: dict[str, int] | None = None              # name -> embedding dim
    missing: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing


def collect_facts() -> Facts:
    f = Facts()

    try:
        from backend.core.encodings import ENCODINGS
        f.encodings = {
            name: {"max_features": spec.max_features,
                   "qubits_for": spec.qubits_for,
                   "label": spec.label}
            for name, spec in ENCODINGS.items()
        }
    except Exception:
        f.missing.append("encodings")

    try:
        from backend.core.registry import MODEL_REGISTRY
        f.models = {name: getattr(cls, "kind", "classical")
                    for name, cls in MODEL_REGISTRY.items()}
    except Exception:
        f.missing.append("models")

    try:
        from backend.core.embed import BACKBONES
        f.backbones = {name: spec.dim for name, spec in BACKBONES.items()}
    except Exception:
        f.missing.append("backbones")

    if f.missing:
        log.debug("validate: registries not available yet: %s", f.missing)
    return f


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------

def _check_dataset(ds: DatasetMeta | None, cfg: RunConfig,
                   err: Callable[[str], None], warn: Callable[[str], None]) -> None:
    if ds is None:
        err(f"Dataset {cfg.dataset_id!r} was not found. Upload it again.")
        return

    if ds.n_classes < 2:
        err(f"This dataset has {ds.n_classes} class. Classification needs at "
            "least two.")

    if ds.counts:
        smallest = min(ds.counts, key=lambda c: ds.counts[c])
        fewest = ds.counts[smallest]
        if fewest < MIN_PER_CLASS_SPLIT:
            err(f"Class {smallest!r} has {fewest} image(s). A stratified "
                f"train/test split needs at least {MIN_PER_CLASS_SPLIT}.")
        elif fewest < MIN_PER_CLASS_COMFORT:
            warn(f"Class {smallest!r} has only {fewest} images. Per-class "
                 "precision and recall for it will be very noisy.")

        largest = max(ds.counts.values())
        if fewest and largest / fewest >= IMBALANCE_WARN:
            warn(f"Classes are imbalanced ({largest}:{fewest}). Compare macro "
                 "F1 rather than accuracy — accuracy flatters the majority class.")

    if ds.n_samples < SMALL_DATASET_WARN:
        warn(f"Only {ds.n_samples} images. Differences of a few percent "
             "between models will not be meaningful.")


def _check_split(cfg: RunConfig, ds: DatasetMeta | None,
                 err: Callable[[str], None], warn: Callable[[str], None]) -> int:
    """Returns the training-set size, which the PCA rule needs."""
    if not (MIN_TEST_SIZE <= cfg.test_size <= MAX_TEST_SIZE):
        err(f"Test size must be between {MIN_TEST_SIZE:g} and "
            f"{MAX_TEST_SIZE:g}; got {cfg.test_size:g}.")
        return 0

    if ds is None:
        return 0

    n_train = int(math.floor(ds.n_samples * (1 - cfg.test_size)))
    for cls, n in (ds.counts or {}).items():
        n_test = int(math.floor(n * cfg.test_size))
        if n_test < 1:
            err(f"At a {cfg.test_size:.0%} test split, class {cls!r} ({n} "
                "images) gets no test samples at all.")
        elif n_test < MIN_TEST_PER_CLASS:
            warn(f"Class {cls!r} will have only {n_test} test image(s). Its "
                 "recall can only take a few discrete values.")
    return n_train


def _check_features(cfg: RunConfig, facts: Facts, n_train: int,
                    err: Callable[[str], None], warn: Callable[[str], None]) -> None:
    if cfg.n_features < MIN_FEATURES:
        err(f"Feature count must be at least {MIN_FEATURES}; got {cfg.n_features}.")
        return

    dim = DEFAULT_EMBED_DIM
    if facts.backbones is not None:
        if cfg.backbone not in facts.backbones:
            err(f"Unknown backbone {cfg.backbone!r}. Available: "
                f"{sorted(facts.backbones)}.")
        else:
            dim = facts.backbones[cfg.backbone]

    if cfg.n_features > dim:
        err(f"Cannot reduce to {cfg.n_features} features: {cfg.backbone} "
            f"produces only {dim} dimensions.")

    # PCA cannot produce more components than it has training rows.
    if n_train and cfg.n_features > n_train:
        err(f"PCA cannot produce {cfg.n_features} components from "
            f"{n_train} training images.")

    if facts.encodings is None:
        return

    if cfg.encoding not in facts.encodings:
        err(f"Unknown encoding {cfg.encoding!r}. Available: "
            f"{sorted(facts.encodings)}.")
        return

    spec = facts.encodings[cfg.encoding]
    if cfg.n_features > spec["max_features"]:
        err(f"{spec['label']} supports at most {spec['max_features']} "
            f"features; you selected {cfg.n_features}.")
        return

    try:
        qubits = int(spec["qubits_for"](cfg.n_features))
    except Exception:
        return

    wants_quantum = _has_quantum(cfg, facts)

    # A one-qubit circuit has no room for an entangling ansatz, and a
    # variational model raises at FIT time rather than here — which means the
    # Run button stays enabled and the job dies at 60%. Block it at the button.
    # Reachable via amplitude: ceil(log2 2) == 1.
    if wants_quantum and qubits < MIN_QUBITS:
        err(f"{spec['label']} maps {cfg.n_features} features onto {qubits} "
            f"qubit(s). Quantum models need at least {MIN_QUBITS}. Increase "
            "the feature count or pick another encoding.")
        return

    if wants_quantum and qubits > QUBIT_WARN:
        warn(f"{qubits} qubits means a {2**qubits:,}-dimensional state per "
             "image. Simulation time roughly doubles with every extra qubit — "
             "check the cost estimate before running.")


def _has_quantum(cfg: RunConfig, facts: Facts) -> bool:
    if facts.models is None:
        return True                       # assume yes; warnings stay useful
    return any(facts.models.get(m) == "quantum" for m in cfg.models)


def _check_models(cfg: RunConfig, facts: Facts,
                  err: Callable[[str], None], warn: Callable[[str], None]) -> None:
    if not cfg.models:
        err("Select at least one model to run.")
        return

    dupes = {m for m in cfg.models if cfg.models.count(m) > 1}
    if dupes:
        err(f"Model(s) selected more than once: {sorted(dupes)}.")

    if facts.models is None:
        return

    unknown = [m for m in cfg.models if m not in facts.models]
    if unknown:
        err(f"Unknown model(s): {sorted(unknown)}. Available: "
            f"{sorted(facts.models)}.")
        return

    kinds = {facts.models[m] for m in cfg.models}
    if "quantum" not in kinds:
        warn("No quantum model selected. The benchmark will have nothing to "
             "compare the classical baseline against.")
    if "classical" not in kinds:
        warn("No classical model selected. Without a baseline, a quantum "
             "accuracy number means nothing on its own.")


def _check_ops(cfg: RunConfig,
               err: Callable[[str], None], warn: Callable[[str], None]) -> None:
    seen: set[str] = set()
    for item in cfg.ops:
        if item.op not in OP_REGISTRY:
            err(f"Unknown preprocessing step {item.op!r}.")
            continue
        if item.op in seen:
            warn(f"Step {item.op!r} is listed twice; the last settings win.")
        seen.add(item.op)

        schema = OP_REGISTRY[item.op].params_schema
        for key, val in (item.params or {}).items():
            if key not in schema:
                warn(f"{OP_REGISTRY[item.op].label}: unknown setting {key!r} "
                     "will be ignored.")
                continue
            meta = schema[key]
            if meta["type"] == "enum":
                if val not in meta["options"]:
                    err(f"{OP_REGISTRY[item.op].label}: {key} must be one of "
                        f"{meta['options']}; got {val!r}.")
            elif meta["type"] in ("int", "float"):
                try:
                    num = float(val)
                except (TypeError, ValueError):
                    err(f"{OP_REGISTRY[item.op].label}: {key} must be a "
                        f"number; got {val!r}.")
                    continue
                if not (meta["min"] <= num <= meta["max"]):
                    warn(f"{OP_REGISTRY[item.op].label}: {key}={val} is "
                         f"outside {meta['min']}–{meta['max']} and will be "
                         "clamped.")

    if not cfg.ops:
        warn("No preprocessing selected. Images go to the backbone as-is, "
             "grid paper and all. Useful as a baseline, weak as a result.")

    if "denoise" in seen:
        params = next(o.params for o in cfg.ops if o.op == "denoise")
        if (params or {}).get("method") == "nlmeans":
            warn("Denoise is set to nlmeans (~300 ms per image). That adds "
                 "roughly five minutes to every run that is not cached.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def validate(cfg: RunConfig, ds: DatasetMeta | None,
             facts: Facts | None = None) -> ValidationResult:
    """The single source of truth for whether a config can run.

    `ds` is what `store.get_dataset(cfg.dataset_id)` returned — None is a
    legitimate input and produces an error, not an exception.
    """
    result = ValidationResult()
    err = result.errors.append
    warn = result.warnings.append
    facts = facts if facts is not None else collect_facts()

    _check_dataset(ds, cfg, err, warn)
    n_train = _check_split(cfg, ds, err, warn)
    _check_features(cfg, facts, n_train, err, warn)
    _check_models(cfg, facts, err, warn)
    _check_ops(cfg, err, warn)

    return result


# ---------------------------------------------------------------------------
# Smoke test:  python -m backend.validate
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from backend.core.contracts import OpConfig

    # A hand-built Facts standing in for W4's registries, so every rule is
    # exercised now rather than on the day they land.
    FACTS = Facts(
        encodings={
            "angle_y": {"max_features": 12, "qubits_for": lambda n: n,
                        "label": "Angle (RY)"},
            "zz": {"max_features": 10, "qubits_for": lambda n: n,
                   "label": "ZZ feature map"},
            "amplitude": {"max_features": 64,
                          "qubits_for": lambda n: max(1, math.ceil(math.log2(n))),
                          "label": "Amplitude"},
        },
        models={"svc": "classical", "rf": "classical",
                "qsvc": "quantum", "vqc": "quantum"},
        backbones={"resnet18": 512},
    )

    counts = {"HB": 233, "MI": 239, "Normal": 284, "PMI": 172}
    DS = DatasetMeta("d_a1b2", "ECG", "image_folder", sorted(counts), counts,
                     sum(counts.values()), 4)

    def cfg(**kw) -> RunConfig:
        base = dict(dataset_id="d_a1b2", models=["svc", "qsvc"],
                    ops=[OpConfig("grayscale"), OpConfig("binarize")],
                    n_features=8, encoding="angle_y", test_size=0.2)
        base.update(kw)
        return RunConfig(**base)

    def check(name, r, *, errors=0, must=None):
        assert len(r.errors) == errors, f"{name}: {r.errors}"
        if must:
            blob = " ".join(r.errors + r.warnings)
            assert must in blob, f"{name}: expected {must!r} in {blob!r}"

    # --- the good config is clean ------------------------------------------
    r = validate(cfg(), DS, FACTS)
    assert r.ok and not r.warnings, (r.errors, r.warnings)

    # --- errors ------------------------------------------------------------
    check("missing dataset", validate(cfg(), None, FACTS),
          errors=1, must="was not found")
    check("no models", validate(cfg(models=[]), DS, FACTS),
          errors=1, must="at least one model")
    check("unknown model", validate(cfg(models=["svc", "qnn"]), DS, FACTS),
          errors=1, must="Unknown model")
    check("duplicate model", validate(cfg(models=["svc", "svc"]), DS, FACTS),
          errors=1, must="more than once")
    check("zz over cap", validate(cfg(encoding="zz", n_features=12), DS, FACTS),
          errors=1, must="at most 10 features")
    check("unknown encoding", validate(cfg(encoding="magic"), DS, FACTS),
          errors=1, must="Unknown encoding")
    check("unknown backbone", validate(cfg(backbone="resnet50"), DS, FACTS),
          errors=1, must="Unknown backbone")
    # 900 features legitimately breaks three independent rules at once:
    # the backbone dim, the training-row count, and the encoding cap.
    check("features > dim", validate(cfg(n_features=900), DS, FACTS),
          errors=3, must="only 512 dimensions")
    check("one feature", validate(cfg(n_features=1), DS, FACTS),
          errors=1, must="at least 2")
    check("test size", validate(cfg(test_size=0.9), DS, FACTS),
          errors=1, must="between 0.05 and 0.5")
    check("unknown op", validate(cfg(ops=[OpConfig("sharpen")]), DS, FACTS),
          errors=1, must="Unknown preprocessing step")
    check("bad enum", validate(cfg(ops=[OpConfig("binarize", {"method": "x"})]),
                               DS, FACTS), errors=1, must="must be one of")

    tiny = DatasetMeta("d_t", "t", "image_folder", ["A", "B"],
                       {"A": 1, "B": 30}, 31, 2)
    r = validate(cfg(), tiny, FACTS)
    assert not r.ok and any("stratified" in e for e in r.errors), r.errors

    # PCA cannot exceed the training rows
    small = DatasetMeta("d_s", "s", "image_folder", ["A", "B"],
                        {"A": 12, "B": 12}, 24, 2)
    check("pca > n_train", validate(cfg(n_features=30), small, FACTS),
          errors=2, must="PCA cannot produce")      # also over the angle_y cap

    # --- warnings only, still runnable -------------------------------------
    r = validate(cfg(models=["svc", "rf"]), DS, FACTS)
    assert r.ok and any("No quantum model" in w for w in r.warnings)

    r = validate(cfg(models=["qsvc"]), DS, FACTS)
    assert r.ok and any("No classical model" in w for w in r.warnings)

    r = validate(cfg(ops=[]), DS, FACTS)
    assert r.ok and any("No preprocessing" in w for w in r.warnings)

    r = validate(cfg(ops=[OpConfig("denoise", {"method": "nlmeans"})]), DS, FACTS)
    assert r.ok and any("five minutes" in w for w in r.warnings)

    r = validate(cfg(n_features=12, encoding="angle_y"), DS, FACTS)
    assert r.ok and any("4,096-dimensional" in w for w in r.warnings), r.warnings

    # amplitude is the demo moment: 16 features, 4 qubits, no warning
    r = validate(cfg(n_features=16, encoding="amplitude"), DS, FACTS)
    assert r.ok and not r.warnings, r.warnings

    # ...but 2 features on amplitude is 1 qubit, which no quantum model can use
    check("one qubit", validate(cfg(n_features=2, encoding="amplitude"), DS, FACTS),
          errors=1, must="need at least 2")
    # the same config is fine with no quantum model selected
    assert validate(cfg(n_features=2, encoding="amplitude",
                        models=["svc", "rf"]), DS, FACTS).ok

    r = validate(cfg(ops=[OpConfig("binarize", {"thr": 9999})]), DS, FACTS)
    assert r.ok and any("clamped" in w for w in r.warnings)

    imb = DatasetMeta("d_i", "i", "image_folder", ["A", "B"],
                      {"A": 400, "B": 40}, 440, 2)
    r = validate(cfg(), imb, FACTS)
    assert r.ok and any("imbalanced" in w for w in r.warnings)

    # --- degraded mode: whichever W4 registries are absent ------------------
    live = collect_facts()
    print(f"live registries missing: {live.missing or 'none'}")
    r = validate(cfg(encoding="does_not_exist", models=["not_a_model"]), DS, live)
    blob = " ".join(r.errors)
    # Each rule switches on independently as its registry lands, so assert
    # per-registry rather than on `complete`.
    if live.encodings is not None:
        assert "Unknown encoding" in blob, r.errors
    else:
        assert "Unknown encoding" not in blob
    if live.models is not None:
        assert "Unknown model" in blob, r.errors
    else:
        assert "Unknown model" not in blob

    print("validate OK — 15 error rules, 8 warning rules, degrades cleanly "
          "without W4")
