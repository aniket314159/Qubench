"""
scripts/test_models.py
======================

The n_classes hardcode hunt, run as a script.

Every model x {2 classes, 4 classes} x {angle_y, zz, amplitude}, on synthetic
data shaped like real PCA output. No images, no backbone, no disk -- this runs
in seconds and catches the class of bug that otherwise shows up on Day 4 when
someone uploads a binary dataset.

    python -m scripts.test_models
"""

from __future__ import annotations

import pickle
import sys
import time

import numpy as np
from sklearn.datasets import make_classification

# Run either way: `python -m scripts.test_models` or `python scripts/test_models.py`.
# The second form puts scripts/ on sys.path, not the repo root, so `import
# backend` fails without this. Every other script in here does the same.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from backend.core.contracts import Telemetry
from backend.core.encodings import ENCODINGS, get_encoding
from backend.core.registry import MODEL_REGISTRY, catalog


def make_data(n_classes: int, n_features: int = 6, n: int = 120, seed: int = 0):
    X, y = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=max(2, n_features - 2),
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=seed,
    )
    cut = int(0.8 * n)
    return (
        X[:cut].astype(np.float32), y[:cut].astype(np.int64),
        X[cut:].astype(np.float32), y[cut:].astype(np.int64),
    )


def run_one(model_name: str, enc_name: str, n_classes: int) -> dict:
    enc = get_encoding(enc_name)
    n_features = 6
    Xtr, ytr, Xte, yte = make_data(n_classes, n_features)

    # Exactly what project.py will do: fit the encoding's scaler on TRAIN only.
    scaler = enc.make_scaler().fit(Xtr)
    Xtr_s = np.asarray(scaler.transform(Xtr), dtype=np.float32)
    Xte_s = np.asarray(scaler.transform(Xte), dtype=np.float32)

    cls = MODEL_REGISTRY[model_name]
    params = {"epochs": 8} if model_name == "vqc" else {}

    t0 = time.perf_counter()
    m = cls(
        n_classes=n_classes,
        n_features=n_features,
        encoding=enc,
        seed=42,
        **params,
    ).fit(Xtr_s, ytr)

    proba = m.predict_proba(Xte_s)
    pred = m.predict(Xte_s)
    wall = time.perf_counter() - t0

    assert proba.shape == (len(Xte), n_classes), proba.shape
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert pred.min() >= 0 and pred.max() < n_classes

    tel = m.telemetry()
    assert isinstance(tel, Telemetry)
    assert set(tel.to_dict()) == {
        "fit_seconds", "predict_seconds", "backend", "n_params", "n_qubits",
        "encoding", "circuit_depth", "two_qubit_gates", "state_memory_mb",
        "bandwidth",
    }
    if cls.kind == "quantum":
        assert tel.n_qubits == enc.qubits_for(n_features)
        assert tel.encoding == enc_name
        assert tel.circuit_depth and tel.circuit_depth > 0, "no depth telemetry"
    else:
        assert tel.n_qubits is None and tel.encoding is None

    # Pickle round-trip. Diagnose loads a checkpoint in a FRESH process, so a
    # model that only works in the process that trained it is a model that only
    # works until the demo.
    restored = pickle.loads(pickle.dumps(m))
    assert np.allclose(restored.predict_proba(Xte_s), proba, atol=1e-6), (
        f"{model_name}/{enc_name}: predictions changed across pickle"
    )

    return {
        "acc": float((pred == yte).mean()),
        "wall": wall,
        "tel": tel,
        "params": m.n_trainable_params(),
    }


def main() -> int:
    print(f"{len(catalog())} models x {len(ENCODINGS)} encodings\n")
    failures: list[str] = []

    for n_classes in (2, 4):
        print(f"{'=' * 78}\n{n_classes} CLASSES\n{'=' * 78}")
        print(f"{'model':6s} {'encoding':10s} {'acc':>6s} {'qubits':>6s} "
              f"{'depth':>6s} {'2q':>4s} {'params':>7s} {'fit s':>7s}")
        print("-" * 78)

        for model_name in MODEL_REGISTRY:
            # Classical models ignore the encoding, so running them against all
            # four would be three identical rows. One is enough to prove they
            # accept the argument.
            encs = (
                list(ENCODINGS)
                if MODEL_REGISTRY[model_name].kind == "quantum"
                else ["angle_y"]
            )
            for enc_name in encs:
                try:
                    r = run_one(model_name, enc_name, n_classes)
                    t = r["tel"]
                    print(
                        f"{model_name:6s} {enc_name:10s} {r['acc']:6.3f} "
                        f"{str(t.n_qubits or '-'):>6s} {str(t.circuit_depth or '-'):>6s} "
                        f"{str(t.two_qubit_gates if t.two_qubit_gates is not None else '-'):>4s} "
                        f"{str(r['params'] or '-'):>7s} {t.fit_seconds:7.2f}"
                    )
                except Exception as e:
                    failures.append(f"{model_name}/{enc_name}/{n_classes}c: {e}")
                    print(f"{model_name:6s} {enc_name:10s}   FAIL  {type(e).__name__}: {e}")
        print()

    # The contract that keeps the frontend table renderable: identical key sets.
    Xtr, ytr, Xte, _ = make_data(3, 6)
    enc = get_encoding("angle_y")
    s = enc.make_scaler().fit(Xtr)
    keys = set()
    for name, cls in MODEL_REGISTRY.items():
        kw = {"epochs": 5} if name == "vqc" else {}
        m = cls(n_classes=3, n_features=6, encoding=enc, seed=0, **kw)
        m.fit(np.asarray(s.transform(Xtr), np.float32), ytr)
        keys.add(tuple(sorted(m.telemetry().to_dict())))
    assert len(keys) == 1, "telemetry key sets diverged across models"
    print("telemetry key set identical across all 4 models -- one table renders both kinds")

    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print("\nmodel layer OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
