"""
backend/core/models_quantum.py  --  SEAM 4, quantum half
========================================================

Owner: W4. The highest-risk file in the repo.

Two models, both PennyLane, both `default.qubit` statevector simulation:

  QSVCModel  -- fidelity quantum kernel + SVC(kernel="precomputed")
  VQCModel   -- variational circuit + linear readout head, trained end to end


WHY THERE IS NO TORCH IN HERE
-----------------------------
The obvious VQC is `qml.qnn.TorchLayer` inside an `nn.Sequential`. It is not
worth the dependency. PennyLane's autograd interface with
`diff_method="backprop"` on `default.qubit` differentiates the simulator
directly -- exact gradients, one backward pass, no parameter-shift overhead --
and a linear head is a matmul. So the entire model layer needs only

    numpy + scikit-learn + pennylane

Torch stays confined to embed.py, where a pretrained CNN genuinely requires it.
One fewer framework is one fewer install failure the night before the demo, and
`benchmark.py` can run on a machine that has no deep-learning stack at all.

The `Linear(q -> n_classes)` head still does its job: n_classes is a constructor
argument, the head is shaped from it, and 2-class and 4-class datasets take the
identical code path with no special-casing.


WHY THE GRAM MATRIX IS COMPUTED FROM STATEVECTORS
-------------------------------------------------
`FidelityQuantumKernel` evaluates a separate compute-uncompute circuit per PAIR,
which is O(n^2) simulations -- the notebook measured this and it is why cell 10
exists. Simulating each sample ONCE and taking inner products is O(n)
simulations plus one matmul:

    S = [state(x_i)]           n simulations,  shape (n, 2^q)
    K = |S S-dagger|^2         one BLAS call

Identical numbers, and for n=740 it is the difference between minutes and
seconds. This is the same trick as notebook cell 11, ported.

It is exact simulation, not sampling: there is no shot noise, so K is exactly
PSD and SVC never sees an indefinite kernel. Say "noiseless statevector
simulation" when asked, not "we ran it on a quantum computer".
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pennylane as qml
from numpy.typing import NDArray
from pennylane import numpy as pnp
import warnings

from sklearn.svm import SVC

# sklearn 1.9 deprecated SVC(probability=True) in favour of
# CalibratedClassifierCV(SVC(), ensemble=False); removal lands in 1.11.
# We stay on probability=True for Phase 1: the replacement does not slot into
# GridSearchCV without renaming every grid key to estimator__*, and it does not
# slot into a PRECOMPUTED kernel at all without re-solving the both-axes
# slicing problem QSVC._cv_score exists to handle. scikit-learn is pinned to
# 1.9.1, so nothing breaks. Without this filter the warning fires once per
# candidate per fold -- roughly 130 lines that bury the results table.
# Phase 2: revisit when unpinning sklearn.
warnings.filterwarnings(
    "ignore",
    message=r".*`probability` parameter was deprecated.*",
    category=FutureWarning,
)

from .contracts import BaseModel, EncodingSpec, FeatureMatrix, LabelVector
from .encodings import get_encoding

__all__ = ["QSVCModel", "VQCModel"]


# ---------------------------------------------------------------------------
# Shared quantum plumbing
# ---------------------------------------------------------------------------

class _QuantumModel(BaseModel):
    """Device/QNode lifecycle and telemetry, shared by both quantum models.

    A QNode holds a reference to a device and to a traced tape; neither pickles.
    CheckpointBundle pickles the whole estimator, so every quantum model MUST
    drop its QNode on the way out and rebuild it on the way in. Doing that here
    once means a Phase-2 model cannot forget to.
    """

    kind = "quantum"
    backend_name = "pennylane.default.qubit"

    # Searched by QSVCModel when bandwidth="auto". Spans three orders of
    # magnitude because the usable window moves with both the encoding and
    # n_features -- see the measured table in encodings.py.
    BANDWIDTH_GRID = (1.0, 0.5, 0.25, 0.1, 0.05)

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        if self.encoding is None:
            raise ValueError(f"{self.name}: quantum models require an EncodingSpec")
        self.n_qubits: int = self.encoding.qubits_for(self.n_features)
        self.bandwidth: float = 1.0     # resolved during _fit when "auto"
        self._dev = None
        self._qnode = None

    def _scale(self, X: FeatureMatrix) -> NDArray[np.float64]:
        """Apply the bandwidth multiplier.

        X arrives already scaled by project.py into the encoding's conventional
        range. Bandwidth shrinks that range towards zero to pull the states
        closer together and stop the kernel concentrating onto the identity.

        Amplitude embedding is exempt: its input must stay unit-norm, and
        scaling a unit vector by a constant then re-normalising is the identity
        map. Bandwidth is meaningless there, so it is ignored rather than
        silently doing nothing surprising.
        """
        X = np.asarray(X, dtype=np.float64)
        if self.encoding.name == "amplitude" or self.bandwidth == 1.0:
            return X
        return X * float(self.bandwidth)

    # -- device / qnode ----------------------------------------------------
    @property
    def device(self):
        if self._dev is None:
            self._dev = qml.device("default.qubit", wires=self.n_qubits)
        return self._dev

    def _wires(self) -> range:
        return range(self.n_qubits)

    # -- pickling ----------------------------------------------------------
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_dev"] = None
        state["_qnode"] = None
        # EncodingSpec holds function references. They are module-level so they
        # would pickle, but storing the NAME means a checkpoint stays loadable
        # after the registry is edited, which is the actual requirement.
        state["encoding"] = self.encoding.name
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        enc = state.get("encoding")
        self.__dict__.update(state)
        if isinstance(enc, str):
            self.encoding = get_encoding(enc)

    # -- telemetry ---------------------------------------------------------
    def _circuit_resources(self) -> tuple[int, int]:
        """(depth, two_qubit_gates) measured from the real decomposed circuit.

        `level="device"` matters: at the default level a template counts as ONE
        gate spanning N wires, so AngleEmbedding on 8 qubits would report
        "1 gate, 8-qubit". Only the device level expands it into RY x8.
        """
        spec = self.encoding
        x = np.zeros(
            spec.qubits_for(self.n_features)
            if spec.name == "amplitude"
            else self.n_features,
            dtype=np.float64,
        )
        if spec.name == "amplitude":
            x = np.zeros(2 ** self.n_qubits)
            x[0] = 1.0

        def _probe(xx):
            self._apply_encoding(xx)
            return qml.state()

        try:
            res = qml.specs(
                qml.QNode(_probe, qml.device("default.qubit", wires=self.n_qubits)),
                level="device",
            )(x).resources
            two_q = sum(c for size, c in res.gate_sizes.items() if size >= 2)
            return int(res.depth), int(two_q)
        except Exception:
            # Telemetry must never be the thing that fails a run.
            return 0, 0

    def _apply_encoding(self, x) -> None:
        self.encoding.template(x, wires=self._wires())

    def quantum_telemetry(self) -> dict[str, Any]:
        depth, two_q = self._circuit_resources()
        return {
            "n_qubits": int(self.n_qubits),
            "encoding": self.encoding.name,
            "circuit_depth": depth,
            "two_qubit_gates": two_q,
            "state_memory_mb": round(
                self.encoding.state_memory_mb(self.n_features), 4
            ),
            # Reported by every quantum model, not just QSVC: a VQC inherits
            # whatever bandwidth it was given, and that number explains a
            # score near chance more often than the ansatz does.
            "bandwidth": float(self.bandwidth),
        }


# ---------------------------------------------------------------------------
# QSVC -- fidelity quantum kernel
# ---------------------------------------------------------------------------

class QSVCModel(_QuantumModel):
    """K(x, y) = |<phi(x)|phi(y)>|^2, then a precomputed-kernel SVC."""

    name = "qsvc"
    label = "Quantum Kernel SVC"
    param_schema = {
        "tune": {
            "type": "bool", "default": True, "label": "Tune bandwidth + C",
            "help": "Cross-validates the kernel bandwidth. An untuned fidelity "
                    "kernel can score below chance for reasons unrelated to "
                    "whether quantum helps. Leave this on.",
        },
        "bandwidth": {
            "type": "float", "min": 0.01, "max": 1.0, "default": 1.0,
            "label": "Bandwidth", "help": "Used only when tuning is off. "
                                          "1.0 = the encoding's natural range.",
        },
        "C": {
            "type": "float", "min": 0.01, "max": 100.0, "default": 1.0,
            "label": "C", "help": "Used only when tuning is off.",
        },
        "cv": {"type": "int", "min": 2, "max": 10, "default": 5, "label": "CV folds"},
        "batch_size": {
            "type": "int", "min": 16, "max": 1024, "default": 256,
            "label": "Sim batch", "help": "Statevectors simulated per call. "
                                          "Lower this if memory is tight.",
        },
    }

    # -- statevectors ------------------------------------------------------
    def _state_qnode(self):
        if self._qnode is None:
            @qml.qnode(self.device)
            def _state(x):
                self._apply_encoding(x)
                return qml.state()

            self._qnode = _state
        return self._qnode

    def _states(self, X: FeatureMatrix) -> NDArray[np.complex64]:
        """float32[n, N] -> complex64[n, 2^q].

        Chunked because the intermediate is n x 2^q complex128: at q=12 and
        n=1000 a single call would allocate 65 MB before the downcast, and the
        simulator allocates working copies on top of that.
        """
        qn = self._state_qnode()
        Xb = self._scale(X)
        bs = max(int(self.params.get("batch_size", 256)), 1)
        out = np.empty((len(X), 2 ** self.n_qubits), dtype=np.complex64)
        for i in range(0, len(X), bs):
            out[i : i + bs] = np.asarray(qn(Xb[i : i + bs]), dtype=np.complex64)
        return out

    @staticmethod
    def _gram(A: NDArray[np.complex64], B: NDArray[np.complex64]) -> NDArray[np.float64]:
        K = np.abs(A @ B.conj().T) ** 2
        return np.clip(K.astype(np.float64), 0.0, 1.0)   # float32 drift -> 1.0000002

    # -- fit / predict -----------------------------------------------------
    def _fit(self, X: FeatureMatrix, y: LabelVector) -> None:
        from .models_classical import safe_cv_folds   # local: avoids a cycle

        tune = bool(self.params.get("tune", True))
        folds = safe_cv_folds(y, int(self.params.get("cv", 5))) if tune else 0
        c_grid = [0.1, 1.0, 10.0, 100.0]

        t0 = time.perf_counter()

        if folds >= 2:
            # Each bandwidth is a different feature map, so the statevectors
            # have to be recomputed per candidate -- but only |grid| times, not
            # once per CV fold: the Gram matrix is reused across folds by
            # slicing it, which is the whole advantage of a precomputed kernel.
            best = (-np.inf, 1.0, 1.0)
            for bw in self.BANDWIDTH_GRID:
                self.bandwidth = bw
                S = self._states(X)
                K = self._gram(S, S)
                for C in c_grid:
                    score = self._cv_score(K, y, C, folds)
                    if score > best[0]:
                        best = (score, bw, C)
                if self.encoding.name == "amplitude":
                    break          # bandwidth is a no-op there; one pass only
            self._cv_best_score: float | None = float(best[0])
            self.bandwidth, C_best = float(best[1]), float(best[2])
        else:
            self._cv_best_score = None
            self.bandwidth = float(self.params.get("bandwidth", 1.0))
            C_best = float(self.params.get("C", 1.0))

        self._train_states = self._states(X)
        self._kernel_seconds = time.perf_counter() - t0

        K = self._gram(self._train_states, self._train_states)
        self._est = SVC(
            kernel="precomputed", C=C_best, probability=True,
            random_state=self.seed,
        ).fit(K, y)
        self._best_params = {"bandwidth": self.bandwidth, "C": C_best}

    def _cv_score(self, K: NDArray[np.float64], y: LabelVector, C: float,
                  folds: int) -> float:
        """Stratified CV on a precomputed kernel.

        Written out rather than handed to GridSearchCV because a precomputed
        kernel has to be sliced on BOTH axes -- K[train][:, train] to fit and
        K[test][:, train] to score. Passing K straight to GridSearchCV silently
        slices rows only and scores against the wrong columns.
        """
        from sklearn.metrics import f1_score
        from sklearn.model_selection import StratifiedKFold

        scores = []
        skf = StratifiedKFold(folds, shuffle=True, random_state=self.seed)
        for tr, te in skf.split(K, y):
            est = SVC(kernel="precomputed", C=C).fit(K[np.ix_(tr, tr)], y[tr])
            scores.append(
                f1_score(y[te], est.predict(K[np.ix_(te, tr)]), average="macro")
            )
        return float(np.mean(scores))

    def _predict_proba(self, X: FeatureMatrix) -> NDArray[np.float64]:
        # Rows index test samples, columns index TRAINING samples, in the exact
        # order SVC saw at fit time. Transposing this is the classic silent bug:
        # the shapes are compatible when n_test == n_train, so it only breaks
        # on the demo dataset.
        K = self._gram(self._states(X), self._train_states)
        p = self._est.predict_proba(K)
        if p.shape[1] != self.n_classes:
            full = np.zeros((len(X), self.n_classes), dtype=np.float64)
            full[:, self._est.classes_.astype(int)] = p
            return full
        return p

    def n_trainable_params(self) -> int | None:
        return int(self._est.n_support_.sum()) if self._fitted else None

    @property
    def best_params(self) -> dict[str, Any]:
        """Shown on the Benchmark screen. The bandwidth the kernel actually
        used is the single most load-bearing number in the whole run."""
        return dict(getattr(self, "_best_params", {}))

    def quantum_telemetry(self) -> dict[str, Any]:
        t = super().quantum_telemetry()
        if self._fitted:
            # The Gram matrix, not one statevector, is what actually sits in RAM.
            n = len(self._train_states)
            t["state_memory_mb"] = round(
                n * (2 ** self.n_qubits) * 8 / (1024 ** 2), 4
            )
        return t


# ---------------------------------------------------------------------------
# VQC -- variational classifier
# ---------------------------------------------------------------------------

class VQCModel(_QuantumModel):
    """encoding -> StronglyEntanglingLayers(trainable) -> <Z_i> -> Linear -> softmax.

    The readout is q expectation values fed to a (q x n_classes) linear head.
    Dynamic class count is free: the head is built from `self.n_classes` and
    nothing else in the model knows how many classes there are.
    """

    name = "vqc"
    label = "Variational Quantum Classifier"
    param_schema = {
        "n_layers": {"type": "int", "min": 1, "max": 6, "default": 3,
                     "label": "Ansatz layers"},
        "epochs": {"type": "int", "min": 5, "max": 200, "default": 40,
                   "label": "Epochs"},
        "lr": {"type": "float", "min": 0.001, "max": 0.5, "default": 0.05,
               "label": "Learning rate"},
        "batch_size": {"type": "int", "min": 4, "max": 128, "default": 24,
                       "label": "Batch size"},
        "bandwidth": {
            "type": "float", "min": 0.01, "max": 1.0, "default": 1.0,
            "label": "Encoding bandwidth",
            "help": "Not auto-tuned here -- a VQC fit is far too expensive to "
                    "cross-validate five times. Read the value QSVC picked off "
                    "the benchmark table and set it here.",
        },
    }

    @property
    def best_params(self) -> dict[str, Any]:
        return {
            "n_layers": int(getattr(self, "_n_layers", 0)),
            "bandwidth": float(self.bandwidth),
        }

    def _fit(self, X: FeatureMatrix, y: LabelVector) -> None:
        if self.n_qubits < 2:
            raise ValueError(
                f"vqc needs >= 2 qubits, got {self.n_qubits}. With "
                f"'{self.encoding.name}' that means at least "
                f"{3 if self.encoding.name == 'amplitude' else 2} features."
            )

        n_layers = int(self.params.get("n_layers", 3))
        epochs = int(self.params.get("epochs", 40))
        lr = float(self.params.get("lr", 0.05))
        bs = int(self.params.get("batch_size", 24))
        rng = np.random.default_rng(self.seed)

        @qml.qnode(self.device, interface="autograd", diff_method="backprop")
        def circuit(x, weights):
            self._apply_encoding(x)
            qml.StronglyEntanglingLayers(weights, wires=self._wires())
            return [qml.expval(qml.PauliZ(i)) for i in self._wires()]

        self._qnode = circuit
        w_shape = qml.StronglyEntanglingLayers.shape(n_layers, self.n_qubits)

        # Small init keeps the circuit near identity for the first few steps.
        # Large random weights start deep in a barren plateau and the loss never
        # moves, which reads as "quantum models do not work" rather than
        # "this initialisation was wrong".
        weights = pnp.array(rng.normal(0, 0.1, w_shape), requires_grad=True)
        W = pnp.array(
            rng.normal(0, 0.5, (self.n_qubits, self.n_classes)), requires_grad=True
        )
        b = pnp.zeros(self.n_classes, requires_grad=True)

        self.bandwidth = float(self.params.get("bandwidth", 1.0))
        Xp = pnp.array(self._scale(X), requires_grad=False)
        yp = np.asarray(y, dtype=int)

        def cost(weights, W, b, Xb=None, yb=None):
            feats = qml.math.stack(circuit(Xb, weights), axis=-1)   # (n, q)
            z = feats @ W + b
            z = z - qml.math.max(z, axis=1, keepdims=True)          # logsumexp shift
            p = pnp.exp(z) / pnp.sum(pnp.exp(z), axis=1, keepdims=True)
            return -pnp.mean(pnp.log(p[pnp.arange(len(yb)), yb] + 1e-12))

        opt = qml.AdamOptimizer(lr)
        n = len(Xp)
        self._loss_history: list[float] = []

        for _ in range(epochs):
            order = rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            for s in range(0, n, bs):
                idx = order[s : s + bs]
                if len(idx) < 2:
                    continue
                (weights, W, b), loss = opt.step_and_cost(
                    cost, weights, W, b,
                    Xb=Xp[idx], yb=pnp.array(yp[idx], requires_grad=False),
                )
                epoch_loss += float(loss)
                n_batches += 1
            if n_batches:
                self._loss_history.append(epoch_loss / n_batches)

        # Demote autograd tensors to plain numpy so the checkpoint pickles
        # without dragging autograd's tracing machinery along.
        self._weights = np.asarray(weights, dtype=np.float64)
        self._W = np.asarray(W, dtype=np.float64)
        self._b = np.asarray(b, dtype=np.float64)
        self._n_layers = n_layers

    def _forward(self, X: FeatureMatrix) -> NDArray[np.float64]:
        @qml.qnode(self.device)
        def circuit(x, weights):
            self._apply_encoding(x)
            qml.StronglyEntanglingLayers(weights, wires=self._wires())
            return [qml.expval(qml.PauliZ(i)) for i in self._wires()]

        out = circuit(self._scale(X), self._weights)
        return np.stack([np.asarray(o) for o in out], axis=-1)

    def _predict_proba(self, X: FeatureMatrix) -> NDArray[np.float64]:
        z = self._forward(X) @ self._W + self._b
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def n_trainable_params(self) -> int | None:
        if not self._fitted:
            return None
        return int(self._weights.size + self._W.size + self._b.size)

    @property
    def loss_history(self) -> list[float]:
        """Per-epoch mean loss. A flat line here is a barren plateau, and it is
        the first thing to look at when a VQC lands at chance accuracy."""
        return list(getattr(self, "_loss_history", []))

    def quantum_telemetry(self) -> dict[str, Any]:
        t = super().quantum_telemetry()
        if self._fitted:
            depth, two_q = self._ansatz_resources()
            t["circuit_depth"] = depth
            t["two_qubit_gates"] = two_q
        return t

    def _ansatz_resources(self) -> tuple[int, int]:
        """Encoding + ansatz together. The base class measures the encoding
        alone, which understates a VQC by the whole trainable block."""
        try:
            def _probe(x, w):
                self._apply_encoding(x)
                qml.StronglyEntanglingLayers(w, wires=self._wires())
                return qml.state()

            x = (
                np.eye(2 ** self.n_qubits)[0]
                if self.encoding.name == "amplitude"
                else np.zeros(self.n_features)
            )
            res = qml.specs(
                qml.QNode(_probe, qml.device("default.qubit", wires=self.n_qubits)),
                level="device",
            )(x, self._weights).resources
            return int(res.depth), int(
                sum(c for size, c in res.gate_sizes.items() if size >= 2)
            )
        except Exception:
            return 0, 0
