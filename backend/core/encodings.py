"""
backend/core/encodings.py  --  SEAM 3.5
=======================================

The encoding registry. Owner: W4.

One EncodingSpec binds three things that MUST move together:

    qubit count   <--  qubits_for(n_features)
    scaler        <--  make_scaler()          (fitted on TRAIN only, in project.py)
    circuit       <--  template(x, wires)

Picking an encoding in the UI picks all three. They can never be mismatched
because there is no way to set one without the other two.

PennyLane only. `import qiskit` must not appear in this package.


WHY THESE FOUR, AND NOT "ANGLE + A CNOT RING"
---------------------------------------------
For a fidelity kernel K(x,y) = |<phi(x)|phi(y)>|^2, applying a FIXED unitary U
after the data encoding changes nothing:

    |<phi(x)| U-dagger U |phi(y)>|^2  ==  |<phi(x)|phi(y)>|^2

So `AngleEmbedding` followed by a ring of CNOTs produces a Gram matrix that is
numerically identical to plain `AngleEmbedding`. It looks entangled in a circuit
diagram and buys exactly zero expressivity in the kernel. `test_encodings.py`
asserts this, so nobody re-adds it by accident.

Entanglement only earns its place two ways, and the registry has one of each:

  * `zz`        -- the two-qubit rotations are DATA-DEPENDENT: RZ(f(x_i, x_j)).
  * `angle_x2`  -- re-encode AFTER entangling, so U sits between two data layers.

`angle_y` is kept deliberately as the honest product-kernel baseline. It is
efficiently simulable classically, and saying so out loud is a stronger position
in a viva than pretending otherwise. It is also the control that makes
"entanglement bought us X points" a measurable claim rather than an assertion.


SCALER RANGES ARE NOT DECORATION
--------------------------------
An encoding maps a real number onto a rotation angle. RY(0) and RY(2*pi) are the
same state, so feeding unbounded PCA output into a rotation silently wraps
distant samples onto identical states and destroys the kernel. Every spec here
declares the range its circuit is valid over. The existing Qiskit notebook fed
`X_train_raw.npy` (unscaled TruncatedSVD output) straight into a ZZ feature map;
that is exactly the bug this structure makes impossible to reproduce.

The range below is the CONVENTIONAL range for each circuit. It is not
automatically the range that generalises best -- see below.


KERNEL BANDWIDTH  (measured, not assumed)
-----------------------------------------
Fidelity kernels concentrate. Measured on 6 features / 6 qubits / 4 classes:

    ZZ, MinMax [0, 2*pi] :  mean off-diagonal K = 0.020  ->  accuracy 0.367
    ZZ, MinMax [0, 0.25] :  mean off-diagonal K = 0.361  ->  accuracy 0.867
    ZZ, MinMax [0, 0.02] :  mean off-diagonal K = 0.989  ->  accuracy 0.433

Too wide and every state is orthogonal to every other, K -> I, and the SVC
memorises the training set and generalises at chance. Too narrow and every state
is the same state, K -> J, and there is nothing to separate. The usable window
sits in between and it is NARROWER for `zz` than for `angle_y`, because the
pairwise term (pi - x_i)(pi - x_j) grows quadratically in the input scale.

So the correct bandwidth is not a property of the encoding alone -- it also
moves with n_features. Hardcoding it here would be a guess. Instead the scalers
below fix the conventional range, and the quantum models take a `bandwidth`
multiplier that is cross-validated on the training set alongside C. See
models_quantum.QSVCModel.

This is a known result (Shaydulin & Wild 2022; Canatar et al. 2022) and it is
worth a slide: an untuned quantum kernel can look worse than chance for reasons
that have nothing to do with whether quantum helps.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from .contracts import EncodingSpec

__all__ = [
    "circuit_shape", "bloch_angles", "bloch_demo","ENCODINGS", "L2Normalizer", "get_encoding", "catalog"]


# ---------------------------------------------------------------------------
# Scalers
# ---------------------------------------------------------------------------

def _minmax_0_pi() -> MinMaxScaler:
    """RY(theta) has period 2*pi and RY(0) != RY(pi), so [0, pi] uses the full
    distinguishable range of a single rotation without wrapping."""
    return MinMaxScaler(feature_range=(0.0, np.pi))


def _minmax_0_2pi() -> MinMaxScaler:
    """The ZZ map applies RZ(2*x_i) and RZ(2*(pi-x_i)(pi-x_j)). Qiskit's
    convention assumes inputs already live on [0, 2*pi]."""
    return MinMaxScaler(feature_range=(0.0, 2.0 * np.pi))


class L2Normalizer(BaseEstimator, TransformerMixin):
    """Row-wise L2 normalisation for amplitude embedding.

    Amplitude embedding needs a unit vector, not a bounded one, so MinMax is the
    wrong transform here.

    It deliberately does NOT pad to a power of two. Padding is a circuit
    concern, and `AmplitudeEmbedding(pad_with=0.0)` already does it -- padding
    with zeros preserves the L2 norm, so there is nothing to re-normalise
    afterwards. Doing it here instead would widen X from N columns to
    2^ceil(log2 N) and break the invariant: the model was constructed for
    `n_features=N` and `BaseModel._check_X` rejects anything else. (It did
    reject it. That is how this comment came to exist.)

    What it does guard: a zero row has no direction to normalise to and would
    produce NaN, which surfaces 200 lines later as an all-zero `predict_proba`
    row rather than as an error here. Zero rows become the uniform unit vector.

    Must stay a module-level class: it is pickled inside CheckpointBundle.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"L2Normalizer fitted on {self.n_features_in_} features, "
                f"got {X.shape[1]}"
            )
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        dead = norms.ravel() < 1e-12
        if dead.any():
            X = X.copy()
            X[dead] = 1.0 / np.sqrt(X.shape[1])
            norms[dead] = 1.0
        return (X / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Qubit-count formulas   (module-level, never lambdas -- see note at bottom)
# ---------------------------------------------------------------------------

def _q_identity(n_features: int) -> int:
    return int(n_features)


def _q_log2(n_features: int) -> int:
    return int(np.ceil(np.log2(max(int(n_features), 2))))


def _tq_zero(n_features: int) -> int:
    return 0


def _tq_ring(n_features: int) -> int:
    """One CNOT ring. A 1-qubit "ring" is empty; 2 qubits need only one link."""
    n = int(n_features)
    return 0 if n < 2 else (1 if n == 2 else n)


def _tq_zz_full(n_features: int) -> int:
    """Full entanglement: every unordered pair, two CNOTs each."""
    n = int(n_features)
    return n * (n - 1)


# ---------------------------------------------------------------------------
# Circuit templates:  (x, wires) -> applies gates, returns nothing
# ---------------------------------------------------------------------------

def angle_y_template(x, wires) -> None:
    """One RY(x_i) per qubit. Product state, zero entanglement."""
    qml.AngleEmbedding(x, wires=wires, rotation="Y")


def angle_x2_template(x, wires) -> None:
    """Encode, entangle, re-encode.

    The CNOT ring sits BETWEEN two data layers, so it does not cancel out of the
    fidelity kernel the way a trailing ring does. RZ on the second pass rather
    than RY because a second RY would partly fold back into the first.
    """
    wires = list(wires)
    qml.AngleEmbedding(x, wires=wires, rotation="Y")
    if len(wires) > 1:
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
        if len(wires) > 2:
            qml.CNOT(wires=[wires[-1], wires[0]])
    qml.AngleEmbedding(x, wires=wires, rotation="Z")


def zz_template(x, wires) -> None:
    """Second-order Pauli-Z evolution, full entanglement, reps=1.

    Gate-for-gate equivalent to the notebook's
    `zz_feature_map(feature_dimension=N, reps=1, entanglement="full")`, so the
    Qiskit results stay comparable after the port.

        H on every wire
        RZ(2*x_i) on wire i
        for each pair i<j:  CNOT(i,j), RZ(2*(pi-x_i)*(pi-x_j)) on j, CNOT(i,j)

    The pair term is where entanglement actually buys expressivity: the rotation
    angle depends on TWO features at once, which no product kernel can express.
    """
    wires = list(wires)
    n = len(wires)
    for i, w in enumerate(wires):
        qml.Hadamard(wires=w)
        qml.RZ(2.0 * x[..., i], wires=w)
    for i in range(n):
        for j in range(i + 1, n):
            qml.CNOT(wires=[wires[i], wires[j]])
            qml.RZ(2.0 * (np.pi - x[..., i]) * (np.pi - x[..., j]), wires=wires[j])
            qml.CNOT(wires=[wires[i], wires[j]])


def amplitude_template(x, wires) -> None:
    """Load the (already unit-norm, already padded) vector as amplitudes.

    `normalize=True` is belt-and-braces: L2Normalizer has done it, but float32
    round-trips can leave the norm a few ULPs off and PennyLane raises on that.
    """
    qml.AmplitudeEmbedding(x, wires=wires, normalize=True, pad_with=0.0)


# ---------------------------------------------------------------------------
# THE REGISTRY
# ---------------------------------------------------------------------------

def _bloch_ry(x, bandwidth: float = 1.0) -> list[dict]:
    """RY(theta) from |0>: the arrow tilts in the XZ plane by theta itself."""
    return [{"theta": float(v) * float(bandwidth), "phi": 0.0} for v in x]


def _bloch_equator(x, bandwidth: float = 1.0) -> list[dict]:
    """H then RZ(phi): H puts every qubit on the equator, so the data shows up
    as the azimuth, not the tilt. Drawing these as RY-style tilts would be the
    wrong picture of the same circuit."""
    return [{"theta": float(np.pi / 2), "phi": float(v) * float(bandwidth)}
            for v in x]


ENCODINGS: dict[str, EncodingSpec] = {
    "angle_y": EncodingSpec(
        name="angle_y",
        label="Angle (RY)",
        max_features=12,
        qubit_formula="N",
        qubits_for=_q_identity,
        make_scaler=_minmax_0_pi,
        template=angle_y_template,
        two_qubit_gates_for=_tq_zero,
        bloch_for=_bloch_ry,
        description=(
            "One rotation per feature, no entanglement. Cheapest and the "
            "honest classical-simulable baseline: the kernel factorises into a "
            "product of per-qubit overlaps. Use it as the control."
        ),
    ),
    "angle_x2": EncodingSpec(
        name="angle_x2",
        label="Angle x2 (ring-entangled)",
        max_features=12,
        qubit_formula="N",
        qubits_for=_q_identity,
        make_scaler=_minmax_0_pi,
        template=angle_x2_template,
        two_qubit_gates_for=_tq_ring,
        bloch_for=_bloch_ry,
        description=(
            "Encode, entangle with a CNOT ring, then re-encode. The ring sits "
            "between two data layers so it genuinely changes the kernel, unlike "
            "a trailing ring, which cancels."
        ),
    ),
    "zz": EncodingSpec(
        name="zz",
        label="ZZ Feature Map (full entanglement)",
        max_features=10,
        qubit_formula="N",
        qubits_for=_q_identity,
        make_scaler=_minmax_0_2pi,
        template=zz_template,
        two_qubit_gates_for=_tq_zz_full,
        bloch_for=_bloch_equator,
        description=(
            "Second-order Pauli-Z map with data-dependent two-qubit rotations. "
            "The pairwise angle depends on two features at once, which no "
            "product kernel can represent. Capped at 10 features because the "
            "gate count grows as N^2."
        ),
    ),
    "amplitude": EncodingSpec(
        name="amplitude",
        label="Amplitude (log2 compression)",
        max_features=64,
        qubit_formula="ceil(log2 N)",
        qubits_for=_q_log2,
        make_scaler=L2Normalizer,
        template=amplitude_template,
        two_qubit_gates_for=None,
        description=(
            "Packs N features into ceil(log2 N) qubits as state amplitudes. "
            "16 features fit in 4 qubits. Cost moves from gate count into state "
            "preparation, which the simulator does for free and real hardware "
            "does not."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get_encoding(name: str) -> EncodingSpec:
    """Fail loudly and usefully. A typo in a RunConfig should not surface as a
    KeyError 400 lines into a training job."""
    try:
        return ENCODINGS[name]
    except KeyError:
        raise KeyError(
            f"unknown encoding {name!r}; available: {sorted(ENCODINGS)}"
        ) from None


# ---------------------------------------------------------------------------
# Bloch readings and circuit shape -- what the interface draws
# ---------------------------------------------------------------------------
# These describe the state after the encoding's FIRST data layer. That is the
# moment worth showing: it is where the patient's numbers physically enter the
# circuit. Everything after it is entanglement, which no single-qubit arrow can
# honestly depict.

def circuit_shape(encoding_name: str, n_features: int,
                  max_ops: int = 72) -> dict:
    """The real decomposed circuit, as data the frontend can draw.

    Returns qubits, depth, two-qubit gate count, and an ordered op list of
    {name, wires}. Read off PennyLane rather than described by hand, so a
    Phase 2 encoding renders correctly the day it is added and a template edit
    can never leave the drawing stale.

    Long circuits are truncated: zz at 10 features is 200+ ops and a diagram
    that wide is unreadable anyway. `ops_truncated` tells the caller to say so.
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

    qnode = qml.QNode(probe, qml.device("default.qubit", wires=q))
    try:
        batch, _ = qml.workflow.construct_batch(qnode, level="device")(x)
        ops = [
            {"name": op.name, "wires": [int(w) for w in op.wires]}
            for op in batch[0].operations
        ]
        res = qml.specs(qnode, level="device")(x).resources
        depth = int(res.depth)
        two_q = int(sum(c for size, c in res.gate_sizes.items() if size >= 2))
    except Exception:
        # A drawing must never be the thing that fails a request.
        return {"qubits": q, "depth": None, "two_qubit_gates": None,
                "ops": None, "ops_truncated": False}

    return {
        "qubits": q,
        "depth": depth,
        "two_qubit_gates": two_q,
        "ops": ops[:max_ops],
        "ops_truncated": len(ops) > max_ops,
    }


def bloch_demo(encoding_name: str, n_features: int) -> list[dict] | None:
    """A fan of rotations: one feature swept from its minimum to its maximum.

    The Configure screen has no patient yet, so it cannot honestly show one.
    What it CAN show is the encoding's geometry — what a feature value
    physically becomes. Each sphere here is the same single feature at a
    different point in its scaled range, not eight different measurements.

    The range is read off the encoding's own scaler rather than hardcoded, so
    angle_y sweeps 0..pi and zz sweeps 0..2pi without this function knowing
    which is which.
    """
    spec = get_encoding(encoding_name)
    if spec.bloch_for is None:
        return None
    n = max(int(spec.qubits_for(n_features)), 1)
    try:
        probe = np.linspace(0.0, 1.0, 16).reshape(-1, 1).repeat(n, axis=1)
        scaled = np.asarray(spec.make_scaler().fit(probe).transform(probe))
        lo, hi = float(scaled[:, 0].min()), float(scaled[:, 0].max())
        return spec.bloch_for(np.linspace(lo, hi, n), 1.0)
    except Exception:
        return None


def bloch_angles(encoding_name: str, x, bandwidth: float = 1.0
                 ) -> list[dict] | None:
    """One patient's scaled features as per-qubit angles, or None."""
    spec = get_encoding(encoding_name)
    if spec.bloch_for is None:
        return None
    try:
        return spec.bloch_for(x, bandwidth)
    except Exception:
        return None


def catalog() -> list[dict]:
    """GET /encodings -- handed to W3, rendered by W2. W2 never hardcodes a
    qubit formula; `readout()` on EncodingSpec generates the live string."""
    return [e.catalog_entry() for e in ENCODINGS.values()]


# ---------------------------------------------------------------------------
# Smoke test:  python -m backend.core.encodings
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def _states(spec: EncodingSpec, X: np.ndarray) -> np.ndarray:
        q = spec.qubits_for(X.shape[1])
        dev = qml.device("default.qubit", wires=q)

        @qml.qnode(dev)
        def f(x):
            spec.template(x, wires=range(q))
            return qml.state()

        return np.asarray(f(X))

    for name, spec in ENCODINGS.items():
        n_feat = 8
        X = rng.random((12, n_feat))
        Xs = spec.make_scaler().fit_transform(X)
        S = _states(spec, Xs)

        q = spec.qubits_for(n_feat)
        assert S.shape == (12, 2 ** q), f"{name}: {S.shape}"
        assert np.allclose(np.linalg.norm(S, axis=1), 1.0, atol=1e-5), name

        K = np.abs(S @ S.conj().T) ** 2
        # float32 features, so 1e-5 -- not 1e-9. K.max() lands at 1.0000002.
        assert np.allclose(K, K.T, atol=1e-5), f"{name}: kernel not symmetric"
        assert np.allclose(K.diagonal(), 1.0, atol=1e-5), f"{name}: diag != 1"
        assert K.min() >= -1e-5 and K.max() <= 1 + 1e-5, f"{name}: K out of [0,1]"

        r = qml.specs(
            qml.QNode(
                lambda x, s=spec, qq=q: (s.template(x, wires=range(qq)), qml.state())[1],
                qml.device("default.qubit", wires=q),
            ),
            level="device",
        )(Xs[0])
        print(
            f"{name:10s} q={q:2d}  dim={2**q:5d}  depth={r.resources.depth:3d}  "
            f"gates={r.resources.num_gates:3d}  "
            f"2q={sum(c for s_, c in r.resources.gate_sizes.items() if s_ >= 2):3d}  "
            f"scaler={type(spec.make_scaler()).__name__}"
        )

    # The claim in the module docstring, asserted so it stays true.
    X = rng.random((10, 6))
    Xs = _minmax_0_pi().fit_transform(X)
    q = 6
    dev = qml.device("default.qubit", wires=q)

    @qml.qnode(dev)
    def plain(x):
        qml.AngleEmbedding(x, wires=range(q), rotation="Y")
        return qml.state()

    @qml.qnode(dev)
    def trailing_ring(x):
        qml.AngleEmbedding(x, wires=range(q), rotation="Y")
        for i in range(q):
            qml.CNOT(wires=[i, (i + 1) % q])
        return qml.state()

    Ka = np.abs(np.asarray(plain(Xs)) @ np.asarray(plain(Xs)).conj().T) ** 2
    Kb = (
        np.abs(
            np.asarray(trailing_ring(Xs)) @ np.asarray(trailing_ring(Xs)).conj().T
        )
        ** 2
    )
    assert np.allclose(Ka, Kb, atol=1e-10), "trailing CNOT ring should be a no-op"

    # ... and that putting the ring BETWEEN two data layers is not a no-op.
    Sa = _states(ENCODINGS["angle_y"], Xs)
    Sb = _states(ENCODINGS["angle_x2"], Xs)
    Kx2 = np.abs(Sb @ Sb.conj().T) ** 2
    assert not np.allclose(Ka, Kx2, atol=1e-3), "angle_x2 should differ from angle_y"

    # Amplitude compression, the number that goes on the slide.
    amp = ENCODINGS["amplitude"]
    assert amp.qubits_for(16) == 4 and amp.qubits_for(8) == 3
    assert ENCODINGS["angle_y"].qubits_for(16) == 16

    # Zero rows must not produce NaN.
    Z = np.zeros((3, 8))
    assert np.isfinite(L2Normalizer().fit_transform(Z)).all()

    print()
    print("trailing CNOT ring is a no-op for the kernel  -- asserted")
    print("ring between two data layers is NOT a no-op   -- asserted")
    print("16 features -> amplitude 4 qubits / angle 16 qubits")
    print("encodings OK")
