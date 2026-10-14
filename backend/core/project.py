"""
backend/core/project.py
=======================

Owner: W4.  float32[n, dim] -> the INVARIANT.

    embeddings  float32[n, dim]      dim = 512 / 2048 / 256 ...
            |
        stratified split           <- FIRST. Always first.
            |
        PCA(n_features)            <- fitted on TRAIN ONLY
            |
        encoding scaler            <- fitted on TRAIN ONLY
            |
    X_train, X_test : float32[*, N]     y_train, y_test : int[*]


ORDER IS THE WHOLE FILE
-----------------------
Split, then fit. Not fit, then split.

Fitting PCA on all n samples and splitting afterwards leaks test-set structure
into the components, and the leak is invisible: no error, no warning, just
accuracy that is 2-5 points too high and does not reproduce on new data. The
existing classical notebook gets this right (cell 16 splits before
TruncatedSVD). The point of putting it in one function is that it stays right
when someone is editing at 2am on Day 4.

The same applies to the scaler. A MinMaxScaler fitted on all the data has seen
the test set's min and max.


WHY PCA AND NOT TruncatedSVD
----------------------------
The notebook used TruncatedSVD because its feature matrix was 462,400 columns
wide and PCA's mean-centring would have densified it. At 512 or 2048 columns
that constraint is gone, and PCA is the better choice here: mean-centred
components map onto a symmetric rotation range, which is what an angle encoding
wants. Uncentred SVD components are all one-signed, so MinMax squashes them into
a narrow band and the encoding wastes half its range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from .contracts import EncodingSpec, FeatureMatrix, LabelVector, RunConfig
from .encodings import get_encoding

__all__ = ["Projection", "project", "apply_projection"]


@dataclass
class Projection:
    """Everything downstream of the split needs, in one object.

    Internal to W4 -- it never crosses a seam, so it lives here rather than in
    contracts.py. `pca` and `scaler` go into the CheckpointBundle so the
    Diagnose screen can replay the exact transform on a single image.
    """
    X_train: FeatureMatrix
    X_test: FeatureMatrix
    y_train: LabelVector
    y_test: LabelVector
    pca: PCA
    scaler: Any
    encoding: EncodingSpec
    explained_variance: float          # fraction retained by n_features
    n_train: int
    n_test: int


def project(
    cfg: RunConfig,
    E: NDArray[np.float32],
    y: LabelVector,
    encoding: EncodingSpec | None = None,
) -> Projection:
    E = np.asarray(E, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    enc = encoding or get_encoding(cfg.encoding)
    n, dim = E.shape

    if len(y) != n:
        raise ValueError(f"{n} embeddings but {len(y)} labels")
    if cfg.n_features > enc.max_features:
        raise ValueError(
            f"encoding '{enc.name}' supports at most {enc.max_features} "
            f"features, got {cfg.n_features}"
        )

    # Stratify unless some class is too rare to appear on both sides, in which
    # case stratifying raises and an unstratified split at least completes.
    counts = np.bincount(y)
    present = int((counts > 0).sum())
    stratify = y if counts[counts > 0].min() >= 2 else None

    # sklearn raises "test_size = 2 should be greater or equal to the number of
    # classes = 4" here, which tells a user nothing actionable. Catch it first.
    n_test = int(np.ceil(n * cfg.test_size))
    if stratify is not None and n_test < present:
        raise ValueError(
            f"a {cfg.test_size:.0%} test split of {n} samples leaves {n_test} "
            f"test images, which cannot cover {present} classes. Upload more "
            f"images or raise the test fraction."
        )

    idx_train, idx_test = train_test_split(
        np.arange(n),
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=stratify,
    )

    # PCA cannot produce more components than min(n_samples, n_features), and
    # sklearn's error for this is opaque. Catch it here where the fix is
    # obvious: lower the feature slider or upload more images.
    max_components = min(len(idx_train), dim)
    if cfg.n_features > max_components:
        raise ValueError(
            f"n_features={cfg.n_features} exceeds what PCA can produce from "
            f"{len(idx_train)} training samples x {dim} dims "
            f"(max {max_components}). Lower the feature count."
        )

    pca = PCA(n_components=cfg.n_features, random_state=cfg.seed)
    P_train = pca.fit_transform(E[idx_train])
    P_test = pca.transform(E[idx_test])

    scaler = enc.make_scaler().fit(P_train)
    X_train = np.asarray(scaler.transform(P_train), dtype=np.float32)
    X_test = np.asarray(scaler.transform(P_test), dtype=np.float32)

    return Projection(
        X_train=X_train,
        X_test=X_test,
        y_train=y[idx_train],
        y_test=y[idx_test],
        pca=pca,
        scaler=scaler,
        encoding=enc,
        explained_variance=float(pca.explained_variance_ratio_.sum()),
        n_train=len(idx_train),
        n_test=len(idx_test),
    )


def apply_projection(
    pca: PCA, scaler: Any, E: NDArray[np.float32]
) -> FeatureMatrix:
    """Replay a fitted projection on new embeddings.

    Used by predict_single. Kept here, next to `project`, so the training path
    and the inference path cannot drift apart: if someone adds a step above,
    the omission is visible in the same screenful of code.
    """
    E = np.asarray(E, dtype=np.float32)
    if E.ndim == 1:
        E = E.reshape(1, -1)
    return np.asarray(scaler.transform(pca.transform(E)), dtype=np.float32)


# ---------------------------------------------------------------------------
# Smoke test:  python -m backend.core.project
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, dim = 200, 512
    y = rng.integers(0, 4, n).astype(np.int64)
    E = (rng.normal(size=(n, dim)) + y[:, None] * 0.4).astype(np.float32)

    for enc_name in ("angle_y", "zz", "amplitude"):
        cfg = RunConfig(dataset_id="d", models=["svc"], n_features=8,
                        encoding=enc_name, test_size=0.2, seed=42)
        p = project(cfg, E, y)
        enc = get_encoding(enc_name)

        assert p.X_train.shape == (160, 8) and p.X_test.shape == (40, 8)
        assert p.X_train.dtype == np.float32
        assert set(np.unique(p.y_train)) <= set(range(4))

        lo, hi = p.X_train.min(), p.X_train.max()
        if enc_name == "angle_y":
            assert abs(lo) < 1e-5 and abs(hi - np.pi) < 1e-5, (lo, hi)
        if enc_name == "amplitude":
            assert np.allclose(np.linalg.norm(p.X_train, axis=1), 1.0, atol=1e-5)

        # Stratification held.
        tr = np.bincount(p.y_train, minlength=4) / len(p.y_train)
        al = np.bincount(y, minlength=4) / len(y)
        assert np.abs(tr - al).max() < 0.05, "split is not stratified"

        print(f"{enc_name:10s} range=[{lo:.3f}, {hi:.3f}]  "
              f"var_retained={p.explained_variance:.3f}")

    # No leakage, checked deterministically. sklearn records how many samples
    # each transformer was fitted on, so this cannot pass by luck the way
    # "does X_test escape the train range?" can -- with 160 train and 40 test
    # points drawn from one distribution, the train range almost always bounds
    # the test set even when the fit was completely clean.
    cfg = RunConfig(dataset_id="d", models=["svc"], n_features=8, encoding="angle_y")
    p = project(cfg, E, y)
    assert p.pca.n_samples_ == p.n_train == 160, p.pca.n_samples_
    assert p.scaler.n_samples_seen_ == p.n_train, p.scaler.n_samples_seen_
    assert p.n_train + p.n_test == n

    # Asking for more components than the training set can supply must be a
    # clear error, not a numpy traceback.
    try:
        project(RunConfig(dataset_id="d", models=["svc"], n_features=45,
                          encoding="amplitude"),
                E[:50], y[:50])
        raise AssertionError("expected a component-count failure")
    except ValueError as e:
        assert "exceeds what PCA can produce" in str(e), e

    # A split too small to cover every class must say so in words.
    try:
        project(RunConfig(dataset_id="d", models=["svc"], n_features=4,
                          encoding="angle_y"),
                E[:10], y[:10])
        raise AssertionError("expected a too-small-split failure")
    except ValueError as e:
        assert "cannot cover" in str(e), e

    print(f"PCA fitted on {p.pca.n_samples_} of {n} samples -- no leakage")
    print("project OK")
