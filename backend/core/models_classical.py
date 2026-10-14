"""
backend/core/models_classical.py  --  SEAM 4, classical half
============================================================

Owner: W4.

Both models take `encoding` in the constructor and ignore it. That is the price
of benchmark.py having exactly one call site with no `if kind == ...` branch,
and it is worth paying.


THE TUNED BASELINE IS THE POINT
-------------------------------
The notebook compares a default `SVC()` at ~86% against QSVC at ~97% and cell 23
(the grid search) is commented out. An 11-point gap against an untuned baseline
is not a result, it is an artefact, and "did you tune the classical model?" is
the first question a judge asks. So SVC tunes by default (`tune=True`) and the
chosen hyperparameters go into telemetry, where the frontend can show them.

If the honest gap turns out to be 4 points, a 4-point gap you can defend beats
an 11-point gap that collapses under one question.


ON SHARING THE ENCODING'S SCALER
--------------------------------
project.py fits ONE scaler -- the one the chosen encoding declares -- and every
model sees the same X. So on `angle_y` the classical models get MinMax[0, pi]
data rather than MinMax[0, 1].

That is deliberate and it is the fair comparison: identical features to both
arms, so any difference is the model, not the preprocessing. It is also
harmless for these two -- RandomForest is invariant to any monotone rescaling,
and SVC's `gamma="scale"` divides by X.var(), which absorbs a linear rescale.

It stops being harmless if you ship `amplitude`, whose L2Normalizer is NOT a
monotone per-feature map: it throws away vector magnitude. That is still a fair
comparison (same X to both arms) but it is a different feature space, so runs
across different encodings are not comparable on the classical row either.
Flag that to W1 before it appears on a slide.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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

from .contracts import BaseModel, FeatureMatrix, LabelVector

__all__ = ["SVCModel", "RandomForestModel", "safe_cv_folds"]


def safe_cv_folds(y: LabelVector, requested: int = 5) -> int:
    """StratifiedKFold needs at least `n_splits` members of the rarest class.

    On a small or badly imbalanced upload this is the difference between a grid
    search and a crash mid-job. Returns 0 when even 2 folds are impossible, and
    the caller skips tuning rather than failing the run.
    """
    if len(y) == 0:
        return 0
    smallest = int(np.bincount(y).min())
    return min(int(requested), smallest) if smallest >= 2 else 0


# ---------------------------------------------------------------------------
# SVC
# ---------------------------------------------------------------------------

class SVCModel(BaseModel):
    """RBF/linear/poly SVC with an optional grid search.

    `probability=True` fits Platt scaling via internal cross-validation, which
    roughly quintuples fit time. It is required: the Diagnose screen shows
    per-class confidence bars, and `decision_function` is not a probability.
    """

    name = "svc"
    kind = "classical"
    label = "SVC (RBF, tuned)"
    backend_name = "sklearn"
    param_schema = {
        "tune": {
            "type": "bool",
            "default": True,
            "label": "Grid search",
            "help": "Untuned baselines inflate the quantum gap. Leave this on.",
        },
        "cv": {"type": "int", "min": 2, "max": 10, "default": 5, "label": "CV folds"},
    }

    # 3 kernels x 4 C x 3 gamma, minus the gamma-irrelevant linear duplicates.
    GRID = [
        {"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1]},
        {"kernel": ["linear"], "C": [0.1, 1, 10, 100]},
        {"kernel": ["poly"], "C": [1, 10], "degree": [2, 3], "gamma": ["scale"]},
    ]

    def _fit(self, X: FeatureMatrix, y: LabelVector) -> None:
        tune = bool(self.params.get("tune", True))
        folds = safe_cv_folds(y, int(self.params.get("cv", 5))) if tune else 0

        if folds >= 2:
            search = GridSearchCV(
                SVC(probability=True, random_state=self.seed),
                self.GRID,
                cv=StratifiedKFold(folds, shuffle=True, random_state=self.seed),
                scoring="f1_macro",   # not accuracy: these datasets are imbalanced
                n_jobs=-1,
                refit=True,
            )
            search.fit(X, y)
            self._est: SVC = search.best_estimator_
            self._best_params: dict[str, Any] = dict(search.best_params_)
            self._cv_score: float | None = float(search.best_score_)
        else:
            self._est = SVC(
                probability=True, random_state=self.seed, kernel="rbf", gamma="scale"
            ).fit(X, y)
            self._best_params = {"kernel": "rbf", "gamma": "scale", "C": 1.0}
            self._cv_score = None

    def _predict_proba(self, X: FeatureMatrix) -> NDArray[np.float64]:
        return self._est.predict_proba(X)

    def n_trainable_params(self) -> int | None:
        """Support vectors. Not "parameters" in the gradient sense, but it is
        the honest measure of an SVC's model size and it is what belongs in the
        column next to the VQC's weight count."""
        return int(self._est.n_support_.sum()) if self._fitted else None

    @property
    def best_params(self) -> dict[str, Any]:
        return dict(getattr(self, "_best_params", {}))


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class RandomForestModel(BaseModel):
    """The second classical arm.

    Worth having precisely because it is scale-invariant: if RF and SVC land far
    apart, the gap is about the feature geometry, not about the scaler. That
    makes it a free diagnostic on every run.
    """

    name = "rf"
    kind = "classical"
    label = "Random Forest"
    backend_name = "sklearn"
    param_schema = {
        "n_estimators": {
            "type": "int", "min": 50, "max": 1000, "default": 300, "label": "Trees",
        },
        "max_depth": {
            "type": "int", "min": 0, "max": 50, "default": 0,
            "label": "Max depth", "help": "0 = unlimited",
        },
    }

    def _fit(self, X: FeatureMatrix, y: LabelVector) -> None:
        depth = int(self.params.get("max_depth", 0) or 0)
        self._est = RandomForestClassifier(
            n_estimators=int(self.params.get("n_estimators", 300)),
            max_depth=depth if depth > 0 else None,
            class_weight="balanced_subsample",
            random_state=self.seed,
            n_jobs=-1,
        ).fit(X, y)

    def _predict_proba(self, X: FeatureMatrix) -> NDArray[np.float64]:
        p = self._est.predict_proba(X)
        # RandomForestClassifier drops classes absent from y_train, so its
        # columns are `classes_`, not range(n_classes). BaseModel.predict_proba
        # enforces the full width, and a silently narrow matrix would shift
        # every argmax. Re-expand into the full class space.
        if p.shape[1] != self.n_classes:
            full = np.zeros((len(X), self.n_classes), dtype=np.float64)
            full[:, self._est.classes_.astype(int)] = p
            return full
        return p

    def n_trainable_params(self) -> int | None:
        if not self._fitted:
            return None
        return int(sum(t.tree_.node_count for t in self._est.estimators_))

    def feature_importances(self) -> list[float] | None:
        """Free interpretability on the Benchmark screen: which of the N PCA
        components the forest actually used."""
        if not self._fitted:
            return None
        return [float(v) for v in self._est.feature_importances_]
