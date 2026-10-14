"""
backend/core/checkpoint.py
==========================

Owner: W4. Pickles a CheckpointBundle to storage/checkpoints/{run_id}__{model}.pkl.

The bundle carries the RECIPE alongside the weights: ops, backbone, encoding,
n_features, the fitted PCA and the fitted scaler. The Diagnose screen must
replay exactly what training did:

    ops -> embed(backbone) -> pca -> scaler -> estimator.predict_proba

If Diagnose applies even slightly different preprocessing, predictions are
quietly wrong and nothing raises: the shapes all still line up, the
probabilities still sum to 1, and the model confidently returns the wrong class.
That is the single worst failure mode in this codebase because it survives
testing and only shows up in front of judges.

So the recipe is never reconstructed from the live RunConfig at inference time.
The user may have moved the feature slider since the run; the config on screen
is not the config that trained this checkpoint.


WHY PICKLE AND NOT JSON
-----------------------
The bundle holds fitted sklearn objects and numpy weight arrays. Pickle is the
right tool and the security caveat does not apply: these files are written by
this application and read by this application, never uploaded.

The cost is that pickle is brittle across library versions, so `save` records
the versions and `load` warns rather than crashing when they differ. A warning
lets a Day-4 demo proceed on a slightly different machine; a hard failure does
not.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

from .contracts import CheckpointBundle
from .paths import checkpoints_dir

__all__ = ["save", "load", "checkpoint_name", "FORMAT_VERSION"]

FORMAT_VERSION = 1


def checkpoint_name(run_id: str, model_name: str) -> str:
    """The filename stored in ModelResult.checkpoint. A name, not a path:
    storage may move between machines, and an absolute path baked into
    runs.json would not survive it."""
    return f"{run_id}__{model_name}.pkl"


def save(bundle: CheckpointBundle) -> str:
    path = checkpoints_dir() / checkpoint_name(bundle.run_id, bundle.model_name)
    payload = {
        "format_version": FORMAT_VERSION,
        "bundle": bundle,
        "versions": _versions(),
    }
    # Write to a temp file then rename. A half-written pickle from an
    # interrupted job would otherwise sit on disk looking valid until Diagnose
    # tries to load it.
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return path.name


def load(name_or_path: str | Path) -> CheckpointBundle:
    path = Path(name_or_path)
    if not path.is_absolute() and not path.exists():
        path = checkpoints_dir() / path.name
    if not path.exists():
        raise FileNotFoundError(
            f"no checkpoint at {path}. Checkpoints are written per run; if the "
            f"run predates the current storage directory, re-run the benchmark."
        )

    with open(path, "rb") as fh:
        payload = pickle.load(fh)

    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{path.name} is format v{payload.get('format_version')}, "
            f"this build reads v{FORMAT_VERSION}. Re-run the benchmark."
        )

    for lib, was in payload.get("versions", {}).items():
        now = _versions().get(lib)
        if now and was and now != was:
            warnings.warn(
                f"{path.name} was written with {lib} {was}, loading under {now}. "
                f"Predictions should still be correct; re-run if they look odd.",
                RuntimeWarning,
                stacklevel=2,
            )

    return payload["bundle"]


def _versions() -> dict[str, str]:
    out: dict[str, Any] = {}
    for lib in ("numpy", "sklearn", "pennylane"):
        try:
            out[lib] = __import__(lib).__version__
        except Exception:
            pass
    return out
