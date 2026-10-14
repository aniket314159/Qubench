"""
core/paths.py  (backend/core/paths.py)
======================================

Every filesystem location in the project, in one place. Frozen with contracts.py.

Why this exists: embed.py writes to artifacts/ and checkpoint.py writes to
checkpoints/, both W4 files, and W4 may never import from backend/. Without a
shared constants module you get three hardcoded Path("storage") strings and a
bug where the API and scripts/pipeline.py disagree about where the cache lives.

Resolution is anchored to this file, not to the working directory, so
`python scripts/pipeline.py` and `uvicorn backend.main:app` see the same disk.
"""

from __future__ import annotations

from pathlib import Path

# backend/core/paths.py -> backend/core -> backend -> repo root.
# If core/ is ever moved to the top level, this becomes parents[1]. The assert
# below fails loudly rather than silently creating storage/ in the wrong place.
ROOT = Path(__file__).resolve().parents[2]

if not (ROOT / "backend").is_dir():
    raise RuntimeError(
        f"paths.ROOT resolved to {ROOT}, which has no backend/ directory. "
        "core/ has moved; fix the parents[] index in core/paths.py."
    )

STORAGE = ROOT / "storage"

DATASETS_DIR = STORAGE / "datasets"        # {ds_id}/{class_name}/*.png
ARTIFACTS_DIR = STORAGE / "artifacts"      # {ds_id}__{backbone}__{ops_hash}.npy
CHECKPOINTS_DIR = STORAGE / "checkpoints"  # {run_id}__{model}.pkl

DATASETS_JSON = STORAGE / "datasets.json"
RUNS_JSON = STORAGE / "runs.json"

PREVIEW_DIRNAME = "_preview"               # reserved; ingest must skip this name

for _d in (DATASETS_DIR, ARTIFACTS_DIR, CHECKPOINTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Accessor functions
# ---------------------------------------------------------------------------
# Prefer these over the constants above in new code. They read
# QUCARDIO_STORAGE at CALL time, which is what lets a module's smoke test
# redirect itself to a temp directory after import:
#
#     os.environ["QUCARDIO_STORAGE"] = tmp        # inside __main__
#     embed(cfg, imgs)                            # writes to tmp, not storage/
#
# Without that, running `python -m backend.core.embed` writes cache files into
# the real storage tree and `clear_cache` deletes real entries.
#
# NOTE: store.py binds DATASETS_JSON / RUNS_JSON at import, so the override
# does NOT move the metadata JSONs. That is deliberate — store.py's own smoke
# test redirects itself — but it means the override is a W4 test hook, not a
# general "run the app against another storage root" switch.

def storage_root() -> Path:
    import os
    env = os.environ.get("QUCARDIO_STORAGE")
    return Path(env) if env else STORAGE


def _sub(name: str) -> Path:
    p = storage_root() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def datasets_dir() -> Path:
    return _sub("datasets")


def artifacts_dir() -> Path:
    return _sub("artifacts")


def checkpoints_dir() -> Path:
    return _sub("checkpoints")


def datasets_json() -> Path:
    return storage_root() / "datasets.json"


def runs_json() -> Path:
    return storage_root() / "runs.json"


def dataset_dir(dataset_id: str) -> Path:
    return DATASETS_DIR / dataset_id


def preview_dir(dataset_id: str) -> Path:
    return DATASETS_DIR / dataset_id / PREVIEW_DIRNAME


def artifact_path(embed_key: str) -> Path:
    """`embed_key` comes from RunConfig.embed_key(). Never build it by hand."""
    return ARTIFACTS_DIR / embed_key


def checkpoint_path(run_id: str, model: str) -> Path:
    return CHECKPOINTS_DIR / f"{run_id}__{model}.pkl"


if __name__ == "__main__":
    for name in ("ROOT", "STORAGE", "DATASETS_DIR", "ARTIFACTS_DIR",
                 "CHECKPOINTS_DIR", "DATASETS_JSON", "RUNS_JSON"):
        print(f"{name:<16} {globals()[name]}")
