"""
backend/main.py
===============

Routes. Nothing else.

House rules, all greppable:
  * No ML logic. No `if model == "qsvc"`. No numpy. No torch.
  * Every route body under 15 lines. If one grows, the logic belongs in a
    module, not in a handler.
  * Every failure returns `{"detail": "..."}`. Never a traceback, never a
    bare 500.

CPU-BOUND WORK AND THE EVENT LOOP
---------------------------------
Ingest, preview and predict all do real pixel work. A sync `def` route is run
by Starlette in a threadpool, so it cannot block the event loop; an `async def`
route that calls sync code blocks everything, including the 800ms job polling.
Routes here are therefore sync `def` unless they need `await` for an upload, in
which case the heavy call goes through `run_in_threadpool`.

W4 ENDPOINTS
------------
Five routes need the model layer. Until those files exist they return 503 with
a readable message rather than crashing at import, so `/docs` is complete from
day one and W2 can see every shape. `GET /health` reports what is wired.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Callable

from fastapi import (
    BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from backend import jobs, store
from backend.core.contracts import DatasetMeta, ProgressFn, RunConfig
from backend.core.ingest import IngestError, ingest_zip, load_stored
from backend.core.preprocess import (
    load_image, ops_catalog, presets_catalog, preview_stages, to_b64,
)
from backend.schemas import (
    BackboneOut, DatasetOut, EncodingOut, EstimateOut, JobOut, JobSubmitOut,
    LeaderboardRowOut, ModelCatalogOut, OpCatalogOut, PredictionOut,
    PresetOut, PreviewIn, PreviewOut, RunConfigIn, RunOut, RunSummaryOut,
    ValidationOut,
)
from backend.validate import validate

log = logging.getLogger(__name__)

app = FastAPI(
    title="Hybrid QML Platform for Early Disease Detection",
    description="SIH 2026 · PS 26139 · classical vs quantum on identical features",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Errors — one shape for every failure
# ---------------------------------------------------------------------------

@app.exception_handler(IngestError)
def _bad_upload(_: Request, exc: IngestError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
def _unhandled(_: Request, exc: Exception) -> JSONResponse:
    """The stack trace goes to the log; the client gets one line. A 500 with a
    Python traceback in the body is the fastest way to look unfinished."""
    log.exception("unhandled error")
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"[:300]},
    )


# ---------------------------------------------------------------------------
# W4 accessors — the model layer, if it exists yet
# ---------------------------------------------------------------------------

W4_OPTIONAL = {
    # Nice-to-have symbols. Absent ones degrade a feature rather than a route:
    # no circuit diagram is a worse screen, not a broken one.
    "encodings_shape": ("encodings", "circuit_shape"),
    "encodings_bloch": ("encodings", "bloch_demo"),
}


def _w4_optional(key: str) -> Any:
    module, symbol = W4_OPTIONAL[key]
    try:
        return getattr(importlib.import_module(f"backend.core.{module}"), symbol)
    except Exception:
        return None


W4 = {
    # These three are the registries' own catalog() accessors, not the raw
    # dicts. W4 sorts models classical-first so the picker's two columns come
    # out stable without W2 sorting them; iterating the dict would lose that.
    "encodings": ("encodings", "catalog"),
    "models": ("registry", "catalog"),
    "backbones": ("embed", "catalog"),
    "estimate": ("estimate", "estimate"),
    "benchmark": ("benchmark", "run_for_dataset"),
    "predict": ("benchmark", "predict_single"),
}


def _w4(key: str) -> Any:
    """Import a model-layer symbol on demand. 503 if it is not written yet."""
    module, symbol = W4[key]
    try:
        return getattr(importlib.import_module(f"backend.core.{module}"), symbol)
    except Exception as exc:
        log.debug("W4 symbol %s unavailable: %s", key, exc)
        raise HTTPException(
            status_code=503,
            detail=f"The model layer is not wired yet "
                   f"(backend/core/{module}.py: {symbol}).",
        )


def _w4_status() -> dict[str, bool]:
    out = {}
    for key, (module, symbol) in W4.items():
        try:
            getattr(importlib.import_module(f"backend.core.{module}"), symbol)
            out[key] = True
        except Exception:
            out[key] = False
    return out


# ---------------------------------------------------------------------------
# Shared dependencies
# ---------------------------------------------------------------------------

def get_dataset_or_404(dataset_id: str) -> DatasetMeta:
    ds = store.get_dataset(dataset_id)
    if ds is None:
        raise HTTPException(404, f"Dataset {dataset_id!r} not found.")
    return ds


def _require_valid(cfg: RunConfig, ds: DatasetMeta) -> None:
    result = validate(cfg, ds)
    if not result.ok:
        raise HTTPException(422, " ".join(result.errors))


def _make_work(cfg: RunConfig, ds: DatasetMeta) -> Callable[[ProgressFn], str]:
    """The job body. Closes over the config so `jobs` stays ignorant of both
    the store and the model layer.

    `run_for_dataset` is the wrapper: it loads the images, replays cfg.ops,
    trains, and returns an assembled RunRecord. Saving is ours — core never
    imports the store."""
    run_for_dataset = _w4("benchmark")

    def work(progress: ProgressFn) -> str:
        record = run_for_dataset(cfg, ds, progress)
        store.save_run(record)
        return record.run_id

    return work


# ===========================================================================
# Health
# ===========================================================================

@app.get("/health")
def health() -> dict[str, Any]:
    """Not in the frozen 16. Additive, and it saves an argument about whether
    the backend is up or the frontend proxy is wrong."""
    return {"ok": True, "version": app.version, "wired": _w4_status()}


# ===========================================================================
# Datasets
# ===========================================================================

@app.post("/datasets", response_model=DatasetOut)
async def create_dataset(
    file: UploadFile = File(...),
    name: str | None = Form(None),
    modality: str = Form("ecg"),
    adapter: str = Form("image_folder"),
) -> dict[str, Any]:
    raw = await file.read()
    label = name or Path(file.filename or "dataset").stem
    meta, previews = await run_in_threadpool(
        ingest_zip, raw, label, modality, adapter
    )
    store.save_dataset(meta)
    return {**meta.to_dict(), "preview_b64": previews}


@app.get("/datasets", response_model=list[DatasetOut])
def list_datasets() -> list[dict[str, Any]]:
    return [d.to_dict() for d in store.list_datasets()]


# ===========================================================================
# Catalogs — every one of these is a registry dump. That is the whole point.
# ===========================================================================

@app.get("/ops/catalog", response_model=list[OpCatalogOut])
def get_ops_catalog() -> list[dict[str, Any]]:
    return ops_catalog()


@app.get("/presets", response_model=list[PresetOut])
def get_presets() -> list[dict[str, Any]]:
    return presets_catalog()


@app.get("/backbones", response_model=list[BackboneOut])
def get_backbones() -> list[dict[str, Any]]:
    return _w4("backbones")()


@app.get("/encodings", response_model=list[EncodingOut])
def get_encodings(n_features: int | None = None) -> list[dict[str, Any]]:
    """Without n_features this is the plain catalog. With it, each entry also
    carries the real decomposed circuit at that width — qubits, depth, gate
    count and the op list — so the Configure screen can draw the circuit the
    user is about to run rather than a generic picture of one."""
    entries = _w4("encodings")()
    if n_features is None:
        return entries
    shape = _w4_optional("encodings_shape")
    if shape is None:
        return entries
    demo = _w4_optional("encodings_bloch")
    for e in entries:
        if n_features <= e["max_features"]:
            e.update(shape(e["name"], n_features))
            if demo is not None:
                e["bloch_demo"] = demo(e["name"], n_features)
    return entries


@app.get("/models/catalog", response_model=list[ModelCatalogOut])
def get_models_catalog() -> list[dict[str, Any]]:
    return _w4("models")()


# ===========================================================================
# Preprocessing preview
# ===========================================================================

@app.post("/preprocess/preview", response_model=PreviewOut)
def preprocess_preview(body: PreviewIn) -> dict[str, Any]:
    ds = get_dataset_or_404(body.dataset_id)
    paths, _ = load_stored(ds)
    src = paths[body.image_index % len(paths)]
    cfg = RunConfigIn(dataset_id=body.dataset_id, ops=body.ops).to_contract()
    stages = preview_stages(load_image(src), cfg.ops)
    return {
        "original_b64": to_b64(stages[0][1]),
        "stages": [{"op": op, "image_b64": to_b64(img)} for op, img in stages[1:]],
    }


# ===========================================================================
# Runs
# ===========================================================================

@app.post("/runs/validate", response_model=ValidationOut)
def validate_run(body: RunConfigIn) -> dict[str, Any]:
    cfg = body.to_contract()
    return validate(cfg, store.get_dataset(cfg.dataset_id)).to_dict()


@app.post("/runs/estimate", response_model=EstimateOut)
def estimate_run(body: RunConfigIn) -> dict[str, Any]:
    cfg = body.to_contract()
    ds = get_dataset_or_404(cfg.dataset_id)
    # estimate() takes a sample COUNT, not a DatasetMeta — core does not import
    # the store's types, and the cost model only ever needed the one number.
    return _w4("estimate")(cfg, ds.n_samples).to_dict()


@app.post("/runs", response_model=JobSubmitOut)
def create_run(body: RunConfigIn, background: BackgroundTasks) -> dict[str, str]:
    cfg = body.to_contract()
    ds = get_dataset_or_404(cfg.dataset_id)
    _require_valid(cfg, ds)          # again, server-side. curl is a client.
    job_id = jobs.submit_job(_make_work(cfg, ds), background.add_task)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str) -> dict[str, Any]:
    state = jobs.get_job(job_id)
    if state is None:
        raise HTTPException(404, f"Job {job_id!r} not found.")
    return state.to_dict()


@app.get("/runs", response_model=list[RunSummaryOut])
def list_runs() -> list[dict[str, Any]]:
    return [r.summary() for r in store.list_runs()]


@app.get("/runs/{run_id}", response_model=RunOut)
def get_run(run_id: str) -> dict[str, Any]:
    record = store.get_run(run_id)
    if record is None:
        raise HTTPException(404, f"Run {run_id!r} not found.")
    return record.to_dict()


@app.get("/leaderboard", response_model=list[LeaderboardRowOut])
def leaderboard() -> list[dict[str, Any]]:
    return store.leaderboard()


# ===========================================================================
# Diagnose
# ===========================================================================

@app.post("/predict/{run_id}/{model}", response_model=PredictionOut)
async def predict(run_id: str, model: str,
                  file: UploadFile = File(...)) -> dict[str, Any]:
    record = store.get_run(run_id)
    if record is None:
        raise HTTPException(404, f"Run {run_id!r} not found.")
    result = record.result_for(model)
    if result is None or not result.checkpoint:
        raise HTTPException(404, f"Run {run_id!r} has no usable {model!r} model.")
    predict_single = _w4("predict")
    raw = await file.read()
    # Returns a plain dict; PredictionOut validates it on the way out.
    return await run_in_threadpool(predict_single, result.checkpoint, raw)


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _banner() -> None:
    wired = _w4_status()
    missing = sorted(k for k, ok in wired.items() if not ok)
    if missing:
        log.warning("model layer not wired: %s — those endpoints return 503",
                    ", ".join(missing))
    else:
        log.info("all six model-layer symbols wired")
