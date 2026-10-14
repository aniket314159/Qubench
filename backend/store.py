"""
backend/store.py
================

SEAM 6 — the store. All metadata persistence, six functions, JSON files.

    list_datasets()  get_dataset(id)  save_dataset(meta)
    list_runs()      get_run(id)      save_run(record)

Phase 3 swaps this file for SQLite. Nothing above it changes, because nothing
above it knows that a JSON file exists. Callers pass and receive dataclasses.

Three properties that matter more than they look:

  * Atomic writes. Every save goes to a temp file and is os.replace()d into
    position. Ctrl-C mid-write leaves the previous good file, not a truncated
    one. On Day 5 you will interrupt the server at a bad moment.
  * Locked. FastAPI BackgroundTasks run in a threadpool, so a job saving a run
    can overlap a request listing runs. One module-level RLock serialises it.
  * Survivable reads. A record that fails to parse is skipped and logged, not
    raised. One bad row must never 500 the whole list endpoint.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.core.contracts import DatasetMeta, RunRecord
from backend.core.paths import DATASETS_JSON, RUNS_JSON

log = logging.getLogger(__name__)

# RLock, not Lock: save_run() calls list_runs() internally.
_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# JSON file primitives
# ---------------------------------------------------------------------------

def _quarantine(path: Path) -> None:
    """Move an unparseable file aside so the app can start with a clean slate
    and you can still inspect what broke."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dead = path.with_name(f"{path.stem}.corrupt-{stamp}.json")
    try:
        shutil.move(str(path), str(dead))
        log.error("%s was unreadable; moved to %s", path.name, dead.name)
    except OSError:
        log.exception("could not quarantine %s", path)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        _quarantine(path)
        return []
    if not isinstance(data, list):
        _quarantine(path)
        return []
    return [r for r in data if isinstance(r, dict)]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomic. Temp file in the same directory, then os.replace()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)          # atomic on POSIX and Windows


def _parse_all(rows: list[dict[str, Any]], cls: type, id_field: str) -> list[Any]:
    """Skip-and-log on bad records. Deliberate asymmetry with save: writes
    validate by construction, reads stay alive."""
    out = []
    for raw in rows:
        try:
            out.append(cls.from_dict(raw))
        except Exception as e:
            log.warning("skipping bad %s record %r: %s",
                        cls.__name__, raw.get(id_field, "?"), e)
    return out


def _upsert(rows: list[dict[str, Any]], id_field: str,
            new_row: dict[str, Any]) -> list[dict[str, Any]]:
    """Replace in place if the id exists, else append. Never duplicates."""
    ident = new_row[id_field]
    for i, row in enumerate(rows):
        if row.get(id_field) == ident:
            rows[i] = new_row
            return rows
    rows.append(new_row)
    return rows


def _newest_first(items: list[Any]) -> list[Any]:
    return sorted(items, key=lambda o: o.created_at, reverse=True)


# ---------------------------------------------------------------------------
# The six functions
# ---------------------------------------------------------------------------

def list_datasets() -> list[DatasetMeta]:
    """GET /datasets. Newest first."""
    with _LOCK:
        return _newest_first(
            _parse_all(_read_rows(DATASETS_JSON), DatasetMeta, "dataset_id")
        )


def get_dataset(dataset_id: str) -> DatasetMeta | None:
    """None, never an exception. The route turns None into a 404."""
    with _LOCK:
        for raw in _read_rows(DATASETS_JSON):
            if raw.get("dataset_id") == dataset_id:
                try:
                    return DatasetMeta.from_dict(raw)
                except Exception as e:
                    log.warning("dataset %s is unparseable: %s", dataset_id, e)
                    return None
    return None


def save_dataset(meta: DatasetMeta) -> DatasetMeta:
    """Upsert by dataset_id. Returns what was written."""
    with _LOCK:
        rows = _read_rows(DATASETS_JSON)
        _write_rows(DATASETS_JSON, _upsert(rows, "dataset_id", meta.to_dict()))
    return meta


def list_runs() -> list[RunRecord]:
    """Full records, newest first. Routes call .summary() on these — do not
    return summaries from here, the leaderboard needs the whole object."""
    with _LOCK:
        return _newest_first(
            _parse_all(_read_rows(RUNS_JSON), RunRecord, "run_id")
        )


def get_run(run_id: str) -> RunRecord | None:
    with _LOCK:
        for raw in _read_rows(RUNS_JSON):
            if raw.get("run_id") == run_id:
                try:
                    return RunRecord.from_dict(raw)
                except Exception as e:
                    log.warning("run %s is unparseable: %s", run_id, e)
                    return None
    return None


def save_run(record: RunRecord) -> RunRecord:
    """Upsert by run_id. Called once at the end of a job, from the worker
    thread — hence the lock."""
    with _LOCK:
        rows = _read_rows(RUNS_JSON)
        _write_rows(RUNS_JSON, _upsert(rows, "run_id", record.to_dict()))
    return record


# ---------------------------------------------------------------------------
# Derived read (not one of the six; adds no state, owns no file)
# ---------------------------------------------------------------------------

def leaderboard() -> list[dict[str, Any]]:
    """GET /leaderboard. Best classical vs best quantum per DATASET, across
    every run of that dataset — not per run. Keeps the route body trivial."""
    board: dict[str, dict[str, Any]] = {}

    for run in list_runs():                     # newest first
        row = board.setdefault(run.dataset_id, {
            "dataset_id": run.dataset_id,
            "dataset_name": run.dataset_name,
            "n_runs": 0,
            "best_classical": None,
            "best_quantum": None,
            "delta": None,
            "last_run_at": run.created_at,
        })
        row["n_runs"] += 1

        for kind in ("classical", "quantum"):
            best = run.best(kind)               # already excludes failed rows
            if best is None:
                continue
            key = f"best_{kind}"
            cur = row[key]
            if cur is None or best.metrics.accuracy > cur["accuracy"]:
                row[key] = {
                    "model": best.model,
                    "label": best.label,
                    "accuracy": best.metrics.accuracy,
                    "f1_macro": best.metrics.f1_macro,
                    "run_id": run.run_id,
                    "encoding": run.config.encoding,
                    "n_features": run.config.n_features,
                }

    for row in board.values():
        c, q = row["best_classical"], row["best_quantum"]
        if c and q:
            row["delta"] = round(q["accuracy"] - c["accuracy"], 4)

    return sorted(board.values(), key=lambda r: r["last_run_at"], reverse=True)


# ---------------------------------------------------------------------------
# Smoke test:  python -m backend.store
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    from backend.core.contracts import (
        Metrics, ModelResult, OpConfig, RunConfig, Telemetry,
    )

    logging.basicConfig(level=logging.WARNING)

    # Redirect to a throwaway directory BEFORE anything runs. This test
    # deliberately corrupts its JSON files to prove the recovery paths work,
    # and it must never do that to real uploaded datasets. Rebinding the
    # module globals is enough: every function below resolves them by name.
    _TMP = Path(tempfile.mkdtemp(prefix="sih_store_test_"))
    DATASETS_JSON = _TMP / "datasets.json"
    RUNS_JSON = _TMP / "runs.json"
    print(f"(testing against {_TMP}, real storage/ untouched)")

    counts = {"HB": 233, "MI": 239, "Normal": 284, "PMI": 172}
    ds = DatasetMeta("d_test01", "ECG Cardiac (Multan)", "image_folder",
                     list(counts), counts, sum(counts.values()), 4)
    save_dataset(ds)
    save_dataset(ds)                                   # upsert, not duplicate
    assert sum(d.dataset_id == "d_test01" for d in list_datasets()) == 1
    assert get_dataset("d_test01").n_classes == 4
    assert get_dataset("nope") is None

    def _result(model, kind, acc):
        return ModelResult(
            model=model, kind=kind, label=model.upper(),
            metrics=Metrics(acc, acc, acc, acc),
            confusion_matrix=[[1]],
            telemetry=Telemetry(0.3, 0.01, "sklearn"),
        )

    cfg = RunConfig("d_test01", ["svc", "qsvc"], [OpConfig("grayscale")])
    save_run(RunRecord("r_test01", "d_test01", ds.name, ds.class_names, cfg,
                       [_result("svc", "classical", 0.91),
                        _result("qsvc", "quantum", 0.94),
                        ModelResult.failed("vqc", "quantum", "VQC", "boom")]))
    save_run(RunRecord("r_test02", "d_test01", ds.name, ds.class_names, cfg,
                       [_result("rf", "classical", 0.88),
                        _result("qsvc", "quantum", 0.90)]))

    run = get_run("r_test01")
    assert run.result_for("vqc").error == "boom"
    assert run.best("quantum").model == "qsvc"

    lb = leaderboard()
    assert len(lb) == 1
    assert lb[0]["best_classical"]["accuracy"] == 0.91      # best ACROSS runs
    assert lb[0]["best_quantum"]["accuracy"] == 0.94
    assert lb[0]["delta"] == 0.03
    assert lb[0]["n_runs"] == 2

    # a corrupt file must not kill the reader
    DATASETS_JSON.write_text("{not json", encoding="utf-8")
    assert list_datasets() == []
    assert not DATASETS_JSON.exists()                      # quarantined

    # a single bad row is skipped, good rows survive
    _write_rows(RUNS_JSON, [{"run_id": "r_broken"}] + _read_rows(RUNS_JSON))
    assert {r.run_id for r in list_runs()} == {"r_test01", "r_test02"}

    shutil.rmtree(_TMP, ignore_errors=True)
    print("store OK — upsert, atomic write, quarantine, skip-bad-row, leaderboard")
