"""
backend/jobs.py
===============

SEAM 5 — the job runner. Phase 1 is a dict plus FastAPI's BackgroundTasks.

    submit_job(work, schedule) -> job_id
    get_job(job_id)            -> JobState | None

Phase 3 swaps this file for Celery. Nothing else changes, because routes only
ever see a `job_id` and a `JobState`, never a thread or a task object.

    queued ──► preprocessing ──► embedding ──► projecting ──► training ──► done
                  │                  │             │             │
                  └──────────────────┴─────────────┴─────────────┴──► failed

A `work` function is anything with the signature:

    work(progress: ProgressFn) -> str        # returns the run_id it created

It receives one callback, `progress(pct, message)`, and knows nothing about
jobs, HTTP, or the store. `benchmark.run_benchmark` is the only real one.

No `import fastapi` here either, even though this file sits in backend/ where
it would be legal. `submit_job` takes a `schedule` callable, so main.py passes
`background_tasks.add_task` and the tests pass a thread. That keeps this file
runnable on its own — which is how the smoke test at the bottom exists.
"""

from __future__ import annotations

import logging
import threading
import traceback
from typing import Any, Callable

from backend.core.contracts import (
    STAGE_PCT, JobState, JobStatus, ProgressFn, new_id, utc_now_iso,
)

log = logging.getLogger(__name__)

# work(progress) -> run_id
JobWork = Callable[[ProgressFn], str]
# schedule(fn, *args) -> None   e.g. BackgroundTasks.add_task, or a Thread
Scheduler = Callable[..., Any]

JOBS: dict[str, JobState] = {}
_META: dict[str, dict[str, Any]] = {}          # created/finished, for eviction
_LOCK = threading.RLock()

MAX_JOBS = 100          # ring buffer; finished jobs evict oldest-first

# One run at a time. ResNet18 inference and a statevector simulator are both
# CPU-bound and single-machine; two concurrent runs do not finish in half the
# time, they finish in rather more than the sum while fighting over 4 cores.
# A second submission sits in `queued` with an honest message instead.
_SLOT = threading.Semaphore(1)

# pct -> status, so `work` only ever calls progress(pct, message) and the state
# machine follows along. Keeps ProgressFn exactly as pinned in contracts.
_LADDER: list[tuple[int, JobStatus]] = sorted(
    ((STAGE_PCT[s], s) for s in
     ("preprocessing", "embedding", "projecting", "training")),
    reverse=True,
)


def _status_for(pct: int) -> JobStatus:
    for anchor, status in _LADDER:
        if pct >= anchor:
            return status
    return "preprocessing"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _evict() -> None:
    """Called with the lock held. Drop the oldest finished jobs once over
    MAX_JOBS. Never evicts a job that is still running."""
    if len(JOBS) <= MAX_JOBS:
        return
    finished = sorted(
        (jid for jid, s in JOBS.items() if s.status in ("done", "failed")),
        key=lambda jid: _META[jid]["created"],
    )
    for jid in finished[: len(JOBS) - MAX_JOBS]:
        JOBS.pop(jid, None)
        _META.pop(jid, None)


def create_job() -> str:
    with _LOCK:
        jid = new_id("j")
        while jid in JOBS:
            jid = new_id("j")
        JOBS[jid] = JobState(status="queued", pct=0, message="Queued")
        _META[jid] = {"created": utc_now_iso(), "finished": None}
        _evict()
    return jid


def get_job(job_id: str) -> JobState | None:
    """GET /jobs/{job_id}. None -> the route returns 404."""
    with _LOCK:
        state = JOBS.get(job_id)
        # Hand back a copy. The caller is serialising it while the worker
        # thread may be mutating the original.
        return None if state is None else JobState(**state.to_dict())


def list_jobs() -> dict[str, JobState]:
    with _LOCK:
        return {jid: JobState(**s.to_dict()) for jid, s in JOBS.items()}


def _update(job_id: str, **fields: Any) -> None:
    with _LOCK:
        state = JOBS.get(job_id)
        if state is None:
            return
        # pct is monotonic. A stage that reports a lower number than the one
        # before must not make the bar jump backwards.
        if "pct" in fields:
            fields["pct"] = max(state.pct, int(fields["pct"]))
        for k, v in fields.items():
            setattr(state, k, v)


def make_progress(job_id: str) -> ProgressFn:
    """The callback handed to `work`. Clamps, keeps pct monotonic, and derives
    `status` from the percentage."""
    def progress(pct: int, message: str = "") -> None:
        pct = max(0, min(100, int(pct)))
        fields: dict[str, Any] = {"pct": pct, "status": _status_for(pct)}
        if message:
            fields["message"] = message
        _update(job_id, **fields)
    return progress


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_job(job_id: str, work: JobWork) -> None:
    """The scheduled body. Never raises — a background task that throws dies
    silently in the threadpool and the frontend polls a job that never moves.
    Everything becomes a `failed` state with a message instead."""
    if job_id not in JOBS:
        log.error("run_job called for unknown job %s", job_id)
        return

    _update(job_id, status="queued", message="Waiting for a free slot")
    with _SLOT:                                    # serialise runs
        _update(job_id, status="preprocessing", pct=STAGE_PCT["preprocessing"],
                message="Starting")
        try:
            run_id = work(make_progress(job_id))
            _update(job_id, status="done", pct=100, run_id=run_id,
                    message="Done", error=None)
        except Exception as exc:
            log.exception("job %s failed", job_id)
            _update(
                job_id,
                status="failed",
                message="Run failed",
                # Type plus message, not the traceback. The traceback is in the
                # server log; the UI needs one readable line.
                error=f"{type(exc).__name__}: {exc}"[:500] or
                      traceback.format_exc(limit=1)[:500],
            )
        finally:
            with _LOCK:
                if job_id in _META:
                    _META[job_id]["finished"] = utc_now_iso()


def submit_job(work: JobWork, schedule: Scheduler) -> str:
    """POST /runs. Returns immediately with a job_id; `schedule` decides when
    the work actually starts.

        jid = jobs.submit_job(work, background_tasks.add_task)
    """
    job_id = create_job()
    schedule(run_job, job_id, work)
    return job_id


def thread_scheduler(fn: Callable[..., Any], *args: Any) -> None:
    """A `schedule` that does not need FastAPI. Used by the smoke test and by
    scripts/pipeline.py if it ever wants the same progress plumbing."""
    threading.Thread(target=fn, args=args, daemon=True).start()


def reset() -> None:
    """Tests only. Clears the registry."""
    with _LOCK:
        JOBS.clear()
        _META.clear()


# ---------------------------------------------------------------------------
# Smoke test:  python -m backend.jobs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.CRITICAL)     # the failure test logs

    def wait_for(job_id: str, *, timeout: float = 5.0) -> JobState:
        end = time.time() + timeout
        while time.time() < end:
            s = get_job(job_id)
            if s and s.status in ("done", "failed"):
                return s
            time.sleep(0.01)
        raise AssertionError(f"job {job_id} never finished: {get_job(job_id)}")

    # --- happy path, and the status ladder follows pct ---------------------
    seen: list[tuple[str, int]] = []

    def good(progress: ProgressFn) -> str:
        for pct, msg in [(5, "ingest"), (20, "embed"), (55, "project"),
                         (60, "Training svc (1/2)"), (95, "Training qsvc (2/2)")]:
            progress(pct, msg)
            s = get_job(jid)
            seen.append((s.status, s.pct))
            time.sleep(0.01)
        return "r_abc123"

    jid = create_job()
    thread_scheduler(run_job, jid, good)    # bind jid BEFORE the thread starts;
                                            # `good` closes over it
    final = wait_for(jid)
    assert final.status == "done" and final.pct == 100
    assert final.run_id == "r_abc123" and final.error is None
    assert [s for s, _ in seen] == ["preprocessing", "embedding", "projecting",
                                    "training", "training"], seen

    # --- failure path: the exception becomes a state, never a lost job -----
    def bad(progress: ProgressFn) -> str:
        progress(20, "embedding")
        raise MemoryError("statevector needs 8.0 GB")

    f = wait_for(submit_job(bad, thread_scheduler))
    assert f.status == "failed"
    assert f.error == "MemoryError: statevector needs 8.0 GB"
    assert f.run_id is None
    assert f.pct == 20, "pct freezes where it failed"

    # --- pct never goes backwards -----------------------------------------
    def jittery(progress: ProgressFn) -> str:
        progress(60, "training")
        progress(20, "oops, a stale callback")
        assert get_job(jid2).pct == 60
        return "r_mono"

    jid2 = create_job()
    run_job(jid2, jittery)
    assert get_job(jid2).pct == 100

    # --- runs are serialised, not interleaved ------------------------------
    order: list[str] = []

    def slow(tag: str) -> JobWork:
        def work(progress: ProgressFn) -> str:
            order.append(f"{tag}-start")
            time.sleep(0.15)
            order.append(f"{tag}-end")
            return f"r_{tag}"
        return work

    a = submit_job(slow("A"), thread_scheduler)
    time.sleep(0.02)
    b = submit_job(slow("B"), thread_scheduler)
    time.sleep(0.02)
    assert get_job(b).status == "queued", "second run should wait its turn"
    wait_for(a); wait_for(b, timeout=5.0)
    assert order == ["A-start", "A-end", "B-start", "B-end"], order

    # --- get_job returns a snapshot, not a live handle ---------------------
    snap = get_job(a)
    snap.pct = -999
    assert get_job(a).pct == 100

    assert get_job("j_nope") is None

    # --- eviction keeps the dict bounded, never drops a live job -----------
    reset()
    for _ in range(MAX_JOBS + 20):
        wait_for(submit_job(lambda p: "r_x", thread_scheduler), timeout=5.0)
    assert len(JOBS) <= MAX_JOBS, len(JOBS)

    reset()
    print("jobs OK — status ladder, failure capture, monotonic pct, "
          "serialised runs, snapshot reads, eviction")
