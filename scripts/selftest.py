#!/usr/bin/env python3
"""
scripts/selftest.py
===================

Every check in the project, one command.

    python scripts/selftest.py           # fast suite, ~30s
    python scripts/selftest.py --full    # adds the model matrix, ~3 min
    python scripts/selftest.py --list    # show what would run, run nothing

Six groups:

  1. IMPORTS      every module loads
  2. INVARIANTS   the greps that protect the architecture
  3. SELF-TESTS   each module's own `if __name__ == "__main__"` block
  4. API          all 17 routes, in-process, against a real uploaded dataset
  5. FIXTURES     regenerate and validate against the response models
  6. MODELS       every model x encoding x {2, 4} classes   (--full only)

Exit code 0 if everything passed. Safe to run any time: the API group cleans
up the dataset it creates, and the module self-tests that touch disk already
redirect themselves to a temp directory.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

G, Y, R, DIM, OFF = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"

results: list[tuple[str, str, str, float]] = []   # group, name, state, seconds


def record(group: str, name: str, ok: bool | None, detail: str = "",
           secs: float = 0.0) -> None:
    state = "pass" if ok else ("skip" if ok is None else "FAIL")
    colour = G if ok else (Y if ok is None else R)
    results.append((group, name, state, secs))
    t = f"{secs:5.1f}s" if secs >= 0.05 else "      "
    print(f"  {colour}{state:<4}{OFF} {t}  {name:<34} {DIM}{detail[:60]}{OFF}")


# ---------------------------------------------------------------------------
# 1. imports
# ---------------------------------------------------------------------------

MODULES = [
    "backend.core.paths", "backend.core.contracts", "backend.core.ingest",
    "backend.core.preprocess", "backend.core.encodings",
    "backend.core.models_classical", "backend.core.models_quantum",
    "backend.core.registry", "backend.core.embed", "backend.core.project",
    "backend.core.benchmark", "backend.core.checkpoint",
    "backend.core.estimate",
    "backend.store", "backend.jobs",
    "backend.validate", "backend.schemas", "backend.main",
]


def group_imports() -> None:
    import importlib
    print(f"\n1. IMPORTS\n{'-' * 74}")
    for m in MODULES:
        t0 = time.perf_counter()
        try:
            importlib.import_module(m)
            record("imports", m, True, secs=time.perf_counter() - t0)
        except Exception as e:
            record("imports", m, False, f"{type(e).__name__}: {e}",
                   time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# 2. architecture invariants
# ---------------------------------------------------------------------------

def _grep(patterns: list[str], root: Path, suffixes: set[str]) -> list[str]:
    hits, rx = [], re.compile("|".join(patterns))
    if not root.exists():
        return hits
    for p in sorted(root.rglob("*")):
        if p.suffix not in suffixes or "node_modules" in p.parts:
            continue
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if line.lstrip().startswith(("#", "//", "*")):
                continue
            if rx.search(line):
                hits.append(f"{p.relative_to(ROOT)}:{i}")
    return hits


INVARIANTS = [
    ("no web stack under core/",
     [r"^\s*(from|import)\s+fastapi", r"^\s*(from|import)\s+pydantic"],
     "backend/core", {".py"}),
    ("core/ never imports upward",
     [r"from\s+backend\.(main|jobs|store|validate|schemas)"],
     "backend/core", {".py"}),
    ("no qiskit anywhere",
     [r"^\s*(from|import)\s+qiskit"], "backend", {".py"}),
    ("fetch() only in api.js",
     [r"\bfetch\s*\("], "frontend/src/pages", {".jsx", ".js"}),
    ("fetch() only in api.js",
     [r"\bfetch\s*\("], "frontend/src/components", {".jsx", ".js"}),
]


def group_invariants() -> None:
    print(f"\n2. INVARIANTS\n{'-' * 74}")
    seen = set()
    for name, pats, sub, sfx in INVARIANTS:
        hits = _grep(pats, ROOT / sub, sfx)
        key = (name, sub)
        if key in seen:
            continue
        seen.add(key)
        record("invariants", f"{name} ({sub})", not hits,
               ", ".join(hits[:2]) if hits else "clean")


# ---------------------------------------------------------------------------
# 3. module self-tests
# ---------------------------------------------------------------------------

SELF_TESTS = [
    "backend.core.paths", "backend.core.contracts", "backend.core.ingest",
    "backend.core.preprocess", "backend.core.encodings",
    "backend.core.registry", "backend.core.embed", "backend.core.project",
    "backend.core.estimate", "backend.store", "backend.jobs",
    "backend.validate", "backend.schemas",
]


def _run(args: list[str], timeout: int = 900) -> tuple[bool, str]:
    p = subprocess.run([sys.executable, *args], cwd=ROOT,
                       capture_output=True, text=True, timeout=timeout)
    out = (p.stdout.strip().splitlines() or [""])[-1]
    err = (p.stderr.strip().splitlines() or [""])[-1]
    return p.returncode == 0, (out if p.returncode == 0 else err)


def group_selftests() -> None:
    print(f"\n3. SELF-TESTS\n{'-' * 74}")
    for m in SELF_TESTS:
        t0 = time.perf_counter()
        ok, msg = _run(["-m", m])
        record("self-tests", m, ok, msg, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# 4. API — all 17 routes in-process
# ---------------------------------------------------------------------------

def _synthetic_zip(n_per_class: int = 8) -> bytes:
    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for ci, cls in enumerate(["HB", "MI", "Normal", "PMI"]):
            for i in range(n_per_class):
                a = np.full((240, 360, 3), 255, np.uint8)
                for x in range(0, 360, 10):
                    a[:, x] = (255, 200, 205)
                for row in range(3):
                    base = 50 + row * 70
                    for x in range(360):
                        yv = int(base + 10 * np.sin(x / 7.0 + ci + i))
                        a[max(0, yv - 1):yv + 2, x] = (10, 10, 10)
                a[:, 179:182] = (10, 10, 10)
                b = io.BytesIO()
                Image.fromarray(a).save(b, format="PNG")
                zf.writestr(f"ECG/{cls}/{i}.png", b.getvalue())
    return buf.getvalue()


def group_api() -> None:
    print(f"\n4. API\n{'-' * 74}")
    import logging
    logging.disable(logging.ERROR)
    try:
        from fastapi.testclient import TestClient
    except Exception as exc:
        # starlette >= 1.6 requires httpx2 for TestClient. It is a test-only
        # dependency, so a missing one skips this group rather than failing a
        # suite that is otherwise green.
        logging.disable(logging.NOTSET)
        hint = "pip install httpx2" if "httpx" in str(exc) else str(exc)[:50]
        record("api", "TestClient unavailable", None, hint)
        return

    from backend.core.paths import dataset_dir
    from backend.main import app

    c = TestClient(app)
    ds_id = None

    try:
        r = c.get("/health")
        wired = r.json().get("wired", {})
        record("api", "GET /health", r.status_code == 200,
               "wired: " + ",".join(k for k, v in wired.items() if v))

        t0 = time.perf_counter()
        r = c.post("/datasets",
                   files={"file": ("t.zip", _synthetic_zip(), "application/zip")},
                   data={"name": "selftest (delete me)"})
        ok = r.status_code == 200
        if ok:
            ds_id = r.json()["dataset_id"]
            detail = (f"{r.json()['n_classes']} classes, "
                      f"{r.json()['n_samples']} imgs, "
                      f"{len(r.json()['preview_b64'])} thumbs")
        else:
            detail = str(r.json())[:60]
        record("api", "POST /datasets", ok, detail, time.perf_counter() - t0)
        if not ok:
            return

        record("api", "GET /datasets",
               any(d["dataset_id"] == ds_id for d in c.get("/datasets").json()))

        for ep in ["/ops/catalog", "/presets", "/encodings", "/models/catalog",
                   "/backbones"]:
            r = c.get(ep)
            if r.status_code == 200:
                record("api", f"GET {ep}", True,
                       f"{len(r.json())} entries")
            elif r.status_code == 503:
                record("api", f"GET {ep}", None, r.json()["detail"][:60])
            else:
                record("api", f"GET {ep}", False, str(r.json())[:60])

        t0 = time.perf_counter()
        r = c.post("/preprocess/preview", json={
            "dataset_id": ds_id,
            "ops": [{"op": "crop_border"}, {"op": "remove_grid"},
                    {"op": "grayscale"}, {"op": "binarize"},
                    {"op": "remove_lead_lines"}]})
        record("api", "POST /preprocess/preview", r.status_code == 200,
               " -> ".join(s["op"] for s in r.json().get("stages", []))[:60],
               time.perf_counter() - t0)

        good = {"dataset_id": ds_id, "models": ["svc", "qsvc"],
                "n_features": 8, "encoding": "angle_y",
                "ops": [{"op": "grayscale"}, {"op": "binarize"}]}

        r = c.post("/runs/validate", json=good)
        record("api", "POST /runs/validate (ok)",
               r.status_code == 200 and not r.json()["errors"],
               f"{len(r.json().get('warnings', []))} warning(s)")

        r = c.post("/runs/validate", json={**good, "test_size": 0.9})
        record("api", "POST /runs/validate (bad)",
               r.status_code == 200 and bool(r.json()["errors"]),
               (r.json().get("errors") or [""])[0][:58])

        r = c.post("/runs/validate", json={**good, "nfeatures": 8})
        record("api", "POST /runs/validate (typo -> 422)", r.status_code == 422,
               "unknown field rejected")

        for ep, payload in [("/runs/estimate", good), ("/runs", good)]:
            r = c.post(ep, json=payload)
            if r.status_code == 200:
                record("api", f"POST {ep}", True, str(r.json())[:58])
            elif r.status_code == 503:
                record("api", f"POST {ep}", None, r.json()["detail"][:58])
            else:
                record("api", f"POST {ep}", False, str(r.json())[:58])

        record("api", "GET /runs", c.get("/runs").status_code == 200)
        record("api", "GET /leaderboard", c.get("/leaderboard").status_code == 200)

        for ep in ["/runs/r_nope", "/jobs/j_nope"]:
            record("api", f"GET {ep} -> 404", c.get(ep).status_code == 404)

        r = c.post("/datasets", files={"file": ("x.zip", b"not a zip",
                                                "application/zip")})
        record("api", "POST /datasets (junk) -> 400",
               r.status_code == 400 and "detail" in r.json(),
               r.json().get("detail", "")[:58])

    finally:
        logging.disable(logging.NOTSET)
        if ds_id:
            shutil.rmtree(dataset_dir(ds_id), ignore_errors=True)
            from backend.core.paths import DATASETS_JSON
            if DATASETS_JSON.exists():
                rows = [d for d in json.loads(DATASETS_JSON.read_text())
                        if d.get("dataset_id") != ds_id]
                DATASETS_JSON.write_text(json.dumps(rows, indent=2))
            record("api", "cleanup", True, f"removed {ds_id}")


# ---------------------------------------------------------------------------
# 5. fixtures
# ---------------------------------------------------------------------------

def group_fixtures() -> None:
    print(f"\n5. FIXTURES\n{'-' * 74}")
    t0 = time.perf_counter()
    ok, msg = _run(["scripts/gen_fixtures.py"])
    n = len(list((ROOT / "fixtures").glob("*.json")))
    record("fixtures", "gen_fixtures.py", ok,
           f"{n} files, {msg}"[:60], time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# 6. model matrix  (--full)
# ---------------------------------------------------------------------------

def group_models(full: bool) -> None:
    print(f"\n6. MODELS\n{'-' * 74}")
    if not full:
        record("models", "scripts/test_models.py", None, "skipped (use --full)")
        return
    t0 = time.perf_counter()
    ok, msg = _run(["-m", "scripts.test_models"], timeout=2400)
    record("models", "every model x encoding x {2,4} classes", ok, msg,
           time.perf_counter() - t0)


# ---------------------------------------------------------------------------

def main() -> int:
    full = "--full" in sys.argv
    if "--list" in sys.argv:
        print("1. IMPORTS      ", len(MODULES), "modules")
        print("2. INVARIANTS   ", len(INVARIANTS), "greps")
        print("3. SELF-TESTS   ", len(SELF_TESTS), "modules")
        print("4. API           17 routes, in-process")
        print("5. FIXTURES      gen_fixtures.py")
        print("6. MODELS        scripts/test_models.py   (--full only)")
        return 0

    start = time.perf_counter()
    print(f"selftest — {ROOT}")
    group_imports()
    group_invariants()
    group_selftests()
    group_api()
    group_fixtures()
    group_models(full)

    passed = sum(1 for *_, s, _ in ((None, *r[1:]) for r in results) if s == "pass")
    failed = [r for r in results if r[2] == "FAIL"]
    skipped = sum(1 for r in results if r[2] == "skip")

    print("\n" + "=" * 74)
    print(f"  {G}{passed} passed{OFF}   {Y}{skipped} skipped{OFF}   "
          f"{R}{len(failed)} failed{OFF}      {time.perf_counter() - start:.1f}s")
    if failed:
        print()
        for g, n, _, _ in failed:
            print(f"  {R}FAIL{OFF} [{g}] {n}")
        return 1
    if not full:
        print(f"  {DIM}run with --full to include the model matrix{OFF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
