#!/usr/bin/env python3
"""
scripts/status.py
=================

Project inventory. What is built, what is a stub, what is missing.

    python scripts/status.py            # the table
    python scripts/status.py --test     # also run every module's self-test

A file under 200 bytes counts as a stub — an empty placeholder, not an
implementation. That distinction matters here: an empty `encodings.py` imports
perfectly fine, so `import` alone cannot tell you whether W4 has started.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUB_BYTES = 200

# (path, owner, note, has_selftest)
SECTIONS: list[tuple[str, list[tuple[str, str, str, bool]]]] = [
    ("W1 — frozen contracts", [
        ("backend/core/contracts.py", "W1", "every dataclass + BaseModel ABC", True),
        ("backend/core/paths.py", "W1", "every filesystem location", True),
        ("backend/schemas.py", "W1", "Pydantic mirrors + drift guard", True),
    ]),
    ("W3 — backend", [
        ("backend/__init__.py", "W1", "must exist, must be empty", False),
        ("backend/core/__init__.py", "W1", "must exist, must be empty", False),
        ("backend/store.py", "W3", "SEAM 6 — JSON persistence", True),
        ("backend/jobs.py", "W3", "SEAM 5 — job runner", True),
        ("backend/validate.py", "W3", "config rules", True),
        ("backend/main.py", "W3", "17 routes", False),
        ("backend/core/ingest.py", "W3", "SEAM 1 — zip to DatasetMeta", True),
        ("backend/core/preprocess.py", "W3", "SEAM 2 — op registry", True),
    ]),
    ("W4 — model layer", [
        ("backend/core/encodings.py", "W4", "SEAM 3.5 — 4 encodings", True),
        ("backend/core/registry.py", "W4", "SEAM 4 — MODEL_REGISTRY", True),
        ("backend/core/models_classical.py", "W4", "SVC, RandomForest", False),
        ("backend/core/models_quantum.py", "W4", "QSVC, VQC", False),
        ("backend/core/embed.py", "W4", "SEAM 3 — 4 backbones + cache", True),
        ("backend/core/project.py", "W4", "split -> PCA -> scaler, no leakage", True),
        ("backend/core/benchmark.py", "W4", "train loop, metrics, run_for_dataset", False),
        ("backend/core/checkpoint.py", "W4", "bundle save/load", False),
        ("backend/core/estimate.py", "W4", "cost model + calibration", True),
        ("scripts/pipeline.py", "W4", "CLI, no web layer", False),
    ]),
    ("W2 — frontend", [
        ("frontend/src/api.js", "W2", "the only fetch() in the app", False),
        ("frontend/src/mock.js", "W2", "MOCK=true fixture switch", False),
        ("frontend/src/App.jsx", "W2", "router + shell", False),
        ("frontend/src/pages/Upload.jsx", "W2", "", False),
        ("frontend/src/pages/Configure.jsx", "W2", "the centrepiece", False),
        ("frontend/src/pages/Benchmark.jsx", "W2", "", False),
        ("frontend/src/pages/Diagnose.jsx", "W2", "", False),
        ("frontend/vite.config.js", "W2", "proxy to :8000", False),
    ]),
    ("W1 — fixtures (generated: python scripts/gen_fixtures.py)", [
        (f"fixtures/{n}.json", "W1", note, False) for n, note in [
            ("health", "GET /health"),
            ("dataset", "POST /datasets"),
            ("datasets", "GET /datasets"),
            ("ops_catalog", "GET /ops/catalog"),
            ("presets", "GET /presets"),
            ("backbones", "GET /backbones"),
            ("encodings", "GET /encodings"),
            ("models_catalog", "GET /models/catalog"),
            ("preview", "POST /preprocess/preview"),
            ("validate", "POST /runs/validate"),
            ("estimate", "POST /runs/estimate"),
            ("job_submit", "POST /runs"),
            ("job_progress", "GET /jobs/{id} — poll sequence"),
            ("runs", "GET /runs"),
            ("run_full", "GET /runs/{id}"),
            ("leaderboard", "GET /leaderboard"),
            ("predict", "POST /predict/{run}/{model}"),
        ]
    ]),
    ("Tooling", [
        ("scripts/status.py", "W1", "this file", False),
        ("scripts/gen_fixtures.py", "W1", "regenerate fixtures from live code", False),
        ("scripts/test_models.py", "W4", "n_classes hunt across models x encodings", False),
        ("scripts/calibrate.py", "W4", "fit the cost model to THIS machine", False),
        ("scripts/selftest.py", "W1", "every check, one command", False),
    ]),
    ("Project files", [
        ("README.md", "W1", "project overview", False),
        ("backend/README.md", "W1", "backend setup", False),
        ("backend/requirements.txt", "W1", "pinned deps", False),
        ("backend/requirements.lock.txt", "W1", "pip freeze", False),
        (".gitignore", "W1", "", False),
        ("run.sh", "W1", "", False),
    ]),
]

GREEN, YELLOW, RED, DIM, OFF = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"


def classify(path: Path) -> tuple[str, str, int]:
    if not path.exists():
        return "missing", RED, 0
    size = path.stat().st_size
    if path.name == "__init__.py":
        return ("ok" if size == 0 else "not empty"), (GREEN if size == 0 else YELLOW), size
    # The stub heuristic is about unwritten SOURCE. A generated fixture can be
    # legitimately tiny — job_submit.json is {"job_id": "..."} and complete.
    if path.suffix not in {".py", ".jsx", ".js"}:
        return ("built" if size else "empty"), (GREEN if size else YELLOW), size
    if size < STUB_BYTES:
        return "stub", YELLOW, size
    return "built", GREEN, size


def main() -> int:
    run_tests = "--test" in sys.argv
    totals = {"built": 0, "stub": 0, "missing": 0, "ok": 0,
              "not empty": 0, "empty": 0}
    selftests: list[tuple[str, str]] = []

    for title, entries in SECTIONS:
        print(f"\n{title}")
        print("─" * 74)
        for rel, owner, note, has_test in entries:
            path = ROOT / rel
            state, colour, size = classify(path)
            totals[state] = totals.get(state, 0) + 1
            kb = f"{size/1024:6.1f}K" if size else "      -"
            print(f"  {colour}{state:<9}{OFF} {kb}  {rel:<38} {DIM}{note}{OFF}")
            if has_test and state == "built":
                selftests.append((rel, "backend." + rel[8:-3].replace("/", ".")))

    built = totals["built"] + totals["ok"]
    print("\n" + "─" * 74)
    print(f"  {GREEN}built {built}{OFF}   {YELLOW}stub {totals['stub']}{OFF}   "
          f"{RED}missing {totals['missing']}{OFF}")

    if not run_tests:
        print(f"\n{DIM}  run with --test to execute every module self-test{OFF}")
        return 0

    print(f"\nSelf-tests ({len(selftests)})")
    print("─" * 74)
    failed = 0
    for rel, module in selftests:
        proc = subprocess.run([sys.executable, "-m", module], cwd=ROOT,
                              capture_output=True, text=True, timeout=300)
        tail = (proc.stdout.strip().splitlines() or ["(no output)"])[-1]
        if proc.returncode == 0:
            print(f"  {GREEN}pass{OFF}  {rel:<38} {DIM}{tail[:60]}{OFF}")
        else:
            failed += 1
            err = (proc.stderr.strip().splitlines() or ["?"])[-1]
            print(f"  {RED}FAIL{OFF}  {rel:<38} {err[:60]}")
    print("─" * 74)
    print(f"  {len(selftests) - failed}/{len(selftests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
