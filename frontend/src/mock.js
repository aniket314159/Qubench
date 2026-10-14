/**
 * frontend/src/mock.js
 * ====================
 *
 * Serves fixtures/ so the app is fully clickable with no backend running:
 *
 *     VITE_MOCK=true npm run dev
 *
 * The fixtures are generated, not written by hand:
 *
 *     python scripts/gen_fixtures.py
 *
 * Every one is validated against its Pydantic response model before being
 * written, so what you build against here is what the server will send. Re-run
 * that script after any change to contracts.py, schemas.py or a registry.
 *
 * Only api.js imports this. Components never touch it, so deleting MOCK later
 * means deleting one import.
 */

import health from "../../fixtures/health.json";
import dataset from "../../fixtures/dataset.json";
import datasets from "../../fixtures/datasets.json";
import opsCatalog from "../../fixtures/ops_catalog.json";
import presets from "../../fixtures/presets.json";
import backbones from "../../fixtures/backbones.json";
import encodings from "../../fixtures/encodings.json";
import modelsCatalog from "../../fixtures/models_catalog.json";
import preview from "../../fixtures/preview.json";
import validate from "../../fixtures/validate.json";
import estimate from "../../fixtures/estimate.json";
import jobSubmit from "../../fixtures/job_submit.json";
import jobProgress from "../../fixtures/job_progress.json";
import runs from "../../fixtures/runs.json";
import runFull from "../../fixtures/run_full.json";
import leaderboard from "../../fixtures/leaderboard.json";
import predict from "../../fixtures/predict.json";

const FIXTURES = {
  health,
  dataset,
  datasets,
  ops_catalog: opsCatalog,
  presets,
  backbones,
  encodings,
  models_catalog: modelsCatalog,
  preview,
  validate,
  estimate,
  job_submit: jobSubmit,
  job_progress: jobProgress,
  runs,
  run_full: runFull,
  leaderboard,
  predict,
};

/** Latency is deliberate. A mock that answers instantly hides every missing
 *  loading state, and those are exactly what breaks on the day the real
 *  backend is plugged in. */
const DEFAULT_DELAY = 320;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function get(name, delay = DEFAULT_DELAY) {
  await sleep(delay);
  const data = FIXTURES[name];
  if (data === undefined) throw new Error(`no fixture named ${name}`);
  // Deep copy so a component that mutates a response cannot poison the next
  // read — the real API hands out a fresh object every time.
  return structuredClone(data);
}

/**
 * Job polling, animated.
 *
 * job_progress.json is a sequence of eleven JobState snapshots, not one. Each
 * poll advances one step, so the progress bar actually moves, the status
 * ladder walks preprocessing → embedding → projecting → training → done, and
 * JobProgress.jsx can be built without a server. A static 74% would let a
 * broken bar ship.
 */
const cursors = new Map();

export async function job(jobId) {
  await sleep(120);
  const i = cursors.get(jobId) ?? 0;
  const step = jobProgress[Math.min(i, jobProgress.length - 1)];
  cursors.set(jobId, i + 1);
  return structuredClone(step);
}

/** Call between demo runs so the next one starts at 0% again. */
export function resetJobs() {
  cursors.clear();
}

/** Force the next poll of `jobId` to return a failure, for building the error
 *  state. There is no other way to see it without breaking the backend. */
export function failNext(jobId) {
  cursors.set(jobId, -1);
  FIXTURES.job_progress = [
    {
      status: "failed",
      pct: 41,
      message: "Run failed",
      run_id: null,
      error: "MemoryError: statevector for 14 qubits needs 8.0 GB",
    },
  ];
}
