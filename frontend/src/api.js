/**
 * frontend/src/api.js
 * ===================
 *
 * The only file in this app that calls fetch(). Everything else takes props.
 *
 * Two rules that keep Phase 2 free:
 *
 *   1. No model, op, encoding, backbone or class name is written down here or
 *      in any component. They all arrive from the catalog endpoints. If you
 *      find yourself typing "qsvc" into a .jsx file, the seam has leaked.
 *   2. Every function returns plain data or throws ApiError. No component
 *      should ever see a Response object or a status code.
 *
 * MOCK MODE
 * ---------
 * `VITE_MOCK=true npm run dev` serves everything from fixtures/ with a small
 * delay, so the whole app is clickable with no Python running. The fixtures are
 * generated from the live backend by `python scripts/gen_fixtures.py`, so they
 * are the real shapes, not hand-written guesses.
 */

import * as mock from "./mock";

export const MOCK = import.meta.env.VITE_MOCK === "true";
const BASE = import.meta.env.VITE_API_BASE ?? "/api";

/** Everything thrown by this module. `detail` is the backend's own message —
 *  safe and written for a human, so components can render it directly. */
export class ApiError extends Error {
  constructor(detail, status) {
    super(detail);
    this.name = "ApiError";
    this.detail = detail;
    this.status = status;
  }
}

async function request(path, { method = "GET", body, form, signal } = {}) {
  let res;
  try {
    res = await fetch(BASE + path, {
      method,
      signal,
      headers: form ? undefined : { "Content-Type": "application/json" },
      body: form ?? (body === undefined ? undefined : JSON.stringify(body)),
    });
  } catch (e) {
    if (e.name === "AbortError") throw e;
    // Distinguish "server is not running" from "server said no". The first is
    // the commonest thing that happens during development and deserves its own
    // sentence rather than a generic network error.
    throw new ApiError(
      "Cannot reach the backend. Start it with ./run.sh api, or run the app " +
        "with VITE_MOCK=true to use fixtures.",
      0,
    );
  }

  if (res.status === 204) return null;

  let data = null;
  try {
    data = await res.json();
  } catch {
    /* empty or non-JSON body */
  }

  if (!res.ok) {
    // FastAPI puts a string in `detail` for our errors and an array of field
    // objects for 422s. Flatten the array into something a banner can show.
    let detail = data?.detail;
    if (Array.isArray(detail)) {
      detail = detail
        .map((d) => `${(d.loc ?? []).slice(1).join(".")}: ${d.msg}`)
        .join("; ");
    }
    throw new ApiError(detail || `Request failed (${res.status})`, res.status);
  }
  return data;
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

/** `{ ok, version, wired: { encodings, models, backbones, estimate,
 *  benchmark, predict } }`. Use `wired` to hide controls whose backend half is
 *  not written yet rather than letting the user hit a 503. */
export const getHealth = () => (MOCK ? mock.get("health") : request("/health"));

// ---------------------------------------------------------------------------
// Datasets
// ---------------------------------------------------------------------------

/** POST a zip. Returns DatasetMeta plus `preview_b64` — four data URIs, ready
 *  to drop straight into <img src>. The whole Upload screen renders from this
 *  one response; there is no second request. */
export async function uploadDataset(file, { name, modality = "ecg" } = {}) {
  if (MOCK) return mock.get("dataset", 900);
  const form = new FormData();
  form.append("file", file);
  if (name) form.append("name", name);
  form.append("modality", modality);
  return request("/datasets", { method: "POST", form });
}

/** DatasetMeta[]. `preview_b64` is empty here on purpose — the list endpoint
 *  does not ship four base64 images per row. */
export const listDatasets = () =>
  MOCK ? mock.get("datasets") : request("/datasets");

// ---------------------------------------------------------------------------
// Catalogs — every control on the Configure screen is built from these
// ---------------------------------------------------------------------------

/** `{ op, label, order, params_schema, locked, description }[]`, already in
 *  execution order. Render in the order given: it is the order ops run in, and
 *  that order is load-bearing (grid removal must precede greyscale). */
export const getOpsCatalog = () =>
  MOCK ? mock.get("ops_catalog") : request("/ops/catalog");

/** `{ name, label, ops, description }[]` */
export const getPresets = () =>
  MOCK ? mock.get("presets") : request("/presets");

/** `{ name, label, dim, input_size }[]` */
export const getBackbones = () =>
  MOCK ? mock.get("backbones") : request("/backbones");

/** `{ name, label, max_features, qubit_formula, description }[]`.
 *
 *  Pass `nFeatures` and each entry also carries the REAL decomposed circuit at
 *  that width — qubits, depth, two-qubit gate count and the op list — so the
 *  circuit diagram draws what is about to run rather than a generic picture. */
export const getEncodings = (nFeatures) =>
  MOCK
    ? mock.get("encodings")
    : request(`/encodings${nFeatures ? `?n_features=${nFeatures}` : ""}`);

/** `{ name, kind, label, param_schema }[]`, classical first then quantum. */
export const getModelsCatalog = () =>
  MOCK ? mock.get("models_catalog") : request("/models/catalog");

/** All five catalogs at once. The Configure screen needs every one before it
 *  can render anything, so fetch them together and fail together. */
export async function getCatalogs() {
  const [ops, presets, backbones, encodings, models] = await Promise.all([
    getOpsCatalog(),
    getPresets(),
    getBackbones(),
    getEncodings(),
    getModelsCatalog(),
  ]);
  return { ops, presets, backbones, encodings, models };
}

// ---------------------------------------------------------------------------
// Preprocessing preview
// ---------------------------------------------------------------------------

/** `{ original_b64, stages: [{ op, image_b64 }] }`. Roughly 200 KB, so debounce
 *  this behind slider drags rather than firing on every pixel of movement. */
export const previewPreprocess = (datasetId, ops, imageIndex = 0, signal) =>
  MOCK
    ? mock.get("preview", 500)
    : request("/preprocess/preview", {
        method: "POST",
        body: { dataset_id: datasetId, ops, image_index: imageIndex },
        signal,
      });

// ---------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------

/** `{ errors, warnings }`. Non-empty `errors` disables the Run button; both
 *  arrays are plain sentences written to be shown verbatim. */
export const validateRun = (config) =>
  MOCK
    ? mock.get("validate")
    : request("/runs/validate", { method: "POST", body: config });

/** `{ n_qubits, hilbert_dim, sims, mem_mb, est_seconds }` */
export const estimateRun = (config) =>
  MOCK
    ? mock.get("estimate")
    : request("/runs/estimate", { method: "POST", body: config });

/** `{ job_id }`. The run happens in the background; poll with pollJob. */
export const startRun = (config) =>
  MOCK
    ? mock.get("job_submit")
    : request("/runs", { method: "POST", body: config });

/** `{ status, pct, message, run_id, error }` */
export const getJob = (jobId) =>
  MOCK ? mock.job(jobId) : request(`/jobs/${encodeURIComponent(jobId)}`);

/** Run summaries, newest first. */
export const listRuns = () => (MOCK ? mock.get("runs") : request("/runs"));

/** The full RunRecord: config, results, confusion matrices, telemetry. */
export const getRun = (runId) =>
  MOCK ? mock.get("run_full") : request(`/runs/${encodeURIComponent(runId)}`);

/** Best classical vs best quantum per dataset, across every run. */
export const getLeaderboard = () =>
  MOCK ? mock.get("leaderboard") : request("/leaderboard");

// ---------------------------------------------------------------------------
// Diagnose
// ---------------------------------------------------------------------------

/** `{ label, confidences, latency_ms }`. `confidences` is keyed by class name
 *  and sums to 1. */
export async function predict(runId, model, file) {
  if (MOCK) return mock.get("predict", 700);
  const form = new FormData();
  form.append("file", file);
  return request(
    `/predict/${encodeURIComponent(runId)}/${encodeURIComponent(model)}`,
    { method: "POST", form },
  );
}

// ---------------------------------------------------------------------------
// Polling
// ---------------------------------------------------------------------------

export const POLL_MS = 800;

/**
 * Poll a job until it finishes. Resolves with the final JobState.
 *
 *   const stop = pollJob(jobId, setJob).catch(...)
 *
 * Three things this handles that a bare setInterval does not:
 *   - it stops on `done` and `failed`, not just `done`;
 *   - one failed poll does not kill the loop, because a reload in dev drops a
 *     request and the run is still going;
 *   - `signal` lets a component abort on unmount, so an unmounted Benchmark
 *     page does not keep setting state.
 */
export function pollJob(
  jobId,
  onUpdate,
  { signal, intervalMs = POLL_MS } = {},
) {
  return new Promise((resolve, reject) => {
    let misses = 0;

    const tick = async () => {
      if (signal?.aborted) return resolve(null);
      try {
        const state = await getJob(jobId);
        misses = 0;
        onUpdate?.(state);
        if (state.status === "done" || state.status === "failed") {
          return resolve(state);
        }
      } catch (e) {
        if (signal?.aborted) return resolve(null);
        // A job that 404s is gone for good; anything else may be transient.
        if (e.status === 404) return reject(e);
        if (++misses >= 5) return reject(e);
      }
      setTimeout(tick, intervalMs);
    };

    tick();
  });
}

// ---------------------------------------------------------------------------
// Small helpers the screens share
// ---------------------------------------------------------------------------

/** Build the ops array a run expects from a catalog plus a set of enabled
 *  names. Locked ops are never sent — the backend appends them itself. */
export function opsFromState(catalog, enabled, params = {}) {
  return catalog
    .filter((o) => !o.locked && enabled.has(o.op))
    .map((o) => ({ op: o.op, params: params[o.op] ?? {} }));
}

/** Default parameter values for one op, straight out of its schema. */
export function defaultParams(opSpec) {
  return Object.fromEntries(
    Object.entries(opSpec.params_schema ?? {}).map(([k, v]) => [k, v.default]),
  );
}

/** "8 features → 8 qubits → 256 dimensions". The backend builds the same
 *  string, but the slider needs it before the next request lands. */
export function qubitReadout(encoding, nFeatures) {
  const q =
    encoding.qubit_formula === "N"
      ? nFeatures
      : Math.max(1, Math.ceil(Math.log2(Math.max(nFeatures, 1))));
  return {
    qubits: q,
    hilbert: 2 ** q,
    text: `${nFeatures} features → ${q} qubit${q === 1 ? "" : "s"} → ${(
      2 ** q
    ).toLocaleString()} dimensions`,
  };
}
