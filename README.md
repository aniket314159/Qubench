# QuBench

**Hybrid Quantum Machine Learning Platform for Early Disease Detection**
Qiskit Fall Fest 2026 · PS-Q1

A web platform that benchmarks **classical ML against quantum ML on identical
features**, for cardiovascular disease detection from ECG images.

The claim under test is not "quantum is better". It is: _given the same images,
the same preprocessing, the same embedding and the same principal components,
does a quantum kernel separate the classes better than an RBF kernel?_ Every
number the platform reports is a controlled comparison, and the classical arm
is grid-searched before every one of them.

---

## Table of contents

- [QuBench](#qubench)
  - [Table of contents](#table-of-contents)
  - [Quickstart — macOS / Linux](#quickstart--macos--linux)
  - [Quickstart — Windows](#quickstart--windows)
    - [Five differences that bite](#five-differences-that-bite)
    - [Two things that are the same](#two-things-that-are-the-same)
    - [Resetting storage](#resetting-storage)
  - [System architecture](#system-architecture)
    - [The run lifecycle](#the-run-lifecycle)
    - [Quantum framework](#quantum-framework)
  - [The invariant](#the-invariant)
  - [The six seams](#the-six-seams)
  - [What ships today](#what-ships-today)
  - [The five screens](#the-five-screens)
  - [API](#api)
    - [The contract that holds it together](#the-contract-that-holds-it-together)
  - [Storage](#storage)
  - [Results so far](#results-so-far)
  - [Testing](#testing)
  - [Extending it](#extending-it)
  - [Troubleshooting](#troubleshooting)
  - [What is not built](#what-is-not-built)
  - [Layout](#layout)

---

## Quickstart — macOS / Linux

Requires **Python 3.11+** (PennyLane 0.45 will not install on 3.10) and
**Node 18+**. No GPU, no database, no Docker, no quantum hardware.

```bash
git clone <repo> && cd sih-draft

# --- backend -------------------------------------------------------------
python3 -m venv .venv
source .venv/bin/activate

# CPU torch FIRST. The default PyPI wheel pulls ~2.5 GB of CUDA packages this
# project never executes.
pip install torch==2.14.0 torchvision==0.29.0 \
    --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt

# --- frontend ------------------------------------------------------------
cd frontend && npm install && cd ..

# --- run both ------------------------------------------------------------
./run.sh all
```

|               |                                |
| ------------- | ------------------------------ |
| App           | <http://localhost:5173>        |
| API docs      | <http://localhost:8000/docs>   |
| What is wired | <http://localhost:8000/health> |

`./run.sh api` and `./run.sh web` start one half each.
`cd frontend && VITE_MOCK=true npm run dev` runs the whole UI from generated
fixtures with no Python at all.

---

## Quickstart — Windows

Same requirements: **Python 3.11+** and **Node 18+**. Use **PowerShell**, not
`cmd` — the activation script and `run.ps1` both assume it.

```powershell
git clone <repo>
cd sih-draft

# --- backend -------------------------------------------------------------
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# CPU torch FIRST. The default PyPI wheel pulls ~2.5 GB of CUDA packages this
# project never executes.
pip install torch==2.14.0 torchvision==0.29.0 `
    --index-url https://download.pytorch.org/whl/cpu
pip install -r backend\requirements.txt

# --- frontend ------------------------------------------------------------
cd frontend; npm install; cd ..

# --- run both (opens two PowerShell windows) -----------------------------
.\run.ps1 all
```

`run.sh` is bash and will not run here; **`run.ps1` is the Windows
equivalent** and takes the same arguments (`api`, `web`, `all`).

### Five differences that bite

**PowerShell blocks the activation script by default.** If
`Activate.ps1` fails with "running scripts is disabled", allow it for your user
once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**`python`, not `python3`.** The `python3` alias usually does not exist, and a
bare `python` may open the Microsoft Store instead — if it does, install
Python from python.org and tick _Add python.exe to PATH_.

**`curl` is not curl.** In PowerShell `curl` is an alias for
`Invoke-WebRequest`, which takes different flags. Use `curl.exe` explicitly:

```powershell
curl.exe -s localhost:8000/health
curl.exe -s -F "file=@C:\path\to\ecg.zip" localhost:8000/datasets
```

**Environment variables use `$env:`.** The dataset-id exports in this README
become:

```powershell
$env:BAL = (python -c "from backend import store; print(next(d.dataset_id for d in store.list_datasets() if 'balanced' in d.name.lower()))")
python -m scripts.pipeline --dataset-id $env:BAL --backbone resnet18 --features 8
```

And mock mode, which has no `VITE_MOCK=true npm run dev` prefix syntax:

```powershell
cd frontend
$env:VITE_MOCK = "true"; npm run dev
```

**Backslashes in paths, forward slashes in URLs.** `backend\requirements.txt`
on the command line; `/api/datasets` unchanged in the app.

### Two things that are the same

`storage/` paths are resolved by `core/paths.py` from the file's own location,
so they work unchanged. And the Vite proxy must still target the literal
`http://127.0.0.1:8000` rather than `localhost` — Node prefers `::1` on
Windows too, and uvicorn binds IPv4.

### Resetting storage

```powershell
Remove-Item -Recurse -Force storage
python backend\core\paths.py
```

---

## System architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  FRONTEND — React · Vite · Tailwind v4                                   │
│                                                                          │
│   Upload ─► Configure ─► Benchmark ─► Diagnose        Leaderboard        │
│                                                                          │
│   api.js  ← the ONLY file that calls fetch(). Everything else takes props│
│   No model, op, encoding, backbone or class name appears in any .jsx     │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │  REST/JSON · 800 ms job polling
┌───────────────────────────────┴──────────────────────────────────────────┐
│  API LAYER — FastAPI                                                     │
│   main.py      17 routes, zero ML logic, every body under ~15 lines      │
│   schemas.py   Pydantic mirrors of the dataclasses + a drift guard       │
│   validate.py  15 error rules, 8 warning rules                           │
│   jobs.py      SEAM 5 · background worker, one run at a time             │
│   store.py     SEAM 6 · six functions over JSON files                    │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │  plain Python calls · no HTTP below here
┌───────────────────────────────┴──────────────────────────────────────────┐
│  CORE — pure Python. `import fastapi` appears nowhere below this line.    │
│                                                                          │
│  ┌── data in ──────────────┐   ┌── features out ──────────────────────┐  │
│  │ ingest.py     SEAM 1    │   │ embed.py      SEAM 3   4 backbones   │  │
│  │ preprocess.py SEAM 2    │   │ encodings.py  SEAM 3.5 4 encodings   │  │
│  │                         │   │ project.py    split → PCA → scaler    │  │
│  │  ── HANDOFF ───────────►│   │ registry.py   SEAM 4   4 models      │  │
│  │  uint8[n,224,224,3]     │   │ models_classical · models_quantum    │  │
│  │  y:int[n], class_names  │   │ benchmark.py  orchestrator           │  │
│  │                         │   │ estimate.py   cost model             │  │
│  └─────────────────────────┘   │ checkpoint.py bundle save/load       │  │
│                                └──────────────────────────────────────┘  │
│                                                                          │
│    ══════════════ THE INVARIANT ═══════════════════════════════          │
│    X : float32[n, N]   y : int[n]   class_names : list[str]              │
│    Nothing else crosses into the model layer. Ever.                      │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
┌───────────────────────────────┴──────────────────────────────────────────┐
│  STORAGE — flat files, no database                                       │
│  storage/datasets/{ds_id}/{class}/*      ingested images                 │
│  storage/artifacts/{ds_id}__{backbone}__{ops_hash}.npy   embedding cache │
│  storage/checkpoints/{run_id}__{model}.pkl               trained bundles │
│  storage/datasets.json · runs.json · calibration.json                    │
└──────────────────────────────────────────────────────────────────────────┘
```

### The run lifecycle

```
POST /datasets   zip → classes inferred from folder names → stratified
                 subsample to ≤1000 (floor of 10/class) → 4 preview thumbs

Configure        five catalog endpoints drive every control on the screen
                 validate + estimate + preview refire on every change

POST /runs       validated again server-side, then a background job:
                   5%  ingest      load images from disk
                  20%  preprocess  apply cfg.ops per image
                  ──── HANDOFF ────────────────────────────
                  55%  embed       backbone forward, or cache hit
                  60%  project     split → PCA(N) → encoding scaler
               60-95%  train       each model: fit, predict, metrics,
                                   telemetry, checkpoint
                 100%  save        store.save_run(record)

GET /jobs/{id}   polled every 800 ms; status, pct, message
GET /runs/{id}   metrics, confusion matrices, per-class, telemetry
POST /predict    one image, replaying the checkpoint's OWN recipe
```

### Quantum framework

**PennyLane only. No Qiskit anywhere.** One framework under time pressure;
`qml.specs(level="device")` gives real depth and gate counts in a single call;
`diff_method="backprop"` on `default.qubit` differentiates the simulator
directly, so the variational model needs no deep-learning framework at all.
Torch is confined to `embed.py`, where a pretrained CNN genuinely requires it.

Two implementation choices worth knowing:

**The Gram matrix is built from cached statevectors.** Simulating each sample
once and taking inner products is O(n) simulations plus one BLAS call, rather
than the O(n²) compute-uncompute circuits a pairwise fidelity kernel would
need. Identical numbers, minutes into seconds.

**The kernel bandwidth is cross-validated, not assumed.** Fidelity kernels
concentrate: too wide and every state is orthogonal to every other, too narrow
and they are all identical. Measured on our data, `zz` scores **0.774 untuned
and 0.903 tuned** — the largest single effect in the project.

Simulation is exact statevector — noiseless, no shot noise, no hardware queue.
The interface says so and never claims otherwise.

---

## The invariant

```
ingest → preprocess → embed → project → encode → train → evaluate
```

Everything upstream of the model layer collapses to exactly:

```
X : float32[n, N]     y : int[n]     class_names : list[str]
```

Models never see images, file paths, or a `DatasetMeta`. `n_classes` is always
a constructor argument, and `BaseModel.fit()` raises if any label falls outside
`[0, n_classes)` — so a hardcoded `4` dies loudly the first time someone
uploads a two-class dataset.

PCA and every scaler are fitted on the **training split only**. `project.py`
does the split first and the fit second, in one function, so it stays that way.

---

## The six seams

Each is a plain Python dict. Adding an implementation means writing a function
and adding one dict entry. Nothing else changes.

| #   | Seam             | File                 | Ships today                                                                   |
| --- | ---------------- | -------------------- | ----------------------------------------------------------------------------- |
| 1   | Ingest adapter   | `core/ingest.py`     | `image_folder`                                                                |
| 2   | Preprocessing op | `core/preprocess.py` | 9 ops + 2 locked, 4 presets                                                   |
| 3   | Backbone         | `core/embed.py`      | `resnet18` (512) · `resnet50` (2048) · `resnet50_pool1` (64) · `pixels` (256) |
| 3.5 | Encoding         | `core/encodings.py`  | `angle_y` · `angle_x2` · `zz` · `amplitude`                                   |
| 4   | Model registry   | `core/registry.py`   | `svc` · `rf` · `qsvc` · `vqc`                                                 |
| 5   | Job runner       | `backend/jobs.py`    | BackgroundTasks + dict                                                        |
| 6   | Store            | `backend/store.py`   | JSON files                                                                    |

**Phase 2 touches only seam 4. Phase 3 touches only seams 1, 2, 3, 5, 6.**
That is the pitch, and the seams are how it is backed up.

---

## What ships today

**Preprocessing** — 11 ops in a fixed execution order:

```
100 crop_border   200 deskew    300 remove_grid   400 grayscale
500 clahe         600 denoise   700 binarize      800 remove_lead_lines
850 invert        900 to_rgb (locked)             1000 resize (locked)
```

Order is a property of the op, not of the config. `remove_grid` identifies ECG
paper by **colour saturation**, so it must run before `grayscale` or it
silently does nothing. `apply_ops` sorts by order regardless of how the list
arrives. Presets: `raw` · `ecg_clean_scan` · `ecg_grid_paper` · `ecg_photo`.

**Encodings** — `angle_y` (N qubits, max 12), `angle_x2` (ring-entangled,
max 12), `zz` (max 10), `amplitude` (⌈log₂N⌉ qubits, max 64). Measured at 8
features: `angle_y` is depth 1 with **zero** entangling gates; `zz` is depth 41
with 56.

**Models** — SVC (26-candidate grid search by default), Random Forest,
Quantum Kernel SVC (bandwidth + C cross-validated), Variational Quantum
Classifier. Telemetry uses **one identical key set** for both kinds, quantum
fields nullable, so a single table renders everything.

---

## The five screens

**Upload** — dropzone, detected classes with counts, four real thumbnails from
one response, an automatic imbalance warning at 3:1 or worse, and a list of
datasets already ingested on the machine.

**Configure** — the centrepiece. Preprocessing ops with live parameter
controls, a live preview strip of every stage, the backbone selector with
variance-retained readout, the feature slider (capped by the selected
encoding), the encoding picker, **the real decomposed quantum circuit**
(click to zoom), a Bloch fan showing what the encoding does to a feature across
its range, the model picker, the cost strip, and the validation banner.

**Benchmark** — progress ladder, an activity transcript, live circuit
telemetry, and a Bloch fan while the job runs; then the results: macro-F1 hero
with the quantum/classical delta, the metrics table, accuracy and log-scale
time charts, both confusion matrices side by side, per-class recall, and the
telemetry table.

**Diagnose** — one image against one trained model, per-class confidence, the
patient's own Bloch angles, and a "not a medical device" strip.

**Leaderboard** — best classical vs best quantum per dataset, across every run.

Everything on these screens is rendered from catalog endpoints. **No model, op,
encoding, backbone or class name is written in any `.jsx` file** — that is what
makes the Phase 2 claim true.

---

## API

17 routes. `GET /health` reports which model-layer symbols are wired.

| Method | Path                        | Purpose                                             |
| ------ | --------------------------- | --------------------------------------------------- |
| GET    | `/health`                   | liveness + which symbols are wired                  |
| POST   | `/datasets`                 | multipart zip → `DatasetMeta` + 4 preview data URIs |
| GET    | `/datasets`                 | list, newest first                                  |
| GET    | `/ops/catalog`              | 11 ops with `params_schema`                         |
| GET    | `/presets`                  | 4 presets                                           |
| GET    | `/backbones`                | 4 backbones                                         |
| GET    | `/encodings`                | 4 encodings; `?n_features=N` adds the real circuit  |
| GET    | `/models/catalog`           | 4 models, classical first                           |
| POST   | `/preprocess/preview`       | one image, every stage, as data URIs                |
| POST   | `/runs/validate`            | `{errors, warnings}` — errors disable Run           |
| POST   | `/runs/estimate`            | qubits, Hilbert dim, simulations, MB, seconds       |
| POST   | `/runs`                     | `{job_id}`                                          |
| GET    | `/jobs/{job_id}`            | status, pct, message, run_id, error                 |
| GET    | `/runs`                     | run summaries                                       |
| GET    | `/runs/{run_id}`            | full record                                         |
| GET    | `/leaderboard`              | best classical vs quantum per dataset               |
| POST   | `/predict/{run_id}/{model}` | label, confidences, latency, Bloch angles           |

### The contract that holds it together

`core/contracts.py` is frozen. It defines every dataclass, the `BaseModel` ABC,
the embedding cache key and the telemetry key set. `backend/schemas.py` mirrors
it in Pydantic and its self-test **fails if the two drift** — add a field to a
dataclass and forget the schema, and `python -m backend.schemas` exits 1.

---

## Storage

```
storage/
├── datasets.json                                    list[DatasetMeta]
├── runs.json                                        list[RunRecord]
├── calibration.json                                 machine-specific cost fit
├── datasets/{ds_id}/{class_name}/*                  ingested images
├── datasets/{ds_id}/_preview/*.png                  4 thumbnails
├── artifacts/{ds_id}__{backbone}__{ops_hash}.npy    embedding cache
└── checkpoints/{run_id}__{model}.pkl                CheckpointBundle
```

Paths come from `core/paths.py` — never write `Path("storage")` by hand, since
the API and `scripts/pipeline.py` have different working directories.

**The cache key is deliberate.** Changing `n_features` or `encoding` **hits**
the cache; changing `ops` recomputes. Users move the feature slider constantly
and touch the op list rarely. `RunConfig.embed_key()` computes it in one place
so the API and the CLI cannot disagree.

**Checkpoints carry their own recipe** — ops, backbone, encoding, n*features,
the fitted PCA and scaler, alongside the weights. Diagnose replays \_that*,
never the live config. Otherwise a user who moved a slider after the run would
get confidently wrong predictions with nothing raising.

Reset everything:

```bash
rm -rf storage/ && python backend/core/paths.py
```

---

## Results so far

Measured on the Ch. Pervaiz Elahi Institute of Cardiology ECG image dataset
(Khan, Hussain & Malik, _Data in Brief_, 2021), balanced to 74 images per class
across Normal / abnormal heartbeat / MI / history of MI.

| backbone · encoding · features   | best classical | best quantum |      delta |
| -------------------------------- | -------------: | -----------: | ---------: |
| `resnet50_pool1` · `angle_y` · 8 |          0.577 |    **0.608** | **+0.031** |
| `resnet18` · `angle_y` · 8       |          0.532 |        0.533 |     +0.001 |
| `resnet18` · `zz` · 8            |          0.517 |        0.517 |      0.000 |
| `resnet18` · `amplitude` · 16    |          0.531 |        0.494 |     −0.037 |

Macro-F1, 80/20 stratified, seed 42.

**Read these honestly.** Quantum leads on one configuration of four, by three
points, on 296 images. Published double-digit gaps on this task compare against
an untuned baseline; ours does not. The full dataset is 11:1 imbalanced and
scores 0.760 _accuracy_ at 0.484 macro-F1 — which is why macro-F1 is the
headline everywhere in this product.

The 928-image figure quoted in the QuCardio paper is **not reproducible** from
the published release: it reports 240 MI images where 77 exist.

**Timing** at 742 training samples: SVC grid-searched 6.6 s · Random Forest
0.6 s · QSVC tuned 0.9 s · VQC ~30 s at 15 epochs. ResNet18 embeds at ~23
ms/image; preprocessing dominates a cold run.

---

## Testing

```bash
python scripts/selftest.py           # ~20 s: imports, invariants, 13 self-tests,
                                     # all 17 routes in-process, fixtures
python scripts/selftest.py --full    # adds the model × encoding × class matrix
python scripts/status.py --test      # file inventory + every self-test
```

Every module self-tests. `python -m backend.core.preprocess` asserts the
colour-order trap; `python -m backend.core.project` proves PCA saw only the
training rows; `python -m backend.schemas` is the drift guard.

Three architecture greps, all of which must print nothing:

```bash
grep -rn "fastapi\|pydantic" backend/core/
grep -rn "qiskit" backend/
grep -rn "fetch(" frontend/src/pages frontend/src/components
```

Other tooling:

```bash
python scripts/gen_fixtures.py       # 17 fixtures, regenerated from live code
python -m scripts.test_models        # every model × encoding × {2,4} classes
python -m scripts.calibrate --dataset-id $DS --backbone resnet18
python -m scripts.pipeline --dataset-id $DS --backbone resnet18 \
       --features 8 --encoding zz --models svc,rf,qsvc
```

`scripts/pipeline.py` runs the entire pipeline with **no web layer at all** —
if the table prints there, it will print through the API.

---

## Extending it

| Want to add          | Touch                                     | Frontend work |
| -------------------- | ----------------------------------------- | ------------- |
| A model (Phase 2)    | `models_*.py` + one line in `registry.py` | **none**      |
| A preprocessing op   | `preprocess.py` `OP_REGISTRY`             | **none**      |
| An encoding          | `encodings.py` `ENCODINGS`                | **none**      |
| A backbone           | `embed.py` `BACKBONES`                    | **none**      |
| A modality (Phase 3) | `ingest.py` `ADAPTERS`                    | **none**      |

"None" holds because `ParamControl` renders every control from the
`params_schema` the backend ships, and the quantum circuit is drawn from the
**real decomposed op list** rather than a hand-written description. A new
encoding draws itself correctly the day it lands.

After any change to `contracts.py`, `schemas.py` or a registry:

```bash
python scripts/gen_fixtures.py && python scripts/selftest.py
```

---

## Troubleshooting

**`AggregateError [ECONNREFUSED]` in the Vite proxy** — Node resolves
`localhost` to `::1` first and uvicorn binds `127.0.0.1`. The proxy target in
`frontend/vite.config.js` must be the literal `http://127.0.0.1:8000`.

**`ModuleNotFoundError: No module named 'backend'`** — run from the repo root.
`No module named 'backend.core.X'` means a **file** is missing;
`No module named 'cv2'` means a **package** is missing.

**`Form data requires "python-multipart"`** — stale venv; re-install
`backend/requirements.txt`.

**Uploads succeed but the list is empty** — read the error the Upload screen
shows. An empty list and a failed request mean opposite things.

**A job sits at 60%** — training, usually VQC (~80 s at 40 epochs versus under
2 s for QSVC). **There is no cancel**: a Python worker thread cannot be killed
from outside, which is exactly why the cost estimate appears _before_ the Run
button.

**A quantum model scores near chance** — check the bandwidth before anything
else. QSVC cross-validates it; VQC inherits whatever it is given, and VQC on
`zz` at bandwidth 1.0 is the known bad combination.

**Reloads are slow** — the venv is inside `backend/`. Move it to the repo root
so `--reload-dir backend` excludes it.

---

## What is not built

Honest scope for the internal round:

- **Phase 3 modalities** (X-ray, MRI) — one ingest adapter exists, and only ECG
  has been tested.
- **Celery / SQLite** — seam 5 is a dict, seam 6 is JSON files, by design.
- **Grad-CAM, kernel-matrix preview, similarity histogram** — designed, not
  implemented; each needs a new backend field.
- **Auth, multi-user, deployment** — out of Phase 1 scope entirely.
- **A second dataset.** One dataset cannot separate a real effect from one
  dataset's quirks, and this is the most important gap in the result above.

---

## Layout

```
sih-draft/
├── README.md · run.sh · run.ps1 · .gitignore
├── backend/
│   ├── main.py · schemas.py · store.py · jobs.py · validate.py
│   ├── requirements.txt · requirements.lock.txt
│   └── core/  paths · contracts · ingest · preprocess · embed · encodings
│              project · registry · models_classical · models_quantum
│              benchmark · estimate · checkpoint
├── frontend/
│   └── src/  api.js · mock.js · App.jsx · pages/ (5) · components/ (~29)
├── fixtures/     17 JSON, generated from live code
├── scripts/      selftest · status · gen_fixtures · pipeline · test_models · calibrate
└── storage/      gitignored except .gitkeep
```

**Team:** four workstreams — W1 contracts and integration, W2 frontend,
W3 backend, W4 model layer.
