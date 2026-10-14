import { useEffect, useMemo, useRef, useState } from "react";

import {
  ApiError,
  defaultParams,
  estimateRun,
  getCatalogs,
  getEncodings,
  opsFromState,
  previewPreprocess,
  startRun,
  validateRun,
} from "../api";
import BlochSpheres from "../components/BlochSpheres";
import Card from "../components/Card";
import CostStrip from "../components/CostStrip";
import EncodingPanel from "../components/EncodingPanel";
import ModelPicker from "../components/ModelPicker";
import OpList from "../components/OpList";
import PresetSelect from "../components/PresetSelect";
import PreviewStrip from "../components/PreviewStrip";
import QuantumCircuit from "../components/QuantumCircuit";
import Spinner from "../components/Spinner";
import ValidationBanner from "../components/ValidationBanner";

/**
 * Screen 2 of 5, and the one the project is really about.
 *
 * Everything on this page is built from the five catalog endpoints. There is
 * not one op name, model name, encoding name or backbone name written down in
 * this file — which is exactly why adding a Phase 2 model costs zero frontend
 * work.
 *
 * Three requests fire as the config changes, all debounced together:
 *   POST /runs/validate   errors disable Run
 *   POST /runs/estimate   the cost strip
 *   POST /preprocess/preview  the stage strip
 */

const DEBOUNCE_MS = 450;

export default function Configure({ dataset, onRun, onBack }) {
  const [catalogs, setCatalogs] = useState(null);
  const [loadError, setLoadError] = useState(null);

  const [preset, setPreset] = useState(null);
  const [dirty, setDirty] = useState(false);
  const [enabledOps, setEnabledOps] = useState(new Set());
  const [opParams, setOpParams] = useState({});

  const [backbone, setBackbone] = useState(null);
  const [encoding, setEncoding] = useState(null);
  const [nFeatures, setNFeatures] = useState(8);

  const [models, setModels] = useState(new Set());
  const [modelParams, setModelParams] = useState({});

  const [validation, setValidation] = useState(null);
  const [estimate, setEstimate] = useState(null);
  const [preview, setPreview] = useState(null);
  const [previewErr, setPreviewErr] = useState(null);
  const [checking, setChecking] = useState(false);
  const [estimateOff, setEstimateOff] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);
  const [circuit, setCircuit] = useState(null);

  // ---- load the catalogs, then seed every control from them ---------------
  useEffect(() => {
    let alive = true;
    getCatalogs()
      .then((c) => {
        if (!alive) return;
        setCatalogs(c);
        setBackbone(c.backbones[0]?.name ?? null);
        setEncoding(c.encodings[0]?.name ?? null);
        // "Everything classical, plus the fast quantum one" is the sensible
        // first race. VQC is opt-in because it costs about eighty seconds.
        const cheapQuantum = c.models.find((m) => m.kind === "quantum");
        setModels(
          new Set([
            ...c.models
              .filter((m) => m.kind === "classical")
              .map((m) => m.name),
            ...(cheapQuantum ? [cheapQuantum.name] : []),
          ]),
        );
        const second = c.presets[1] ?? c.presets[0];
        if (second) applyPreset(second, c.ops);
      })
      .catch(
        (e) =>
          alive && setLoadError(e instanceof ApiError ? e.detail : String(e)),
      );
    return () => {
      alive = false;
    };
  }, []);

  function applyPreset(p, ops) {
    setPreset(p.name);
    setDirty(false);
    setEnabledOps(new Set(p.ops.map((o) => o.op)));
    const params = {};
    for (const spec of ops) {
      const fromPreset = p.ops.find((o) => o.op === spec.op);
      params[spec.op] = {
        ...defaultParams(spec),
        ...(fromPreset?.params ?? {}),
      };
    }
    setOpParams(params);
  }

  // ---- the config object, rebuilt whenever anything changes ---------------
  const config = useMemo(() => {
    if (!catalogs || !dataset || !encoding || !backbone) return null;
    return {
      dataset_id: dataset.dataset_id,
      models: [...models],
      ops: opsFromState(catalogs.ops, enabledOps, opParams),
      preset_name: dirty ? null : preset,
      backbone,
      n_features: nFeatures,
      encoding,
      test_size: 0.2,
      seed: 42,
      model_params: modelParams,
    };
  }, [
    catalogs,
    dataset,
    models,
    enabledOps,
    opParams,
    preset,
    dirty,
    backbone,
    encoding,
    nFeatures,
    modelParams,
  ]);

  // ---- validate + estimate, debounced -------------------------------------
  useEffect(() => {
    if (!config) return;
    setChecking(true);
    const id = setTimeout(async () => {
      try {
        setValidation(await validateRun(config));
      } catch {
        setValidation(null);
      }
      try {
        setEstimate(await estimateRun(config));
        setEstimateOff(false);
      } catch (e) {
        // 503 means estimate.py is not wired; anything else is transient.
        if (e instanceof ApiError && e.status === 503) setEstimateOff(true);
        setEstimate(null);
      }
      setChecking(false);
    }, DEBOUNCE_MS);
    return () => clearTimeout(id);
  }, [config]);

  // ---- preview, debounced separately because it is ~200 KB ---------------
  const opsKey = JSON.stringify(config?.ops ?? []);
  const previewAbort = useRef(null);
  useEffect(() => {
    if (!dataset || !catalogs) return;
    const id = setTimeout(async () => {
      previewAbort.current?.abort();
      const ctrl = new AbortController();
      previewAbort.current = ctrl;
      try {
        setPreviewErr(null);
        setPreview(
          await previewPreprocess(
            dataset.dataset_id,
            JSON.parse(opsKey),
            0,
            ctrl.signal,
          ),
        );
      } catch (e) {
        if (e.name !== "AbortError") {
          setPreviewErr(e instanceof ApiError ? e.detail : "Preview failed.");
        }
      }
    }, DEBOUNCE_MS);
    return () => clearTimeout(id);
  }, [opsKey, dataset, catalogs]);

  // ---- the circuit for the current encoding + feature count ---------------
  // Separate from validate/estimate: it changes only with these two inputs, and
  // re-deriving a circuit on every op toggle would be wasted work.
  useEffect(() => {
    if (!encoding || !nFeatures) return;
    let alive = true;
    const id = setTimeout(() => {
      getEncodings(nFeatures)
        .then((list) => {
          if (alive) setCircuit(list.find((e) => e.name === encoding) ?? null);
        })
        .catch(() => alive && setCircuit(null));
    }, DEBOUNCE_MS);
    return () => {
      alive = false;
      clearTimeout(id);
    };
  }, [encoding, nFeatures]);

  // ---- keep the feature count inside the encoding's ceiling ---------------
  useEffect(() => {
    const enc = catalogs?.encodings.find((e) => e.name === encoding);
    if (enc && nFeatures > enc.max_features) setNFeatures(enc.max_features);
  }, [encoding, catalogs, nFeatures]);

  const blocked = (validation?.errors?.length ?? 0) > 0;

  const submit = async () => {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const { job_id } = await startRun(config);
      onRun?.(job_id, config);
    } catch (e) {
      setSubmitError(
        e instanceof ApiError ? e.detail : "Could not start the run.",
      );
      setSubmitting(false);
    }
  };

  if (loadError) {
    return (
      <div className="mx-auto max-w-[1440px] px-9 py-7">
        <p
          role="alert"
          className="rounded-lg border-l-[3px] border-alert bg-alert/5 px-4 py-3 text-[12px]"
        >
          <span className="font-semibold text-ink">
            Could not load the catalogs.
          </span>{" "}
          {loadError}
        </p>
      </div>
    );
  }

  if (!catalogs || !config) {
    return (
      <p className="flex items-center gap-2 px-9 py-16 text-[13px] text-muted">
        <Spinner size={16} /> Loading what this backend can do
      </p>
    );
  }

  return (
    <div className="mx-auto max-w-[1440px] px-9 py-7">
      <header>
        <h1 className="text-[25px] font-semibold text-ink">
          Configure the run
        </h1>
        <p className="mt-1 text-[13px] text-body">
          Four choices. Each one changes what the models see.
        </p>
      </header>

      <div className="mt-6 grid gap-6 xl:grid-cols-3">
        {/* ---- column 1: preprocessing ---- */}
        <div className="space-y-6">
          <Card
            title="Clean up the images"
            sub="Fixed order. Toggle on or off, tune where shown."
          >
            <PresetSelect
              presets={catalogs.presets}
              value={preset}
              dirty={dirty}
              onChange={(name) =>
                applyPreset(
                  catalogs.presets.find((p) => p.name === name),
                  catalogs.ops,
                )
              }
            />
            <div className="mt-3">
              <OpList
                catalog={catalogs.ops}
                enabled={enabledOps}
                params={opParams}
                onToggle={(op, on) => {
                  setDirty(true);
                  setEnabledOps((prev) => {
                    const next = new Set(prev);
                    on ? next.add(op) : next.delete(op);
                    return next;
                  });
                }}
                onParam={(op, k, v) => {
                  setDirty(true);
                  setOpParams((p) => ({ ...p, [op]: { ...p[op], [k]: v } }));
                }}
              />
            </div>
          </Card>

          <Card
            title="What the models will see"
            sub="Redrawn on every change, from one image."
          >
            <PreviewStrip
              preview={preview}
              loading={checking}
              error={previewErr}
            />
          </Card>
        </div>

        {/* ---- column 2: features and encoding ---- */}
        <div className="space-y-6">
          <Card
            title="Turn images into numbers"
            sub="A pretrained network, then PCA down to a few features."
          >
            <EncodingPanel
              backbones={catalogs.backbones}
              encodings={catalogs.encodings}
              backbone={backbone}
              encoding={encoding}
              nFeatures={nFeatures}
              onBackbone={setBackbone}
              onEncoding={setEncoding}
              onFeatures={setNFeatures}
            />
          </Card>

          {/* Column 2 is the encoding column and was the short one — this
              lands under the panel it explains and evens the three columns
              out rather than stretching any of them. */}
          <Card
            title="What the encoding does"
            sub="One feature, swept from its smallest value to its largest."
          >
            <BlochSpheres
              angles={circuit?.bloch_demo}
              encodingLabel={circuit?.name ?? encoding}
              columns={Math.min(circuit?.bloch_demo?.length ?? 4, 6)}
              missingHint={false}
              caption={
                <>
                  Each sphere is the same feature at a different point in its
                  range under{" "}
                  <span className="font-mono">{circuit?.name ?? encoding}</span>
                  . No patient is involved yet — this is the geometry the data
                  will land in.
                </>
              }
            />
          </Card>
        </div>

        {/* ---- column 3: circuit, models, cost, validation ---- */}
        <div className="space-y-6">
          <Card
            title="The circuit that will run"
            sub="Redrawn whenever the encoding or feature count changes."
          >
            <QuantumCircuit
              qubits={circuit?.qubits}
              ops={circuit?.ops}
              depth={circuit?.depth}
              twoQubitGates={circuit?.two_qubit_gates}
              truncated={circuit?.ops_truncated}
              encodingLabel={circuit?.label ?? encoding}
              zoomable
            />
          </Card>

          <Card
            title="Pick what to race"
            sub="Every model sees exactly the same features."
          >
            <ModelPicker
              catalog={catalogs.models}
              selected={models}
              params={modelParams}
              onToggle={(name, on) =>
                setModels((prev) => {
                  const next = new Set(prev);
                  on ? next.add(name) : next.delete(name);
                  return next;
                })
              }
              onParam={(m, k, v) =>
                setModelParams((p) => ({ ...p, [m]: { ...p[m], [k]: v } }))
              }
            />
          </Card>

          <CostStrip
            estimate={estimate}
            loading={checking}
            unavailable={estimateOff}
          />

          <ValidationBanner
            errors={validation?.errors ?? []}
            warnings={validation?.warnings ?? []}
          />

          {submitError && (
            <p
              role="alert"
              className="rounded-lg border-l-[3px] border-alert bg-alert/5 px-4 py-3 text-[12px] text-body"
            >
              {submitError}
            </p>
          )}
        </div>
      </div>

      <footer className="mt-6 flex items-center justify-end gap-3">
        <button
          type="button"
          onClick={onBack}
          className="rounded-[7px] border border-rule bg-white px-5 py-2.5 text-[13px] text-body hover:bg-canvas"
        >
          Back
        </button>
        <button
          type="button"
          disabled={blocked || submitting || models.size === 0}
          onClick={submit}
          className="flex items-center gap-2 rounded-[7px] bg-quantum px-6 py-2.5 text-[13px]
            font-semibold text-white disabled:cursor-not-allowed disabled:opacity-40"
        >
          {submitting && <Spinner size={14} />}
          Run benchmark
        </button>
      </footer>
    </div>
  );
}
