import { useEffect, useState } from "react";

import { ApiError, getRun, listRuns, predict } from "../api";
import BlochSpheres from "../components/BlochSpheres";
import Card from "../components/Card";
import Dropzone from "../components/Dropzone";
import ModelSelect from "../components/ModelSelect";
import PredictionCard from "../components/PredictionCard";
import Spinner from "../components/Spinner";

/**
 * Screen 4 of 5. One image, one trained model.
 *
 * The checkpoint carries its own preprocessing recipe — the ops that trained
 * it, not whatever is currently set on the Configure screen. That matters: a
 * user who moved the feature slider after the run would otherwise get
 * predictions from a pipeline the model never saw, with nothing raising and
 * every probability still summing to one.
 */
export default function Diagnose({ runId, model: initialModel, onBack }) {
  const [runs, setRuns] = useState(null);
  const [chosenRun, setChosenRun] = useState(runId ?? null);
  const [run, setRun] = useState(null);
  const [model, setModel] = useState(initialModel ?? null);

  const [file, setFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    listRuns()
      .then(setRuns)
      .catch(() => setRuns([]));
  }, []);

  useEffect(() => {
    if (!chosenRun) return;
    setRun(null);
    getRun(chosenRun)
      .then((r) => {
        setRun(r);
        const usable = r.results.filter((m) => !m.error && m.checkpoint);
        if (!usable.some((m) => m.model === model)) {
          // Pick the strongest usable model rather than the first — the point
          // of this screen is to show the best thing the run produced.
          const best = [...usable].sort(
            (a, b) => b.metrics.f1_macro - a.metrics.f1_macro,
          )[0];
          setModel(best?.model ?? null);
        }
      })
      .catch((e) =>
        setError(e instanceof ApiError ? e.detail : "Could not load that run."),
      );
  }, [chosenRun]); // eslint-disable-line react-hooks/exhaustive-deps

  // Object URLs leak if they are not revoked, and this screen is used
  // repeatedly during a demo.
  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const onImage = async (f) => {
    setFile(f);
    setPrediction(null);
    setError(null);
    if (!chosenRun || !model) return;
    setBusy(true);
    try {
      setPrediction(await predict(chosenRun, model, f));
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : "Prediction failed.");
    } finally {
      setBusy(false);
    }
  };

  const rerun = () => file && onImage(file);
  const modelLabel =
    run?.results.find((r) => r.model === model)?.label ?? model;

  if (!chosenRun) {
    return (
      <div className="mx-auto max-w-[1440px] px-9 py-7">
        <h1 className="text-[25px] font-semibold text-ink">
          Diagnose a single ECG
        </h1>
        <p className="mt-1 text-[13px] text-body">
          Pick a finished run to use.
        </p>
        <Card className="mt-6 max-w-[640px]" title="Finished runs">
          {runs === null ? (
            <p className="flex items-center gap-2 text-[12px] text-muted">
              <Spinner size={14} /> Loading
            </p>
          ) : runs.length === 0 ? (
            <p className="text-[12px] text-muted">
              No runs yet. Configure and run a benchmark first.
            </p>
          ) : (
            <ul className="divide-y divide-hairline">
              {runs.map((r) => (
                <li
                  key={r.run_id}
                  className="flex items-center justify-between gap-4 py-3"
                >
                  <div>
                    <p className="text-[12.5px] text-ink">{r.dataset_name}</p>
                    <p className="font-mono text-[10.5px] text-muted">
                      {r.run_id} · {r.encoding} · {r.n_features} features
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setChosenRun(r.run_id)}
                    className="rounded-[7px] border border-rule px-3 py-1.5 text-[12px] text-body hover:bg-canvas"
                  >
                    Use this
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-[1440px] px-9 py-7">
      <header>
        <h1 className="text-[25px] font-semibold text-ink">
          Diagnose a single ECG
        </h1>
        <p className="mt-1 text-[13px] text-body">
          Decision support, not a diagnosis. A clinician reviews every result.
        </p>
      </header>

      <div className="mt-6 grid gap-6 lg:grid-cols-[1fr_400px]">
        <div className="space-y-6">
          {imageUrl ? (
            <Card title={file?.name} sub="Uploaded just now, never stored.">
              <img
                src={imageUrl}
                alt="The ECG being classified"
                className="max-h-[420px] w-full rounded-md border border-rule object-contain bg-white"
              />
              <div className="mt-3 flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => {
                    setFile(null);
                    setPrediction(null);
                  }}
                  className="rounded-[7px] border border-rule px-3 py-1.5 text-[12.5px] text-body hover:bg-canvas"
                >
                  Replace image
                </button>
                <button
                  type="button"
                  onClick={rerun}
                  disabled={busy}
                  className="rounded-[7px] border border-rule px-3 py-1.5 text-[12.5px] text-body
                    hover:bg-canvas disabled:opacity-50"
                >
                  Run again
                </button>
              </div>
            </Card>
          ) : (
            <Dropzone
              onFile={onImage}
              busy={busy}
              disabled={!model}
              accept="image/png,image/jpeg"
              match={/\.(png|jpe?g)$/i}
              title="Drop one ECG image here"
              busyTitle="Classifying"
              hint="A single PNG or JPG · preprocessed exactly as the training images were"
              busyHint="Replaying the checkpoint's own pipeline, then the model"
              rejectHint="this screen takes a single PNG or JPG."
            />
          )}

          {error && (
            <p
              role="alert"
              className="rounded-lg border-l-[3px] border-alert bg-alert/5 px-4 py-3 text-[12px] text-body"
            >
              {error}
            </p>
          )}

          <div className="rounded-lg border-l-[3px] border-caution bg-caution/10 px-4 py-3">
            <p className="text-[12px] font-semibold text-ink">
              Not a medical device
            </p>
            <p className="mt-1 text-[11.5px] text-body">
              A research prototype built for a hackathon, trained on a few
              hundred images. It is not a screening tool and must not be used to
              make a clinical decision.
            </p>
          </div>
        </div>

        <div className="space-y-6">
          <PredictionCard prediction={prediction} modelLabel={modelLabel} />

          {/* Rendered for ANY prediction. Gating on `bloch_angles !== undefined`
              hid the whole panel when the backend was older than the field or
              the fixture was stale — which looks identical to "the component is
              broken". Let BlochSpheres say which of the three cases it is. */}
          {prediction && (
            <Card
              title="This patient, as qubits"
              sub="Where their features land after the first encoding layer."
            >
              <BlochSpheres
                angles={prediction.bloch_angles}
                encodingLabel={run?.config?.encoding ?? ""}
                bandwidth={
                  run?.results.find((r) => r.model === model)?.telemetry
                    ?.bandwidth
                }
              />
            </Card>
          )}

          <Card
            title="Which model answers"
            sub="Any model from this run can be used."
          >
            {run ? (
              <>
                <ModelSelect
                  results={run.results}
                  value={model}
                  onChange={(m) => {
                    setModel(m);
                    setPrediction(null);
                  }}
                />
                <p className="mt-3 text-[11px] text-muted">
                  Each checkpoint stores the preprocessing it was trained with,
                  so switching model replays that pipeline — never whatever is
                  currently set on Configure.
                </p>
              </>
            ) : (
              <p className="flex items-center gap-2 text-[12px] text-muted">
                <Spinner size={14} /> Loading the run
              </p>
            )}
          </Card>

          <button
            type="button"
            onClick={onBack}
            className="w-full rounded-[7px] border border-rule bg-white px-4 py-2.5 text-[13px] text-body hover:bg-canvas"
          >
            Back to results
          </button>
        </div>
      </div>
    </div>
  );
}
