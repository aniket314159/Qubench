import { useEffect, useState } from "react";

import { getHealth } from "./api";
import StepBar from "./components/StepBar";
import TopBar from "./components/TopBar";
import Benchmark from "./pages/Benchmark";
import Configure from "./pages/Configure";
import Diagnose from "./pages/Diagnose";
import Leaderboard from "./pages/Leaderboard";
import Upload from "./pages/Upload";

/**
 * The whole app.
 *
 * NO ROUTER. This is a five-step wizard with a step bar, and the step bar is
 * the navigation — adding react-router would mean two sources of truth for
 * "where am I" and buy nothing the design asks for. If deep links to a finished
 * run are wanted later, the swap is small: `step` becomes a route param and
 * this component keeps the rest of its state.
 *
 * Three pieces of state cross screens, and nothing else does:
 *
 *   dataset  chosen on Upload, needed by Configure and the run
 *   jobId    returned by Configure, polled by Benchmark
 *   runId    set when a job finishes, read by Benchmark and Diagnose
 *
 * Each page owns everything else. A page never reaches into another page's
 * state, which is why they can be built in parallel by different people.
 */

const STEPS = ["upload", "configure", "benchmark", "diagnose", "leaderboard"];
const LABELS = {
  upload: "Upload",
  configure: "Configure",
  benchmark: "Benchmark",
  diagnose: "Diagnose",
  leaderboard: "Leaderboard",
};

export default function App() {
  const [step, setStep] = useState("upload");
  const [dataset, setDataset] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [runId, setRunId] = useState(null);
  // The config and cost estimate that produced the current job. Benchmark
  // needs them to describe the circuit while it is still running — afterwards
  // the run record carries everything.
  const [runConfig, setRunConfig] = useState(null);
  const [runEstimate, setRunEstimate] = useState(null);
  const [wired, setWired] = useState(null);

  // One health check at start. It tells the top bar which endpoints are live,
  // so a control whose backend half is unwritten can be hidden rather than
  // letting someone click it and collect a 503.
  useEffect(() => {
    getHealth()
      .then((h) => setWired(h.wired))
      .catch(() => setWired(null));
  }, []);

  const enabled = {
    upload: true,
    configure: !!dataset,
    benchmark: !!jobId || !!runId,
    diagnose: !!runId,
    leaderboard: true,
  };

  const steps = STEPS.map((id, i) => ({
    id,
    label: LABELS[id],
    enabled: enabled[id],
    done: STEPS.indexOf(step) > i && enabled[id],
  }));

  const caption =
    step === "leaderboard"
      ? "all runs"
      : runId
        ? `run ${runId}`
        : jobId
          ? `job ${jobId}`
          : dataset
            ? "draft run"
            : "no run yet";

  return (
    <div className="min-h-screen bg-canvas font-sans text-body">
      <TopBar dataset={dataset} wired={wired} />
      <StepBar steps={steps} current={step} onGo={setStep} right={caption} />

      <main>
        {step === "upload" && (
          <Upload
            onContinue={(d) => {
              setDataset(d);
              // A new dataset invalidates any run in progress — carrying a
              // stale runId into Diagnose would predict with a model trained
              // on different images.
              setJobId(null);
              setRunId(null);
              setStep("configure");
            }}
          />
        )}

        {step === "configure" && (
          <Configure
            dataset={dataset}
            onBack={() => setStep("upload")}
            onRun={(id, cfg, est) => {
              setJobId(id);
              setRunConfig(cfg);
              setRunEstimate(est);
              setRunId(null);
              setStep("benchmark");
            }}
          />
        )}

        {step === "benchmark" && (
          <Benchmark
            jobId={jobId}
            runId={runId}
            config={runConfig}
            estimate={runEstimate}
            onRunFinished={setRunId}
            onDiagnose={(rid) => {
              setRunId(rid);
              setStep("diagnose");
            }}
            onBack={() => setStep("configure")}
          />
        )}

        {step === "diagnose" && (
          <Diagnose runId={runId} onBack={() => setStep("benchmark")} />
        )}

        {step === "leaderboard" && (
          <Leaderboard
            onOpenRun={(rid) => {
              // Drop the draft config and its estimate: they belong to a
              // different run, and showing one run's simulation count beside
              // another's results is worse than showing none.
              setRunConfig(null);
              setRunEstimate(null);
              setRunId(rid);
              setStep("benchmark");
            }}
          />
        )}
      </main>
    </div>
  );
}
