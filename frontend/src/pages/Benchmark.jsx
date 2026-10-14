import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, getEncodings, getRun, pollJob } from "../api";
import AccuracyChart from "../components/AccuracyChart";
import ActivityLog from "../components/ActivityLog";
import BlochSpheres from "../components/BlochSpheres";
import Card from "../components/Card";
import ConfusionMatrix from "../components/ConfusionMatrix";
import JobProgress from "../components/JobProgress";
import LiveTelemetry from "../components/LiveTelemetry";
import MetricsTable from "../components/MetricsTable";
import PerClassBars from "../components/PerClassBars";
import Spinner from "../components/Spinner";
import TelemetryTable from "../components/TelemetryTable";
import TimeChart from "../components/TimeChart";

/**
 * How many spheres per row.
 *
 * A grid docks a partial row to the left, so eight qubits at the caller's
 * default of six columns rendered as a lopsided 6 + 2. Pick the widest
 * arrangement that leaves the fewest empty cells instead — eight becomes two
 * even rows of four.
 */
function fanColumns(n) {
  const wanted = Math.min(n, 6);
  if (wanted < 3) return Math.max(wanted, 1);
  let best = wanted;
  let fewestEmpty = Infinity;
  for (let c = 3; c <= wanted; c++) {
    const empty = (c - (n % c)) % c;
    if (empty <= fewestEmpty) {
      fewestEmpty = empty;
      best = c;
    }
  }
  return best;
}

/**
 * Card framing, applied at the call site rather than in Card.jsx.
 *
 * FAN retunes the shared BlochSpheres for this one placement: it sizes its
 * spheres at a fixed 60px inside a grid whose cells shrink, so a narrow column
 * made them collide and a wide one left them adrift in their cells. Here they
 * fill the cell up to 64px, the labels step up to the design system's micro
 * size (10.5px), and the caption gets the same hairline divider every other
 * card footer in this app uses. CARD_CENTER keeps the panel at the progress
 * card's height and centres its contents rather than stranding them at the
 * top.
 */
const FAN =
  "[&_svg]:h-auto [&_svg]:w-full [&_svg]:max-w-[64px] " +
  "[&_li_p]:text-[10.5px] " +
  "[&>div>p]:mt-4 [&>div>p]:border-t [&>div>p]:border-hairline " +
  "[&>div>p]:pt-3 [&>div>p]:text-[11.5px]";

const CARD_CENTER =
  "flex flex-col [&>div:last-child]:flex [&>div:last-child]:flex-1 " +
  "[&>div:last-child]:items-center [&>div:last-child]:justify-center";

/**
 * Layout for the Activity card in the running view.
 *
 * It is its column's filler: the card stretches to the column bottom and the
 * log inside flexes to absorb whatever height the Bloch card above it did not
 * need — that is what turns the old dead space under Live telemetry into an
 * always-full page. [&>div]:flex lets the card's inner content div become the
 * flex row this needs (Card.jsx wraps children in a plain div), and
 * [&_ol]:min-h-0 lets the log shrink to scroll instead of forcing the card
 * taller. Below lg the columns stack and none of this has any effect.
 */
const ACTIVITY_FILL =
  "flex h-full flex-col [&>div]:flex [&>div]:flex-1 [&>div]:min-h-0 " +
  "[&_ol]:min-h-0 [&_ol]:flex-1";

/**
 * Screen 3 of 5. Two states in one page: running, then results.
 *
 * The transition is automatic — pollJob resolves on done, the run is fetched,
 * and the same component swaps what it renders. A separate "results" route
 * would mean a navigation the user did not ask for at the moment they are
 * watching a progress bar.
 */
function useElapsed(active) {
  const [s, setS] = useState(0);
  const start = useRef(Date.now());
  useEffect(() => {
    if (!active) return;
    start.current = Date.now();
    const id = setInterval(
      () => setS((Date.now() - start.current) / 1000),
      1000,
    );
    return () => clearInterval(id);
  }, [active]);
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}

export default function Benchmark({
  jobId,
  runId,
  config,
  estimate,
  onRunFinished,
  onDiagnose,
  onBack,
}) {
  const [job, setJob] = useState(null);
  const [run, setRun] = useState(null);
  const [error, setError] = useState(null);
  const [activity, setActivity] = useState([]);
  const [circuit, setCircuit] = useState(null);
  const elapsed = useElapsed(!!jobId && !runId);
  const started = useRef(Date.now());

  // The circuit being simulated.
  //
  // A finished run's own config wins over the draft that started it: arriving
  // here from the Leaderboard, or after a reload, there is no draft at all —
  // and a draft left over from a different run would draw the wrong circuit
  // with no sign that anything was amiss.
  const runCfg = run?.config ?? config ?? null;
  useEffect(() => {
    if (!runCfg?.encoding || !runCfg?.n_features) return;
    let alive = true;
    getEncodings(runCfg.n_features)
      .then((list) => {
        if (alive)
          setCircuit(list.find((e) => e.name === runCfg.encoding) ?? null);
      })
      .catch(() => alive && setCircuit(null));
    return () => {
      alive = false;
    };
  }, [runCfg?.encoding, runCfg?.n_features]);

  // Turn the stream of poll updates into a log. Appends only when the message
  // actually changes, so a stage that reports the same text for ten seconds
  // takes one line rather than a dozen.
  const record = (state) => {
    setJob(state);
    setActivity((prev) => {
      if (prev.length && prev[prev.length - 1].message === state.message)
        return prev;
      const s = Math.floor((Date.now() - started.current) / 1000);
      const at = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
      return [...prev, { at, message: state.message }];
    });
  };

  // ---- poll, then fetch ---------------------------------------------------
  useEffect(() => {
    if (!jobId || runId) return;
    started.current = Date.now();
    const ctrl = new AbortController();
    pollJob(jobId, record, { signal: ctrl.signal })
      .then((final) => {
        if (final?.status === "done" && final.run_id)
          onRunFinished?.(final.run_id);
      })
      .catch((e) =>
        setError(
          e instanceof ApiError ? e.detail : "Lost contact with the run.",
        ),
      );
    return () => ctrl.abort();
  }, [jobId, runId, onRunFinished]);

  useEffect(() => {
    if (!runId) return;
    getRun(runId)
      .then(setRun)
      .catch((e) =>
        setError(e instanceof ApiError ? e.detail : "Could not load the run."),
      );
  }, [runId]);

  // ---- the comparison the whole project is about --------------------------
  const summary = useMemo(() => {
    if (!run) return null;
    const ok = run.results.filter((r) => !r.error);
    const pick = (kind) =>
      ok
        .filter((r) => r.kind === kind)
        .sort((a, b) => b.metrics.f1_macro - a.metrics.f1_macro)[0] ?? null;
    const classical = pick("classical");
    const quantum = pick("quantum");
    const best =
      [classical, quantum]
        .filter(Boolean)
        .sort((a, b) => b.metrics.f1_macro - a.metrics.f1_macro)[0] ?? null;
    return {
      classical,
      quantum,
      best,
      delta:
        classical && quantum
          ? quantum.metrics.f1_macro - classical.metrics.f1_macro
          : null,
    };
  }, [run]);

  if (error) {
    return (
      <div className="mx-auto max-w-[1440px] px-9 py-7">
        <p
          role="alert"
          className="rounded-lg border-l-[3px] border-alert bg-alert/5 px-4 py-3 text-[12px] text-body"
        >
          <span className="font-semibold text-ink">Something went wrong.</span>{" "}
          {error}
        </p>
        <button
          type="button"
          onClick={onBack}
          className="mt-4 rounded-[7px] border border-rule bg-white px-4 py-2 text-[13px] text-body"
        >
          Back to Configure
        </button>
      </div>
    );
  }

  // ---- running ------------------------------------------------------------
  if (!run) {
    // The spheres describe the encoding, not the run, but they belong to the
    // wait: they are the geometry the models are working in while the bar
    // moves. `done` and `failed` both mean the answer exists and this view is
    // on its way out, so they are dropped at that point rather than lingering.
    const running = job?.status !== "done" && job?.status !== "failed";
    return (
      <div className="mx-auto max-w-[1440px] px-9 py-7">
        <header>
          <h1 className="text-[25px] font-semibold text-ink">
            Running the benchmark
          </h1>
          <p className="mt-1 text-[13px] text-body">
            Every model sees identical features. You can leave this page — the
            run continues.
          </p>
        </header>
        {/* One grid, bottom-aligned columns, so nothing on this page is
            stranded above dead space. Each column is a flex stack whose last
            card absorbs the leftover height. */}
        <div className="mt-6 grid items-stretch gap-6 lg:grid-cols-[minmax(0,760px)_minmax(0,1fr)]">
          <div className="flex flex-col gap-6">
            <div className="shrink-0">
              {job ? (
                <JobProgress job={job} elapsed={elapsed} estimate={estimate} />
              ) : (
                <p className="flex items-center gap-2 text-[13px] text-muted">
                  <Spinner size={15} /> Waiting for the first update
                </p>
              )}
            </div>

            <Card
              className="flex-1 [&>div]:flex [&>div]:flex-1 [&>div]:items-center"
              title="Live circuit telemetry"
              sub="The circuit these models are running on."
            >
              <LiveTelemetry circuit={circuit} estimate={estimate} />
            </Card>
          </div>

          <div className="flex flex-col gap-6">
            {/* Same panel as Configure, from the same encoding payload. While
                the job runs it is the only thing on this page that shows what
                the data looks like to the simulator. */}
            {running && (
              <Card
                className={`${CARD_CENTER} shrink-0`}
                title="What the encoding does"
                sub="One feature, swept from its smallest value to its largest."
              >
                <div className={`mx-auto w-full max-w-[420px] ${FAN}`}>
                  <BlochSpheres
                    angles={circuit?.bloch_demo}
                    encodingLabel={circuit?.name ?? runCfg?.encoding ?? ""}
                    columns={fanColumns(circuit?.bloch_demo?.length ?? 0)}
                    missingHint={false}
                    caption={
                      <>
                        Each sphere is the same feature at a different point in
                        its range under{" "}
                        <span className="font-mono">
                          {circuit?.name ?? runCfg?.encoding ?? ""}
                        </span>
                        . No patient is involved yet — this is the geometry the
                        data will land in.
                      </>
                    }
                  />
                </div>
              </Card>
            )}

            <Card
              className={ACTIVITY_FILL}
              title="Activity"
              sub="Sampled from the job every 800ms."
            >
              <ActivityLog entries={activity} />
            </Card>
          </div>
        </div>
      </div>
    );
  }

  // ---- results ------------------------------------------------------------
  const classNames = run.class_names ?? [];
  const showcase = summary?.best;

  return (
    <div className="mx-auto max-w-[1440px] px-9 py-7">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-[25px] font-semibold text-ink">Results</h1>
          <p className="mt-1 text-[13px] text-body">
            Same images, same features, same split. Only the model changed.
          </p>
        </div>
        <button
          type="button"
          disabled={!showcase}
          onClick={() => onDiagnose?.(run.run_id, showcase.model)}
          className="rounded-[7px] bg-quantum px-5 py-2.5 text-[13px] font-semibold text-white
            disabled:opacity-40"
        >
          Use in Diagnose
        </button>
      </header>

      {/* hero */}
      {summary && (
        <div
          className="mt-6 flex flex-wrap items-start gap-x-14 gap-y-6 rounded-[10px]
          border border-rule border-l-4 border-l-quantum bg-white px-6 py-5"
        >
          <div>
            <p className="text-[11px] text-muted">Best model</p>
            <p className="mt-1 text-[22px] font-semibold text-ink">
              {summary.best?.label ?? "—"}
            </p>
            <p className="mt-1 font-mono text-[11px] text-muted">
              {run.config.backbone} · {run.config.encoding} ·{" "}
              {run.config.n_features} features
            </p>
          </div>
          <div>
            <p className="text-[11px] text-muted">Macro-F1</p>
            <p className="mt-1 font-mono text-[34px] font-semibold leading-none text-quantum">
              {summary.quantum
                ? `${(summary.quantum.metrics.f1_macro * 100).toFixed(1)}%`
                : "—"}
            </p>
            <p className="mt-1.5 text-[10.5px] text-muted">best quantum</p>
          </div>
          <div>
            <p className="text-[11px] text-muted">Best classical</p>
            <p className="mt-1 font-mono text-[34px] font-semibold leading-none text-classical">
              {summary.classical
                ? `${(summary.classical.metrics.f1_macro * 100).toFixed(1)}%`
                : "—"}
            </p>
            <p className="mt-1.5 text-[10.5px] text-muted">
              {summary.classical?.label ?? ""}
            </p>
          </div>
          <div>
            <p className="text-[11px] text-muted">Difference</p>
            <p className="mt-1 font-mono text-[34px] font-semibold leading-none text-ink">
              {summary.delta == null
                ? "—"
                : `${summary.delta >= 0 ? "+" : "−"}${(Math.abs(summary.delta) * 100).toFixed(1)}%`}
            </p>
            <p className="mt-1.5 text-[10.5px] text-muted">
              {summary.delta == null
                ? "needs both kinds"
                : summary.delta >= 0
                  ? "in favour of quantum"
                  : "in favour of classical"}
            </p>
          </div>
          <p className="max-w-[300px] rounded-lg bg-hairline px-4 py-3 text-[11px] text-body">
            Macro-F1 weights every class equally. Accuracy would reward a model
            that ignored the rarest class, and on an uneven dataset that is
            exactly what happens.
          </p>
        </div>
      )}

      <div className="mt-6 grid gap-6 xl:grid-cols-2">
        <Card title="Every model, side by side">
          <MetricsTable results={run.results} />
        </Card>
        <Card title="How they compare" sub="Same axis, both lanes.">
          <div className="grid gap-8 sm:grid-cols-2">
            <AccuracyChart results={run.results} />
            <TimeChart results={run.results} />
          </div>
        </Card>
      </div>

      {showcase && (
        <div className="mt-6 grid gap-6 xl:grid-cols-2">
          {/* Both winners, side by side. The hero states the delta as one
              number; this pair is where its shape is visible — which class
              quantum recovered and which one it gave up. A kind that never
              ran gets a line of text rather than a blank panel. */}
          <Card
            title="Confusion matrices"
            sub="Rows are the true class, columns the prediction."
          >
            <div className="grid gap-6 sm:grid-cols-2">
              {[
                { kind: "quantum", tone: "text-quantum", result: summary.quantum },
                {
                  kind: "classical",
                  tone: "text-classical",
                  result: summary.classical,
                },
              ].map(({ kind, tone, result }) => (
                <div key={kind}>
                  <p className="text-[11.5px] text-muted">
                    Best {kind}
                    {result && (
                      <span className={`font-medium ${tone}`}>
                        {" · "}
                        {result.label}
                      </span>
                    )}
                  </p>
                  <div className="mt-2">
                    {result ? (
                      <ConfusionMatrix result={result} classNames={classNames} />
                    ) : (
                      <p className="text-[11.5px] text-muted">
                        No matrix for this model.
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </Card>
          <Card
            title="Per class"
            sub="Recall, and how many test images it had."
          >
            <PerClassBars
              perClass={showcase.per_class}
              classNames={classNames}
            />
          </Card>
        </div>
      )}

      <Card
        className="mt-6"
        title="Circuit telemetry"
        sub="One table, both kinds."
      >
        <TelemetryTable results={run.results} />
      </Card>

      <footer className="mt-6 flex items-center justify-between gap-6">
        <p className="max-w-[70ch] text-[11px] text-muted">
          The classical arm was grid-searched before this comparison, the split
          is fixed by seed, and both arms saw byte-identical features. A
          difference of a few points on a small test set is a lead worth
          chasing, not a result worth claiming.
        </p>
        <button
          type="button"
          onClick={onBack}
          className="shrink-0 rounded-[7px] border border-rule bg-white px-5 py-2.5 text-[13px] text-body hover:bg-canvas"
        >
          Change the config
        </button>
      </footer>
    </div>
  );
}
