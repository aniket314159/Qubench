import Spinner from "./Spinner";

/**
 * While the run is going.
 *
 * There is no cancel button, and that is not an oversight: a Python worker
 * thread cannot be killed from outside, so an interface offering "Cancel" would
 * be lying. What it offers instead is the truth — you can leave, it keeps
 * going — plus the cost estimate shown before the button was ever pressed.
 */
const STAGES = [
  ["preprocessing", "Cleaning images"],
  ["embedding", "Extracting features"],
  ["projecting", "Reducing to N dimensions"],
  ["training", "Training models"],
];

export default function JobProgress({ job, elapsed, estimate }) {
  if (!job) return null;
  const failed = job.status === "failed";
  const at = STAGES.findIndex(([s]) => s === job.status);

  return (
    <div className="space-y-5">
      <div className="rounded-[10px] border border-rule bg-white p-5">
        <div className="flex items-baseline justify-between">
          <h2 className="text-[15px] font-semibold text-ink">
            {failed ? "The run stopped" : "Overall progress"}
          </h2>
          <p
            className={`font-mono text-[22px] font-semibold ${
              failed ? "text-alert" : "text-quantum"
            }`}
          >
            {job.pct}%
          </p>
        </div>

        <div className="mt-3 h-2.5 overflow-hidden rounded-full bg-hairline">
          <div
            className={`h-full rounded-full transition-[width] duration-500 ${
              failed ? "bg-alert" : "bg-quantum"
            }`}
            style={{ width: `${Math.max(job.pct, 2)}%` }}
          />
        </div>

        <p className="mt-3 flex items-center gap-2 text-[12.5px] text-body">
          {!failed && job.status !== "done" && (
            <Spinner size={13} className="text-quantum" />
          )}
          {job.message}
        </p>

        <div className="mt-3 flex flex-wrap gap-x-8 gap-y-1 font-mono text-[11px] text-muted">
          <span>elapsed {elapsed}</span>
          {estimate?.est_seconds != null && (
            <span>estimated {Math.round(estimate.est_seconds)} s</span>
          )}
          <span>preprocessing runs on every job, cached or not</span>
        </div>
      </div>

      {failed ? (
        <div
          role="alert"
          className="rounded-lg border-l-[3px] border-alert bg-alert/5 px-4 py-3"
        >
          <p className="text-[12px] font-semibold text-ink">
            One step raised an error and the run ended.
          </p>
          <p className="mt-1 font-mono text-[11.5px] text-body">{job.error}</p>
          <p className="mt-2 text-[11px] text-muted">
            A single model failing does not end a run — it is reported as a
            failed row instead. This ended earlier than that.
          </p>
        </div>
      ) : (
        <ol className="grid gap-2 sm:grid-cols-4">
          {STAGES.map(([id, label], i) => {
            const done = at > i || job.status === "done";
            const now = at === i;
            return (
              <li
                key={id}
                className={`rounded-[7px] border px-3 py-2 ${
                  now
                    ? "border-quantum bg-quantum-soft"
                    : done
                      ? "border-rule bg-white"
                      : "border-rule bg-white opacity-50"
                }`}
              >
                <p className="font-mono text-[10px] text-muted">{id}</p>
                <p
                  className={`mt-0.5 text-[11.5px] ${
                    now
                      ? "font-semibold text-quantum"
                      : done
                        ? "text-ink"
                        : "text-muted"
                  }`}
                >
                  {label}
                </p>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
