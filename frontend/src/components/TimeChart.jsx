/**
 * Training time, on a log scale.
 *
 * Log because the spread is genuinely three orders of magnitude — a tuned SVC
 * around ten seconds, a quantum kernel under one, a variational model near
 * eighty. On a linear axis every bar but the slowest is invisible, and the
 * interesting comparison is exactly the one that disappears.
 */
export default function TimeChart({ results }) {
  const ok = results.filter((r) => !r.error);
  if (!ok.length) return null;

  const times = ok.map((r) => Math.max(r.telemetry.fit_seconds, 0.01));
  const top = Math.max(...times);
  const frac = (t) => Math.log10(t / 0.01) / Math.log10(top / 0.01 || 10);

  return (
    <div>
      <p className="mb-3 text-[11px] text-muted">
        Time to train, log scale — lower is better
      </p>
      <ul className="space-y-3">
        {ok.map((r, i) => {
          const quantum = r.kind === "quantum";
          return (
            <li key={r.model}>
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-[12px] text-ink">{r.label}</span>
                <span className="font-mono text-[12px] text-body">
                  {r.telemetry.fit_seconds < 1
                    ? `${(r.telemetry.fit_seconds * 1000).toFixed(0)} ms`
                    : `${r.telemetry.fit_seconds.toFixed(1)} s`}
                </span>
              </div>
              <div className="mt-1 h-2 overflow-hidden rounded-full bg-hairline">
                <div
                  className={`h-full rounded-full ${quantum ? "bg-quantum" : "bg-classical"}`}
                  style={{ width: `${Math.max(frac(times[i]) * 100, 2)}%` }}
                />
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
