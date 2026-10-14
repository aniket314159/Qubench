import { useEffect, useState } from "react";

import { ApiError, getLeaderboard } from "../api";
import Card from "../components/Card";
import Spinner from "../components/Spinner";

/**
 * Screen 5 of 5. Best classical against best quantum, per dataset, across
 * every run.
 *
 * This screen exists to stop a single lucky run being mistaken for a result.
 * One configuration winning by three points is a lead; five configurations
 * with one win and one loss is a finding, and it is a more honest one.
 *
 * The delta column is deliberately signed and deliberately not celebrated —
 * negative values render in red and stay on the page.
 */
function Delta({ value }) {
  if (value == null)
    return <span className="font-mono text-[13px] text-muted">—</span>;
  const positive = value > 0;
  const zero = Math.abs(value) < 0.0005;
  return (
    <span
      className={`font-mono text-[13px] font-semibold ${
        zero ? "text-muted" : positive ? "text-ok" : "text-alert"
      }`}
    >
      {zero ? "0.000" : `${positive ? "+" : "−"}${Math.abs(value).toFixed(3)}`}
    </span>
  );
}

export default function Leaderboard({ onOpenRun }) {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getLeaderboard()
      .then(setRows)
      .catch((e) =>
        setError(
          e instanceof ApiError ? e.detail : "Could not load the leaderboard.",
        ),
      );
  }, []);

  const wins = rows?.filter((r) => (r.delta ?? 0) > 0).length ?? 0;

  return (
    <div className="mx-auto max-w-[1440px] px-9 py-7">
      <header>
        <h1 className="text-[25px] font-semibold text-ink">Every run so far</h1>
        <p className="mt-1 text-[13px] text-body">
          Best classical against best quantum, per dataset. This is the
          experiment record, not a single result.
        </p>
      </header>

      {error && (
        <p
          role="alert"
          className="mt-6 rounded-lg border-l-[3px] border-alert bg-alert/5 px-4 py-3 text-[12px] text-body"
        >
          {error}
        </p>
      )}

      {rows === null && !error && (
        <p className="mt-8 flex items-center gap-2 text-[13px] text-muted">
          <Spinner size={15} /> Loading every run
        </p>
      )}

      {rows?.length === 0 && (
        <Card className="mt-6 max-w-[560px]" title="Nothing here yet">
          <p className="text-[12px] text-body">
            Run a benchmark and it will appear here. The table only becomes
            interesting after three or four configurations, when it starts
            showing which choices actually moved the number.
          </p>
        </Card>
      )}

      {rows?.length > 0 && (
        <>
          <div className="mt-6 overflow-x-auto rounded-[10px] border border-rule bg-white">
            <table className="w-full border-collapse text-left">
              <thead>
                <tr className="border-b border-rule">
                  <th className="px-5 py-3 text-[10.5px] font-normal text-muted">
                    Dataset
                  </th>
                  <th className="px-3 py-3 text-right text-[10.5px] font-normal text-muted">
                    Runs
                  </th>
                  <th className="px-3 py-3 text-right text-[10.5px] font-normal text-muted">
                    Best classical
                  </th>
                  <th className="px-3 py-3 text-right text-[10.5px] font-normal text-muted">
                    Best quantum
                  </th>
                  <th className="px-3 py-3 text-right text-[10.5px] font-normal text-muted">
                    Difference
                  </th>
                  <th className="px-5 py-3 text-[10.5px] font-normal text-muted">
                    Winning configuration
                  </th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => {
                  const best = r.best_quantum ?? r.best_classical;
                  return (
                    <tr
                      key={r.dataset_id}
                      className="border-b border-hairline last:border-0"
                    >
                      <td className="px-5 py-4">
                        <p className="text-[12.5px] font-medium text-ink">
                          {r.dataset_name}
                        </p>
                        <p className="font-mono text-[10.5px] text-muted">
                          {r.dataset_id}
                        </p>
                      </td>
                      <td className="px-3 py-4 text-right font-mono text-[12px] text-body">
                        {r.n_runs}
                      </td>
                      <td className="px-3 py-4 text-right">
                        <span className="font-mono text-[13px] font-semibold text-classical">
                          {r.best_classical
                            ? r.best_classical.f1_macro.toFixed(3)
                            : "—"}
                        </span>
                        <span className="block text-[10px] text-muted">
                          {r.best_classical?.label ?? ""}
                        </span>
                      </td>
                      <td className="px-3 py-4 text-right">
                        <span className="font-mono text-[13px] font-semibold text-quantum">
                          {r.best_quantum
                            ? r.best_quantum.f1_macro.toFixed(3)
                            : "—"}
                        </span>
                        <span className="block text-[10px] text-muted">
                          {r.best_quantum?.label ?? ""}
                        </span>
                      </td>
                      <td className="px-3 py-4 text-right">
                        <Delta value={r.delta} />
                      </td>
                      <td className="px-5 py-4">
                        {best && (
                          <button
                            type="button"
                            onClick={() => onOpenRun?.(best.run_id)}
                            className="font-mono text-[11px] text-body underline decoration-rule
                              underline-offset-4 hover:text-quantum"
                          >
                            {best.encoding} · {best.n_features} features
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="mt-6 grid gap-6 lg:grid-cols-2">
            <Card title="What this table is for">
              <p className="text-[12px] leading-relaxed text-body">
                One run proves nothing. The comparison is only meaningful
                because the classical arm is grid-searched before every
                comparison, the split is fixed by seed, and both arms are handed
                byte-identical features. Everything that differs between two
                rows is a choice someone made, not noise.
              </p>
            </Card>
            <Card title="Read the difference, not the headline">
              <p className="text-[12px] leading-relaxed text-body">
                Quantum leads on {wins} of {rows.length} dataset
                {rows.length === 1 ? "" : "s"} here, by a few points. That is
                what the measurement says, and it is what we present. Published
                double-digit gaps on this task compare against an untuned
                baseline; ours does not.
              </p>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
