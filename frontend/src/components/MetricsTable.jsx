/**
 * Every model, side by side.
 *
 * Macro-F1 comes FIRST and is the emphasised column. On an imbalanced dataset
 * accuracy is actively misleading — a run here scored 0.760 accuracy with
 * 0.484 macro-F1 by getting none of the rarest class right. Ordering the
 * columns is the cheapest way to make people read the correct one.
 *
 * Failed models render as a row with their error, never as a missing row.
 */
export default function MetricsTable({ results }) {
  const ok = results.filter((r) => !r.error);
  const bestF1 = Math.max(...ok.map((r) => r.metrics.f1_macro), 0);

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-left">
        <thead>
          <tr className="border-b border-rule">
            <th className="py-2 pr-3 text-[10.5px] font-normal text-muted">
              Model
            </th>
            <th className="py-2 px-3 text-right text-[10.5px] font-normal text-muted">
              Macro-F1
            </th>
            <th className="py-2 px-3 text-right text-[10.5px] font-normal text-muted">
              Accuracy
            </th>
            <th className="py-2 px-3 text-right text-[10.5px] font-normal text-muted">
              Precision
            </th>
            <th className="py-2 px-3 text-right text-[10.5px] font-normal text-muted">
              Recall
            </th>
            <th className="py-2 pl-3 text-right text-[10.5px] font-normal text-muted">
              Train
            </th>
          </tr>
        </thead>
        <tbody>
          {results.map((r) => {
            const quantum = r.kind === "quantum";
            const best = !r.error && r.metrics.f1_macro === bestF1;
            if (r.error) {
              return (
                <tr
                  key={r.model}
                  className="border-b border-hairline last:border-0"
                >
                  <td className="py-3 pr-3">
                    <span className="text-[12.5px] text-muted">{r.label}</span>
                  </td>
                  <td colSpan={5} className="py-3 pl-3 text-right">
                    <span className="font-mono text-[11px] text-alert">
                      {r.error}
                    </span>
                  </td>
                </tr>
              );
            }
            return (
              <tr
                key={r.model}
                className={`border-b border-hairline last:border-0 ${
                  quantum ? "bg-quantum-soft/60" : ""
                }`}
              >
                <td className="py-3 pr-3">
                  <span
                    className={`text-[12.5px] ${best ? "font-semibold text-ink" : "text-ink"}`}
                  >
                    {r.label}
                  </span>
                  {best && (
                    <span className="ml-2 rounded-full bg-quantum px-2 py-0.5 text-[10px] font-semibold text-white">
                      best
                    </span>
                  )}
                </td>
                <td className="px-3 py-3 text-right">
                  <span
                    className={`font-mono text-[12px] font-semibold ${
                      quantum ? "text-quantum" : "text-classical"
                    }`}
                  >
                    {r.metrics.f1_macro.toFixed(3)}
                  </span>
                </td>
                {[
                  r.metrics.accuracy,
                  r.metrics.precision,
                  r.metrics.recall,
                ].map((v, i) => (
                  <td
                    key={i}
                    className="px-3 py-3 text-right font-mono text-[12px] text-body"
                  >
                    {v.toFixed(3)}
                  </td>
                ))}
                <td className="py-3 pl-3 text-right font-mono text-[12px] text-body">
                  {r.telemetry.fit_seconds.toFixed(1)} s
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
