/**
 * Macro-F1 per model, as bars, in the two lanes.
 *
 * A chart library would be four hundred kilobytes for what is a div with a
 * width. It would also make it easy to lose the colour rule, which is the one
 * thing this chart must not do: violet is quantum, teal is classical, and the
 * comparison between those two lanes is the entire point of the product.
 */
export default function AccuracyChart({
  results,
  metric = "f1_macro",
  label = "Macro-F1",
}) {
  const ok = results.filter((r) => !r.error);
  if (!ok.length) return null;
  const best = Math.max(...ok.map((r) => r.metrics[metric]));

  return (
    <div>
      <p className="mb-3 text-[11px] text-muted">{label}, higher is better</p>
      <ul className="space-y-3">
        {ok.map((r) => {
          const v = r.metrics[metric];
          const quantum = r.kind === "quantum";
          return (
            <li key={r.model}>
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-[12px] text-ink">{r.label}</span>
                <span
                  className={`font-mono text-[12px] font-semibold ${
                    quantum ? "text-quantum" : "text-classical"
                  }`}
                >
                  {v.toFixed(3)}
                </span>
              </div>
              <div className="mt-1 h-2 overflow-hidden rounded-full bg-hairline">
                <div
                  className={`h-full rounded-full ${quantum ? "bg-quantum" : "bg-classical"}`}
                  style={{
                    width: `${Math.max((v / Math.max(best, 0.0001)) * 100, 2)}%`,
                  }}
                />
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
