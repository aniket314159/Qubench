/**
 * Recall per class, with how many test images each had.
 *
 * The support count is not decoration. A recall of 0.53 on fifteen images can
 * only take sixteen distinct values, so it moves in jumps of 6.7 points and
 * should not be read as precise. Showing the denominator next to the bar is
 * the cheapest possible defence against over-reading a small test set.
 */
export default function PerClassBars({ perClass, classNames }) {
  const rows = classNames.map((n) => [n, perClass?.[n]]).filter(([, m]) => m);

  if (!rows.length) {
    return (
      <p className="text-[11.5px] text-muted">
        No per-class figures for this model.
      </p>
    );
  }

  return (
    <ul className="space-y-3.5">
      {rows.map(([name, m]) => (
        <li key={name}>
          <div className="flex items-baseline justify-between gap-3">
            <span className="text-[12px] text-ink">{name}</span>
            <span className="font-mono text-[12px] font-semibold text-ink">
              {m.recall.toFixed(3)}
            </span>
          </div>
          <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-hairline">
            <div
              className={`h-full rounded-full ${m.recall >= 0.6 ? "bg-quantum" : "bg-caution"}`}
              style={{ width: `${Math.max(m.recall * 100, 1.5)}%` }}
            />
          </div>
          <p className="mt-1 text-[10.5px] text-muted">
            {m.support} test image{m.support === 1 ? "" : "s"}
            {m.support > 0 && (
              <> · one image is {(100 / m.support).toFixed(1)} points</>
            )}
          </p>
        </li>
      ))}
    </ul>
  );
}
