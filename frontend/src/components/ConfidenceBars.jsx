/**
 * Per-class probability, sorted.
 *
 * Sorted rather than fixed in class order because the runner-up is the
 * interesting number: 58% against 31% is a different clinical situation from
 * 58% against 4%, and a fixed order buries that.
 */
export default function ConfidenceBars({ confidences }) {
  const rows = Object.entries(confidences ?? {}).sort((a, b) => b[1] - a[1]);
  if (!rows.length) return null;

  return (
    <ul className="space-y-3">
      {rows.map(([name, v], i) => (
        <li key={name}>
          <div className="flex items-baseline justify-between gap-3">
            <span
              className={`text-[11.5px] ${i === 0 ? "font-semibold text-ink" : "text-body"}`}
            >
              {name}
            </span>
            <span className="font-mono text-[11.5px] font-semibold text-ink">
              {(v * 100).toFixed(1)}%
            </span>
          </div>
          <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-hairline">
            <div
              className={`h-full rounded-full ${i === 0 ? "bg-quantum" : "bg-rule"}`}
              style={{ width: `${Math.max(v * 100, 1)}%` }}
            />
          </div>
        </li>
      ))}
    </ul>
  );
}
