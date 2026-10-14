/**
 * The classes the backend found, with counts.
 *
 * Names come from `meta.class_names` and `meta.counts` — never from a constant
 * in this file. A literal "MI" here would break the moment someone uploads a
 * two-class dataset, which is exactly the case the model layer is tested
 * against.
 */
export default function ClassChips({ meta }) {
  const names = meta.class_names ?? [];
  const counts = meta.counts ?? {};
  const values = names.map((n) => counts[n] ?? 0);
  const largest = Math.max(...values, 1);
  const smallest = Math.min(...values, largest);
  const ratio = smallest ? largest / smallest : Infinity;

  return (
    <div>
      <div className="flex items-baseline justify-between">
        <h2 className="text-[15px] font-semibold text-ink">
          Detected in this dataset
        </h2>
        <p className="font-mono text-[11.5px] text-muted">
          {meta.n_classes} classes · {meta.n_samples.toLocaleString()} images
        </p>
      </div>

      <ul className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {names.map((name) => (
          <li
            key={name}
            className="overflow-hidden rounded-[10px] border border-rule bg-white"
          >
            <div className="h-1 bg-quantum" />
            <div className="px-4 py-3">
              <p className="text-[17px] font-semibold text-ink">{name}</p>
              <p className="mt-2 font-mono text-[20px] font-semibold text-ink">
                {(counts[name] ?? 0).toLocaleString()}
                <span className="ml-1.5 font-sans text-[10.5px] font-normal text-muted">
                  images
                </span>
              </p>
            </div>
          </li>
        ))}
      </ul>

      {ratio >= 3 && (
        <p className="mt-3 rounded-lg border-l-[3px] border-classical bg-classical-soft px-4 py-3 text-[11.5px] text-body">
          <span className="font-semibold text-ink">Classes are uneven</span> —{" "}
          {largest.toLocaleString()} in the largest against{" "}
          {smallest.toLocaleString()} in the smallest. Compare macro-F1 rather
          than accuracy: accuracy rewards a model that ignores the rarest class.
        </p>
      )}
    </div>
  );
}
