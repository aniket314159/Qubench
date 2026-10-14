import Spinner from "./Spinner";

/**
 * Datasets already ingested on this machine.
 *
 * This is what makes a live demo fast: the images are on disk and, for a
 * config that has been run before, the embeddings are cached — so a benchmark
 * returns in seconds rather than a minute.
 */
function balance(counts = {}) {
  const v = Object.values(counts);
  if (!v.length) return null;
  const hi = Math.max(...v);
  const lo = Math.min(...v);
  if (!lo) return null;
  return { hi, lo, ratio: hi / lo };
}

export default function DatasetList({
  datasets,
  loading,
  onSelect,
  selectedId,
  error,
  onRetry,
}) {
  // Order matters: an error must win over the empty state, or a backend that
  // is down reads as a machine with no data.
  if (error) {
    return (
      <div role="alert">
        <p className="text-[12px] text-body">
          <span className="font-semibold text-ink">
            Could not list datasets.
          </span>{" "}
          {error}
        </p>
        <p className="mt-1 text-[11px] text-muted">
          Datasets already ingested are still on disk — this is a connection
          problem, not missing data.
        </p>
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="mt-3 rounded-[7px] border border-rule px-3 py-1.5 text-[12px] text-body hover:bg-canvas"
          >
            Try again
          </button>
        )}
      </div>
    );
  }

  if (loading) {
    return (
      <p className="flex items-center gap-2 text-[12px] text-muted">
        <Spinner size={14} /> Looking for datasets on this machine
      </p>
    );
  }

  if (!datasets?.length) {
    return (
      <p className="text-[12px] text-muted">
        Nothing ingested yet. Upload a zip above and it will appear here, ready
        to reuse without a second upload.
      </p>
    );
  }

  return (
    <ul className="divide-y divide-hairline">
      {datasets.map((d) => {
        const b = balance(d.counts);
        const even = b && b.ratio < 1.5;
        const chosen = d.dataset_id === selectedId;
        return (
          <li
            key={d.dataset_id}
            className="flex flex-wrap items-center gap-x-6 gap-y-2 py-3"
          >
            <div className="min-w-[220px] flex-1">
              <p className="text-[12.5px] font-medium text-ink">{d.name}</p>
              <p className="text-[10.5px] text-muted">
                {d.class_names.join(" · ")}
              </p>
            </div>

            <p className="w-20 text-right font-mono text-[12px] text-body">
              {d.n_samples.toLocaleString()}
            </p>
            <p className="w-16 text-right font-mono text-[12px] text-body">
              {d.n_classes}
            </p>

            <div className="w-40">
              {b && (
                <>
                  <div className="h-1.5 overflow-hidden rounded-full bg-hairline">
                    <div
                      className={`h-full rounded-full ${even ? "bg-ok" : "bg-caution"}`}
                      style={{ width: `${Math.max((b.lo / b.hi) * 100, 4)}%` }}
                    />
                  </div>
                  <p
                    className={`mt-1 font-mono text-[10px] ${even ? "text-ok" : "text-caution"}`}
                  >
                    {even ? "even" : `${Math.round(b.ratio)} : 1`}
                  </p>
                </>
              )}
            </div>

            <button
              type="button"
              onClick={() => onSelect(d)}
              aria-pressed={chosen}
              className={`rounded-[7px] border px-3 py-1.5 text-[12px] transition-colors
                ${
                  chosen
                    ? "border-quantum bg-quantum text-white"
                    : "border-rule bg-white text-body hover:bg-canvas"
                }`}
            >
              {chosen ? "Selected" : "Use this"}
            </button>
          </li>
        );
      })}
    </ul>
  );
}
