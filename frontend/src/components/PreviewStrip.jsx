import Spinner from "./Spinner";

/**
 * What the models will actually see.
 *
 * Every stage of the pipeline on one real image, regenerated as the op list
 * changes. This is the fastest way to catch a preset that is quietly
 * destroying the trace — a binarise threshold that erases the signal looks
 * fine in the checkbox list and obvious here.
 *
 * The payload is around 200 KB, so the caller debounces.
 */
export default function PreviewStrip({ preview, loading, error }) {
  if (error) {
    return <p className="text-[11.5px] text-caution">{error}</p>;
  }
  if (!preview) {
    return (
      <p className="flex items-center gap-2 text-[11.5px] text-muted">
        {loading && <Spinner size={13} />}
        {loading ? "Applying the pipeline to one image" : "No preview yet."}
      </p>
    );
  }

  const stages = [
    { op: "original", image_b64: preview.original_b64 },
    ...preview.stages,
  ];

  return (
    <div className={loading ? "opacity-50 transition-opacity" : ""}>
      <ul className="grid grid-cols-3 gap-3">
        {stages.map((s, i) => (
          <li key={`${s.op}-${i}`}>
            <img
              src={s.image_b64}
              alt={`After ${s.op}`}
              className="aspect-[3/2] w-full rounded-[5px] border border-rule object-cover"
            />
            <p className="mt-1 text-center font-mono text-[10px] text-muted">
              {s.op}
            </p>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-[11px] text-muted">
        The last tile is exactly what the backbone receives.
      </p>
    </div>
  );
}
