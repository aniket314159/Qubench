/**
 * Which trained model answers.
 *
 * Any model from the run can be used, and switching is cheap — the checkpoint
 * carries its own preprocessing recipe, so a different model means a different
 * pipeline replayed, not a different image.
 *
 * Models that failed during the run have no checkpoint and cannot predict.
 * They are listed disabled rather than hidden, so the absence is explained.
 */
export default function ModelSelect({ results, value, onChange }) {
  return (
    <ul className="space-y-1.5">
      {results.map((r) => {
        const usable = !r.error && !!r.checkpoint;
        const on = r.model === value;
        return (
          <li key={r.model}>
            <button
              type="button"
              disabled={!usable}
              aria-pressed={on}
              onClick={() => onChange(r.model)}
              className={`flex w-full items-center gap-3 rounded-[7px] border px-3 py-2 text-left
                ${on ? "border-quantum bg-quantum-soft" : "border-rule bg-white"}
                ${usable ? "hover:bg-canvas" : "cursor-not-allowed opacity-50"}`}
            >
              <span
                className={`h-[11px] w-[11px] shrink-0 rounded-full border
                  ${on ? "border-quantum bg-quantum" : "border-rule"}`}
              />
              <span className="flex-1 text-[12px] text-ink">{r.label}</span>
              <span
                className={`font-mono text-[10.5px] ${
                  r.kind === "quantum" ? "text-quantum" : "text-classical"
                }`}
              >
                {usable
                  ? `macro-F1 ${r.metrics.f1_macro.toFixed(3)}`
                  : "no checkpoint"}
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
