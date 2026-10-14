import { qubitReadout } from "../api";

/**
 * Backbone, feature count, encoding — the three choices that decide what the
 * quantum models actually see.
 *
 * Two things here are easy to get wrong and matter:
 *
 *   The slider picks FEATURES, not qubits. Qubits are derived, and for
 *   amplitude encoding the two diverge sharply — 16 features is 4 qubits.
 *   Labelling this "qubits" would make the best demo moment unexplainable.
 *
 *   The slider's ceiling is the SELECTED encoding's `max_features`. Switching
 *   from angle_y (12) to zz (10) with 12 features chosen would otherwise send
 *   a config the backend rejects.
 */
export default function EncodingPanel({
  backbones,
  encodings,
  backbone,
  encoding,
  nFeatures,
  onBackbone,
  onEncoding,
  onFeatures,
}) {
  const enc = encodings.find((e) => e.name === encoding) ?? encodings[0];
  const max = enc?.max_features ?? 12;
  const readout = enc ? qubitReadout(enc, nFeatures) : null;

  return (
    <div className="space-y-5">
      <div>
        <p className="text-[11.5px] font-medium text-body">Backbone</p>
        <div className="mt-2 grid gap-2 sm:grid-cols-2">
          {backbones.map((b) => {
            const on = b.name === backbone;
            return (
              <button
                key={b.name}
                type="button"
                aria-pressed={on}
                onClick={() => onBackbone(b.name)}
                className={`rounded-[7px] border px-3 py-2 text-left
                  ${on ? "border-quantum bg-quantum-soft" : "border-rule bg-white hover:bg-canvas"}`}
              >
                <span className="block font-mono text-[12px] font-semibold text-ink">
                  {b.name}
                </span>
                <span className="block text-[10px] text-muted">
                  {b.dim} dims · {b.input_size}px
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <div>
        <div className="flex items-baseline justify-between">
          <label htmlFor="nfeat" className="text-[12.5px] font-medium text-ink">
            Keep this many features
          </label>
          <span className="font-mono text-[16px] font-semibold text-quantum">
            {nFeatures}
          </span>
        </div>
        <input
          id="nfeat"
          type="range"
          min={2}
          max={max}
          value={Math.min(nFeatures, max)}
          onChange={(e) => onFeatures(Number(e.target.value))}
          className="mt-2 w-full accent-[var(--color-quantum)]"
        />
        <div className="mt-1 flex justify-between font-mono text-[10px] text-muted">
          <span>2</span>
          <span>
            {max} max for {enc?.name}
          </span>
        </div>
      </div>

      <div>
        <p className="text-[11.5px] font-medium text-body">Encoding</p>
        <ul className="mt-2 space-y-1.5">
          {encodings.map((e) => {
            const on = e.name === encoding;
            return (
              <li key={e.name}>
                <button
                  type="button"
                  aria-pressed={on}
                  onClick={() => onEncoding(e.name)}
                  title={e.description}
                  className={`flex w-full items-center gap-3 rounded-[7px] border px-3 py-2
                    ${on ? "border-quantum bg-quantum-soft" : "border-rule bg-white hover:bg-canvas"}`}
                >
                  <span
                    className={`h-[11px] w-[11px] shrink-0 rounded-full border
                      ${on ? "border-quantum bg-quantum" : "border-rule"}`}
                  />
                  <span className="font-mono text-[12px] font-semibold text-ink">
                    {e.name}
                  </span>
                  <span className="flex-1 truncate text-left text-[11px] text-muted">
                    {e.description}
                  </span>
                  <span className="shrink-0 font-mono text-[10.5px] text-muted">
                    {e.qubit_formula}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      </div>

      {readout && (
        <p className="rounded-md bg-quantum-soft px-3 py-2 font-mono text-[11.5px] text-quantum">
          {readout.text}
        </p>
      )}
    </div>
  );
}
