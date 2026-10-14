/**
 * The four preprocessing presets. Choosing one replaces the whole op list,
 * which is why the label flips to "changed" the moment anything is toggled —
 * a preset name that no longer describes the config is a lie on screen.
 */
export default function PresetSelect({ presets, value, dirty, onChange }) {
  const current = presets.find((p) => p.name === value);
  return (
    <div>
      <div className="flex items-center gap-2">
        <select
          aria-label="Preprocessing preset"
          value={value ?? ""}
          onChange={(e) => onChange(e.target.value)}
          className="flex-1 rounded-[6px] border border-rule bg-hairline px-3 py-2
            font-mono text-[12px] font-semibold text-ink"
        >
          {presets.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name}
            </option>
          ))}
        </select>
        {dirty && <span className="text-[10.5px] text-muted">changed</span>}
      </div>
      {current?.description && (
        <p className="mt-1.5 text-[11px] text-muted">{current.description}</p>
      )}
    </div>
  );
}
