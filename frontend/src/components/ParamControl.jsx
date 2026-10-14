/**
 * One parameter, rendered from its schema. Nothing else in the app knows what
 * parameters exist.
 *
 * THIS COMPONENT IS WHY PHASE 2 IS FREE. A new op or a new model ships a
 * `params_schema` from the backend and its controls appear here with no
 * frontend change at all. The moment someone writes `if (op === "binarize")`
 * in a component, that stops being true.
 *
 * Four types, matching the backend exactly:
 *
 *   int    { min, max, default, label, help? }        slider + number
 *   float  { min, max, default, label, step?, help? } slider + number
 *   enum   { options, default, label, help? }         segmented buttons
 *   bool   { default, label, help? }                  checkbox
 *
 * An unknown type renders a warning rather than nothing, so a backend that
 * grows a fifth type is visible rather than silently missing a control.
 */
export default function ParamControl({
  name,
  schema,
  value,
  onChange,
  disabled,
}) {
  const id = `p-${name}`;
  const v = value ?? schema.default;

  const label = (
    <label htmlFor={id} className="text-[11.5px] text-body">
      {schema.label ?? name}
    </label>
  );
  const help = schema.help && (
    <p className="mt-1 text-[10.5px] leading-snug text-muted">{schema.help}</p>
  );

  if (schema.type === "bool") {
    return (
      <div className={disabled ? "opacity-50" : ""}>
        <div className="flex items-center gap-2">
          <input
            id={id}
            type="checkbox"
            disabled={disabled}
            checked={!!v}
            onChange={(e) => onChange(e.target.checked)}
            className="h-[15px] w-[15px] rounded-[4px] accent-[var(--color-quantum)]"
          />
          {label}
        </div>
        {help}
      </div>
    );
  }

  if (schema.type === "enum") {
    return (
      <div className={disabled ? "opacity-50" : ""}>
        {label}
        <div className="mt-1 flex flex-wrap gap-1">
          {schema.options.map((opt) => {
            const on = v === opt;
            return (
              <button
                key={opt}
                type="button"
                disabled={disabled}
                aria-pressed={on}
                onClick={() => onChange(opt)}
                className={`rounded-[5px] border px-2 py-1 font-mono text-[10.5px]
                  ${
                    on
                      ? "border-quantum bg-quantum-soft font-semibold text-quantum"
                      : "border-rule bg-white text-body hover:bg-canvas"
                  }`}
              >
                {opt}
              </button>
            );
          })}
        </div>
        {help}
      </div>
    );
  }

  if (schema.type === "int" || schema.type === "float") {
    const isInt = schema.type === "int";
    // Float sliders need a step or the browser quantises to 1 and a 0–0.25
    // ratio becomes a two-position switch.
    const step = schema.step ?? (isInt ? 1 : (schema.max - schema.min) / 100);
    const clamp = (n) =>
      Math.min(schema.max, Math.max(schema.min, isInt ? Math.round(n) : n));

    return (
      <div className={disabled ? "opacity-50" : ""}>
        <div className="flex items-baseline justify-between gap-2">
          {label}
          <input
            type="number"
            aria-label={`${schema.label ?? name} value`}
            disabled={disabled}
            value={v}
            min={schema.min}
            max={schema.max}
            step={step}
            onChange={(e) => {
              const n = Number(e.target.value);
              if (!Number.isNaN(n)) onChange(clamp(n));
            }}
            className="w-16 rounded-[5px] border border-rule px-1.5 py-0.5
              text-right font-mono text-[11px] text-ink"
          />
        </div>
        <input
          id={id}
          type="range"
          disabled={disabled}
          value={v}
          min={schema.min}
          max={schema.max}
          step={step}
          onChange={(e) => onChange(clamp(Number(e.target.value)))}
          className="mt-1 w-full accent-[var(--color-quantum)]"
        />
        {help}
      </div>
    );
  }

  return (
    <p className="text-[10.5px] text-caution">
      No control for parameter type “{schema.type}” — add one to ParamControl.
    </p>
  );
}
