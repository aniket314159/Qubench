import OpRow from "./OpRow";

/**
 * Every op the backend offers, in the order it runs.
 *
 * The catalog arrives sorted by `order` and is rendered as given. That order is
 * load-bearing: grid removal identifies ECG paper by colour saturation, so it
 * must run before greyscale or it silently does nothing. Users toggle and tune;
 * they never reorder, and there is no drag handle here on purpose.
 */
export default function OpList({
  catalog,
  enabled,
  params,
  onToggle,
  onParam,
}) {
  return (
    <div>
      <ul>
        {catalog.map((spec) => (
          <OpRow
            key={spec.op}
            spec={spec}
            enabled={enabled.has(spec.op)}
            params={params[spec.op]}
            onToggle={(on) => onToggle(spec.op, on)}
            onParam={(k, v) => onParam(spec.op, k, v)}
          />
        ))}
      </ul>
      <p className="mt-3 rounded-md bg-quantum-soft px-3 py-2 text-[11px] text-quantum">
        Steps run in this fixed order. Grid removal needs the colour, so it runs
        before greyscale.
      </p>
    </div>
  );
}
