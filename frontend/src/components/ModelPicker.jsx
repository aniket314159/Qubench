import { useState } from "react";
import ParamControl from "./ParamControl";

/**
 * Which models to race.
 *
 * Grouped by `kind` and coloured by the product's one rule: violet is quantum,
 * teal is classical, everywhere, always. The groups are derived from the
 * catalog — adding a third kind later would render on its own without a change
 * here.
 *
 * Model parameters use the same ParamControl as preprocessing ops, which is
 * the proof that the control really is generic.
 */
export default function ModelPicker({
  catalog,
  selected,
  params,
  onToggle,
  onParam,
}) {
  const [open, setOpen] = useState(null);
  const kinds = [...new Set(catalog.map((m) => m.kind))];

  return (
    <div className="space-y-4">
      {kinds.map((kind) => (
        <div key={kind}>
          <p
            className={`text-[11px] font-semibold capitalize ${
              kind === "quantum" ? "text-quantum" : "text-classical"
            }`}
          >
            {kind}
          </p>
          <ul className="mt-1.5">
            {catalog
              .filter((m) => m.kind === kind)
              .map((m) => {
                const on = selected.has(m.name);
                const keys = Object.keys(m.param_schema ?? {});
                return (
                  <li
                    key={m.name}
                    className="border-b border-hairline last:border-0"
                  >
                    <div className="flex items-center gap-3 py-2">
                      <input
                        id={`m-${m.name}`}
                        type="checkbox"
                        checked={on}
                        onChange={(e) => onToggle(m.name, e.target.checked)}
                        className={`h-[15px] w-[15px] rounded-[4px] ${
                          kind === "quantum"
                            ? "accent-[var(--color-quantum)]"
                            : "accent-[var(--color-classical)]"
                        }`}
                      />
                      <label
                        htmlFor={`m-${m.name}`}
                        className={`flex-1 text-[12px] ${on ? "text-ink" : "text-muted"}`}
                      >
                        {m.label}
                      </label>
                      {keys.length > 0 && (
                        <button
                          type="button"
                          aria-expanded={open === m.name}
                          onClick={() =>
                            setOpen(open === m.name ? null : m.name)
                          }
                          className="rounded-[5px] px-2 py-1 font-mono text-[10.5px] text-muted
                            hover:bg-hairline hover:text-body"
                        >
                          {keys.length} setting{keys.length > 1 ? "s" : ""}
                        </button>
                      )}
                    </div>

                    {open === m.name && (
                      <div className="grid gap-3 pb-3 pl-[27px] pr-1 sm:grid-cols-2">
                        {keys.map((k) => (
                          <ParamControl
                            key={k}
                            name={`${m.name}-${k}`}
                            schema={m.param_schema[k]}
                            value={params?.[m.name]?.[k]}
                            disabled={!on}
                            onChange={(v) => onParam(m.name, k, v)}
                          />
                        ))}
                      </div>
                    )}
                  </li>
                );
              })}
          </ul>
        </div>
      ))}
    </div>
  );
}
