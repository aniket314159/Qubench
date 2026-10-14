import { useState } from "react";
import ParamControl from "./ParamControl";

/**
 * One preprocessing step.
 *
 * Locked ops (to_rgb, resize) show as always-on and cannot be toggled — they
 * guarantee the handoff type the model layer depends on, and the backend
 * appends them whatever the config says.
 */
export default function OpRow({ spec, enabled, params, onToggle, onParam }) {
  const [open, setOpen] = useState(false);
  const keys = Object.keys(spec.params_schema ?? {});
  const hasParams = keys.length > 0 && !spec.locked;

  return (
    <li className="border-b border-hairline last:border-0">
      <div className="flex items-center gap-3 py-2">
        <input
          type="checkbox"
          id={`op-${spec.op}`}
          checked={spec.locked ? true : enabled}
          disabled={spec.locked}
          onChange={(e) => onToggle(e.target.checked)}
          className="h-[15px] w-[15px] shrink-0 rounded-[4px] accent-[var(--color-quantum)]
            disabled:opacity-45"
        />
        <label
          htmlFor={`op-${spec.op}`}
          title={spec.description}
          className={`flex-1 text-[12.5px] ${
            spec.locked
              ? "text-muted"
              : enabled
                ? "font-medium text-ink"
                : "text-muted"
          }`}
        >
          {spec.label}
        </label>

        {spec.locked ? (
          <span className="text-[10.5px] text-muted">always on</span>
        ) : hasParams ? (
          <button
            type="button"
            aria-expanded={open}
            onClick={() => setOpen((o) => !o)}
            className="rounded-[5px] bg-hairline px-2 py-1 font-mono text-[10.5px] text-body
              hover:bg-rule"
          >
            {keys
              .map((k) => String(params?.[k] ?? spec.params_schema[k].default))
              .join(" · ")}
          </button>
        ) : null}
      </div>

      {open && hasParams && (
        <div className="grid gap-3 pb-3 pl-[27px] pr-1 sm:grid-cols-2">
          {keys.map((k) => (
            <ParamControl
              key={k}
              name={`${spec.op}-${k}`}
              schema={spec.params_schema[k]}
              value={params?.[k]}
              disabled={!enabled}
              onChange={(val) => onParam(k, val)}
            />
          ))}
        </div>
      )}
    </li>
  );
}
