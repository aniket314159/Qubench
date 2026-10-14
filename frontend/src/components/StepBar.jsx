/**
 * 58px, white, one row of five.
 *
 * A step is reachable only when its prerequisite exists — you cannot configure
 * without a dataset or diagnose without a finished run. Disabled steps stay
 * visible rather than disappearing, so the shape of the flow is legible from
 * the first screen.
 */
export default function StepBar({ steps, current, onGo, right }) {
  return (
    <nav
      aria-label="Progress"
      className="flex h-[58px] items-center justify-between border-b border-rule bg-white px-9"
    >
      <ol className="flex items-center gap-7">
        {steps.map((s, i) => {
          const active = s.id === current;
          const done = s.done && !active;
          return (
            <li key={s.id}>
              <button
                type="button"
                disabled={!s.enabled}
                aria-current={active ? "step" : undefined}
                onClick={() => onGo(s.id)}
                className={`flex items-center gap-2.5 rounded-md px-1 py-1 text-[13px]
                  ${active ? "font-semibold text-ink" : "text-muted"}
                  ${s.enabled ? "hover:text-ink" : "cursor-not-allowed opacity-45"}`}
              >
                <span
                  className={`grid h-[18px] w-[18px] place-items-center rounded-full border
                    text-[10.5px] font-semibold
                    ${
                      active
                        ? "border-quantum bg-quantum text-white"
                        : done
                          ? "border-ok text-ok"
                          : "border-rule text-muted"
                    }`}
                >
                  {done ? (
                    <svg
                      width="10"
                      height="10"
                      viewBox="0 0 12 12"
                      aria-hidden="true"
                    >
                      <path
                        d="M2.5 6.2l2.3 2.3L9.5 3.8"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.8"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  ) : (
                    i + 1
                  )}
                </span>
                {s.label}
              </button>
            </li>
          );
        })}
      </ol>

      <p className="font-mono text-[11.5px] text-muted">{right}</p>
    </nav>
  );
}
