/**
 * Errors disable the Run button; warnings do not.
 *
 * Both arrays are plain sentences from the backend and are rendered verbatim.
 * They are written to be read aloud — "Compare macro F1 rather than accuracy,
 * accuracy flatters the majority class" is an argument, not a validation
 * string, and paraphrasing it here would waste it.
 */
export default function ValidationBanner({ errors = [], warnings = [] }) {
  if (!errors.length && !warnings.length) return null;
  const bad = errors.length > 0;
  const items = bad ? errors : warnings;

  return (
    <div
      role={bad ? "alert" : "status"}
      className={`rounded-lg border-l-[3px] px-4 py-3 ${
        bad ? "border-alert bg-alert/5" : "border-caution bg-caution/10"
      }`}
    >
      <p className="text-[12px] font-semibold text-ink">
        {bad
          ? `${errors.length} problem${errors.length > 1 ? "s" : ""} — fix before running`
          : `${warnings.length} warning${warnings.length > 1 ? "s" : ""}, nothing blocking`}
      </p>
      <ul className="mt-1 space-y-1">
        {items.map((m, i) => (
          <li key={i} className="text-[11.5px] leading-snug text-body">
            {m}
          </li>
        ))}
      </ul>
      {bad && warnings.length > 0 && (
        <p className="mt-2 text-[10.5px] text-muted">
          {warnings.length} warning{warnings.length > 1 ? "s" : ""} also, hidden
          until the problems are fixed.
        </p>
      )}
    </div>
  );
}
