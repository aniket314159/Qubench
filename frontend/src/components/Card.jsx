/**
 * A bordered panel. One border, one radius, never a shadow — the design system
 * is explicit about that, and a shadow is the fastest way to make this look
 * like every other dashboard.
 */
export default function Card({ title, sub, right, className = "", children }) {
  return (
    <section
      className={`rounded-[10px] border border-rule bg-white ${className}`}
    >
      {(title || right) && (
        <header className="flex items-start justify-between gap-4 px-5 pt-5">
          <div>
            {title && (
              <h2 className="text-[15px] font-semibold text-ink">{title}</h2>
            )}
            {sub && <p className="mt-0.5 text-[11.5px] text-muted">{sub}</p>}
          </div>
          {right}
        </header>
      )}
      <div className="px-5 pb-5 pt-4">{children}</div>
    </section>
  );
}
