/**
 * A transcript of the run, built from the progress stream.
 *
 * The backend sends one `message` per poll, not a history — so this records
 * each DISTINCT message as it first appears, with the elapsed time at which it
 * did. No backend change, and no invented events: every line here was
 * genuinely emitted by the worker.
 *
 * It matters more than it looks. A progress bar says "something is happening";
 * this says what. "PCA to 8 features", "Training Quantum Kernel SVC (3/3)" is
 * the difference between a demo that looks like a loading screen and one that
 * looks like a pipeline.
 */
export default function ActivityLog({ entries }) {
  if (!entries?.length) {
    return (
      <p className="text-[11.5px] text-muted">
        Waiting for the first update from the worker.
      </p>
    );
  }

  // The card this list sits in is its column's flex filler (Benchmark's
  // running view), so on desktop the list fills whatever height is left and
  // scrolls only once full — a hard max-height would fight that fill. Below
  // lg the columns stack and there is no leftover height, so the old cap
  // stays to keep a long transcript from running the page long.
  return (
    <ol className="min-h-0 max-h-[200px] flex-1 space-y-1.5 overflow-scroll pr-2 scroll-smooth lg:max-h-[200px]">
      {entries.map((e, i) => {
        const last = i === entries.length - 1;
        return (
          <li key={`${e.at}-${i}`} className="flex items-baseline gap-3">
            <span className="w-11 shrink-0 font-mono text-[10.5px] text-muted">
              {e.at}
            </span>
            <span
              className={`mt-[5px] h-[5px] w-[5px] shrink-0 rounded-full ${
                last ? "bg-quantum" : "bg-rule"
              }`}
            />
            <span
              className={`text-[11.5px] leading-snug ${
                last ? "font-medium text-ink" : "text-body"
              }`}
            >
              {e.message}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
