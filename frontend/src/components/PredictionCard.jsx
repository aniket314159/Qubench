import ConfidenceBars from "./ConfidenceBars";

/**
 * The answer.
 *
 * "Most likely class", never "Diagnosis". The heading is the single most
 * load-bearing piece of copy in the product: this is a research prototype
 * trained on a few hundred images, and every word here has to keep that
 * visible without making the screen useless.
 *
 * A low top confidence is called out rather than left for the reader to
 * notice — a 38% winner over a 34% runner-up is not an answer, and the
 * interface should say so.
 */
export default function PredictionCard({ prediction, modelLabel }) {
  if (!prediction) return null;
  const top = prediction.confidences?.[prediction.label] ?? 0;
  const sorted = Object.values(prediction.confidences ?? {}).sort(
    (a, b) => b - a,
  );
  const margin = sorted.length > 1 ? sorted[0] - sorted[1] : 1;
  const weak = top < 0.5 || margin < 0.15;

  return (
    <div className="rounded-[10px] border border-rule bg-white p-5">
      <p className="text-[11px] text-muted">Most likely class</p>
      <div className="mt-1 flex items-baseline justify-between gap-4">
        <p className="text-[26px] font-semibold text-ink">{prediction.label}</p>
        <p className="font-mono text-[30px] font-semibold text-quantum">
          {(top * 100).toFixed(0)}%
        </p>
      </div>
      <p className="mt-1 font-mono text-[10.5px] text-muted">
        {modelLabel} · {prediction.latency_ms.toFixed(0)} ms
      </p>

      {weak && (
        <p className="mt-3 rounded-md border-l-[3px] border-caution bg-caution/10 px-3 py-2 text-[11px] text-body">
          The model is not confident here. The top two classes are within{" "}
          {(margin * 100).toFixed(0)} points of each other — treat this as
          undecided rather than as an answer.
        </p>
      )}

      <div className="mt-4 border-t border-hairline pt-4">
        <ConfidenceBars confidences={prediction.confidences} />
      </div>
    </div>
  );
}
