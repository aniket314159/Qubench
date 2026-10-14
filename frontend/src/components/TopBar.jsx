import { MOCK } from "../api";

/**
 * 62px, navy, the only place navy appears in the product.
 *
 * The dataset caption on the right is the answer to "what am I looking at" and
 * stays visible on every screen, so a judge who glances up mid-demo is never
 * lost.
 */
export default function TopBar({ dataset, wired }) {
  const missing = wired
    ? Object.entries(wired)
        .filter(([, ok]) => !ok)
        .map(([k]) => k)
    : [];

  return (
    <header className="flex h-[62px] items-center justify-between bg-navy px-9">
      <div>
        <p className="text-[17px] font-semibold leading-none text-white">
          QuBench
        </p>
        <p className="mt-1 text-[10.5px] leading-none text-white/55">
          Early disease detection · quantum vs classical
        </p>
      </div>

      <div className="flex items-center gap-4 text-right">
        {MOCK && (
          <span className="rounded-full bg-caution/20 px-2.5 py-1 font-mono text-[10px] text-caution">
            fixtures
          </span>
        )}
        {missing.length > 0 && (
          <span
            title={`Not wired: ${missing.join(", ")}`}
            className="rounded-full bg-alert/20 px-2.5 py-1 font-mono text-[10px] text-alert"
          >
            {missing.length} endpoint{missing.length > 1 ? "s" : ""} offline
          </span>
        )}
        <div>
          <p className="text-[11px] leading-none text-white/80">
            {dataset
              ? `${dataset.name} · ${dataset.n_classes} classes · ${dataset.n_samples.toLocaleString()} images`
              : "No dataset selected"}
          </p>
          <p className="mt-1 text-[10.5px] leading-none text-white/40">
            Runs · Docs
          </p>
        </div>
      </div>
    </header>
  );
}
