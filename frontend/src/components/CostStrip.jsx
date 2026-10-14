import Spinner from "./Spinner";

/**
 * What this run will cost, before it starts.
 *
 * This is the only defence against a config that runs for hours: there is no
 * cancel, because a Python worker thread cannot be killed from outside. So the
 * number has to be here, before the button, rather than as a progress bar
 * afterwards.
 *
 * The state-space figure doubles as the clearest explanation of what the
 * quantum layer is doing — 8 qubits is 256 dimensions, and that is the whole
 * idea in one line.
 */
function seconds(s) {
  if (s == null) return "—";
  if (s < 90) return `${Math.round(s)} s`;
  const m = Math.floor(s / 60);
  return `${m} min ${Math.round(s - m * 60)} s`;
}

export default function CostStrip({ estimate, loading, unavailable }) {
  if (unavailable) {
    return (
      <div className="rounded-[10px] bg-hairline px-5 py-4 text-[11.5px] text-muted">
        Cost estimate is unavailable — the backend's estimate module is not
        wired. The run will still work; you just will not know how long it
        takes.
      </div>
    );
  }

  return (
    <div className="rounded-[10px] bg-quantum px-5 py-4 text-white">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="font-mono text-[34px] font-semibold leading-none">
            {estimate ? estimate.hilbert_dim.toLocaleString() : "—"}
          </p>
          <p className="mt-1.5 text-[11.5px] text-white/70">
            dimensions of state space
          </p>
        </div>
        <div className="text-right font-mono text-[11.5px] text-white/70">
          <p className="text-[12px] text-white">
            {estimate ? `${estimate.n_qubits} qubits` : "—"}
          </p>
          <p className="mt-1">
            {estimate ? `${estimate.sims.toLocaleString()} simulations` : "—"}
          </p>
          <p className="mt-0.5">
            {estimate ? `${estimate.mem_mb.toFixed(1)} MB state` : "—"}
          </p>
        </div>
      </div>

      <div className="mt-3 flex items-center justify-between border-t border-white/20 pt-3">
        <p className="flex items-center gap-2 text-[12.5px] font-semibold">
          {loading && <Spinner size={13} className="text-white/70" />}
          Estimated {seconds(estimate?.est_seconds)}
        </p>
        <p className="text-[11px] text-white/70">
          preprocessing included · no cancel once started
        </p>
      </div>
    </div>
  );
}
