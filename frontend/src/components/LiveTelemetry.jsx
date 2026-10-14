/**
 * The circuit that is running, while it runs.
 *
 * Everything here is known before the first model finishes: qubit count, state
 * space, depth and gate count come from the encoding at this feature width,
 * and the simulation count and memory come from the cost model. After the run
 * the per-model TelemetryTable supersedes this with what each model actually
 * did.
 *
 * The last two cells are properties of the platform rather than of this run,
 * and they are the ones worth saying out loud: PennyLane's default.qubit is an
 * exact statevector simulator, so there is no shot noise and no hardware
 * queue. Saying "simulated exactly" is both accurate and the thing that keeps
 * the rest of the deck honest.
 */
function Cell({ label, value, accent = false }) {
  return (
    <div className="min-w-[96px] flex-1 px-1">
      <p className="text-[10.5px] text-muted">{label}</p>
      <p
        className={`mt-1 font-mono text-[17px] font-semibold ${
          accent ? "text-quantum" : "text-ink"
        }`}
      >
        {value}
      </p>
    </div>
  );
}

export default function LiveTelemetry({ circuit, estimate, telemetry }) {
  // Before any model finishes, `telemetry` is absent and every figure comes
  // from the encoding and the cost model. Once the winning model has run, its
  // real telemetry supersedes the estimate — and only then can the bandwidth
  // be shown at all, because QSVC cross-validates it during the fit.
  // Any ONE of the three sources is enough to fill some cells. Requiring
  // circuit-or-estimate hid the whole panel on a finished run reached from
  // the Leaderboard, where only `telemetry` is present.
  if (!circuit && !estimate && !telemetry) {
    return (
      <p className="text-[11.5px] text-muted">
        Circuit details appear once the encoding is known.
      </p>
    );
  }

  const dash = "\u2014";
  const q = circuit?.qubits ?? estimate?.n_qubits ?? telemetry?.n_qubits;
  const hilbert = estimate?.hilbert_dim ?? (q != null ? 2 ** q : null);

  return (
    <div>
      <div className="flex flex-wrap gap-y-4 divide-x divide-hairline">
        <Cell label="Qubits" value={q ?? dash} accent />
        <Cell
          label="State space"
          value={hilbert != null ? `${hilbert.toLocaleString()} d` : dash}
          accent
        />
        <Cell
          label="Circuit depth"
          value={circuit?.depth ?? telemetry?.circuit_depth ?? dash}
        />
        <Cell
          label="Two-qubit gates"
          value={circuit?.two_qubit_gates ?? telemetry?.two_qubit_gates ?? dash}
        />
        <Cell
          label="Simulations"
          value={estimate?.sims != null ? estimate.sims.toLocaleString() : dash}
        />
        <Cell
          label="State memory"
          value={
            telemetry?.state_memory_mb != null
              ? `${telemetry.state_memory_mb.toFixed(2)} MB`
              : estimate?.mem_mb != null
                ? `${estimate.mem_mb.toFixed(1)} MB`
                : dash
          }
        />
        {/* Only knowable after the fit: QSVC cross-validates the bandwidth, so
            mid-run there is no answer and a placeholder would later turn into
            a different number. */}
        <Cell
          label="Bandwidth"
          value={
            telemetry?.bandwidth != null ? String(telemetry.bandwidth) : dash
          }
          accent={telemetry?.bandwidth != null}
        />
        <Cell label="Shot noise" value="none" />
      </div>

      <p className="mt-4 border-t border-hairline pt-3 text-[11px] leading-snug text-body">
        Depth and gate counts are read from the decomposed circuit, not
        estimated. Simulation is exact — no shot noise, no hardware queue, no
        transpilation. Nothing here ran on a quantum computer, and the interface
        never claims otherwise.
      </p>
    </div>
  );
}
