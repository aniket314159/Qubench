/**
 * One table, both kinds.
 *
 * The backend guarantees an identical telemetry key set for classical and
 * quantum models, with the quantum fields null rather than absent. That is why
 * this renders as one table instead of two — and why an em dash in the qubits
 * column is meaningful rather than missing data.
 */
const COLS = [
  ["n_qubits", "Qubits", (v) => v],
  ["encoding", "Encoding", (v) => v],
  // The bandwidth sits next to the encoding on purpose: the two together are
  // the feature map, and a fidelity kernel at the wrong bandwidth scores near
  // chance for reasons that have nothing to do with the encoding.
  ["bandwidth", "Bandwidth", (v) => (v === 1 ? "1.0" : String(v))],
  ["circuit_depth", "Depth", (v) => v],
  ["two_qubit_gates", "2-qubit gates", (v) => v],
  ["state_memory_mb", "State", (v) => `${v.toFixed(2)} MB`],
  ["n_params", "Params", (v) => v.toLocaleString()],
  ["predict_seconds", "Predict", (v) => `${(v * 1000).toFixed(0)} ms`],
  ["backend", "Simulator", (v) => v],
];

export default function TelemetryTable({ results }) {
  const ok = results.filter((r) => !r.error);
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-left">
        <thead>
          <tr className="border-b border-rule">
            <th className="py-2 pr-3 text-[10.5px] font-normal text-muted">
              Model
            </th>
            {COLS.map(([k, label]) => (
              <th
                key={k}
                className="px-2 py-2 text-right text-[10.5px] font-normal text-muted"
              >
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ok.map((r) => (
            <tr
              key={r.model}
              className={`border-b border-hairline last:border-0 ${
                r.kind === "quantum" ? "bg-quantum-soft/60" : ""
              }`}
            >
              <td className="py-2.5 pr-3 text-[12px] text-ink">{r.label}</td>
              {COLS.map(([k, , fmt]) => {
                const v = r.telemetry[k];
                return (
                  <td
                    key={k}
                    className="px-2 py-2.5 text-right font-mono text-[11.5px] text-body"
                  >
                    {v === null || v === undefined ? (
                      <span className="text-muted">—</span>
                    ) : (
                      fmt(v)
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-3 text-[11px] text-muted">
        Depth and gate counts are read from the circuit that actually ran.
        Bandwidth is the scale the kernel was cross-validated to, not a default.
        Simulation is exact — no shot noise, no hardware queue.
      </p>
    </div>
  );
}
