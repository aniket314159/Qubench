import { useEffect, useRef, useState } from "react";

/**
 * The circuit that will actually run, drawn from the backend's decomposed op
 * list.
 *
 * This component knows NOTHING about encodings. It receives
 * `{ qubits, ops: [{name, wires}] }` — the real PennyLane circuit after device
 * decomposition — and lays it out. An encoding added in Phase 2 draws
 * correctly on the day it ships, and editing a template can never leave this
 * picture stale, because there is no second description of the circuit here.
 *
 * Two-qubit gates render as a control dot and a target joined by a vertical
 * line; single-qubit gates as a labelled box. Columns are assigned greedily:
 * a gate goes in the earliest column where none of its wires are busy, which
 * is the same rule that produces circuit depth — so the drawing's width IS the
 * depth.
 *
 * Pass `zoomable` and the drawing becomes a button: clicking it lifts the same
 * circuit into a centred panel, which is the only way to read a deep circuit
 * in a column this narrow. Nothing about the drawing changes there — it is
 * scaled, not redrawn.
 */
const SINGLE = { Hadamard: "H", PauliX: "X", PauliY: "Y", PauliZ: "Z" };

function layout(ops, qubits) {
  const busy = Array(qubits).fill(0); // next free column per wire
  return ops.map((op) => {
    const wires = op.wires.length ? op.wires : [0];
    const lo = Math.min(...wires);
    const hi = Math.max(...wires);
    let col = 0;
    for (let w = lo; w <= hi; w++) col = Math.max(col, busy[w] ?? 0);
    for (let w = lo; w <= hi; w++) busy[w] = col + 1;
    return { ...op, col, lo, hi, multi: wires.length > 1 };
  });
}

const COL = 34,
  ROW = 30,
  PAD_L = 34,
  PAD_R = 12,
  PAD_Y = 16;

/** How much larger the enlarged circuit may get. Above this the drawing stops
 *  being a circuit and starts being a wall. */
const MAX_ZOOM = 2.5;

/**
 * The drawing on its own.
 *
 * Split out so the zoomed view can show the identical circuit without a second
 * copy of this markup drifting away from the first. Geometry is passed in, so
 * the inline drawing and the enlarged one cannot disagree about it.
 */
function Drawing({ qubits, placed, w, h, cx, cy, depth, className, style }) {
  return (
    <svg
      width={w}
      height={h}
      viewBox={`0 0 ${w} ${h}`}
      role="img"
      aria-label={`Quantum circuit on ${qubits} qubits, depth ${depth}`}
      className={className}
      style={style}
    >
      {/* wires */}
      {Array.from({ length: qubits }, (_, r) => (
        <g key={r}>
          <text
            x={PAD_L - 10}
            y={cy(r) + 3.5}
            textAnchor="end"
            className="fill-[var(--color-muted)] font-mono"
            style={{ fontSize: 9.5 }}
          >
            q{r}
          </text>
          <line
            x1={PAD_L}
            y1={cy(r)}
            x2={w - PAD_R}
            y2={cy(r)}
            stroke="var(--color-rule)"
            strokeWidth="1"
          />
        </g>
      ))}

      {placed.map((op, i) => {
        const x = cx(op.col);
        if (op.multi) {
          const [ctrl, targ] = op.wires;
          return (
            <g key={i}>
              <line
                x1={x}
                y1={cy(op.lo)}
                x2={x}
                y2={cy(op.hi)}
                stroke="var(--color-quantum)"
                strokeWidth="1.4"
              />
              <circle cx={x} cy={cy(ctrl)} r="3.4" fill="var(--color-quantum)" />
              {op.name === "CNOT" ? (
                <>
                  <circle
                    cx={x}
                    cy={cy(targ)}
                    r="6.5"
                    fill="none"
                    stroke="var(--color-quantum)"
                    strokeWidth="1.4"
                  />
                  <line
                    x1={x - 6.5}
                    y1={cy(targ)}
                    x2={x + 6.5}
                    y2={cy(targ)}
                    stroke="var(--color-quantum)"
                    strokeWidth="1.4"
                  />
                </>
              ) : (
                <circle cx={x} cy={cy(targ)} r="3.4" fill="var(--color-quantum)" />
              )}
            </g>
          );
        }
        const label = SINGLE[op.name] ?? op.name;
        const bw = Math.max(20, label.length * 7 + 8);
        return (
          <g key={i}>
            <rect
              x={x - bw / 2}
              y={cy(op.lo) - 9}
              width={bw}
              height={18}
              rx="3.5"
              fill="var(--color-quantum-soft)"
              stroke="var(--color-quantum)"
              strokeWidth="1"
            />
            <text
              x={x}
              y={cy(op.lo) + 3.5}
              textAnchor="middle"
              className="fill-[var(--color-quantum)] font-mono"
              style={{ fontSize: 9.5, fontWeight: 600 }}
            >
              {label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export default function QuantumCircuit({
  qubits,
  ops,
  depth,
  twoQubitGates,
  truncated,
  encodingLabel,
  zoomable = false,
}) {
  const [zoomed, setZoomed] = useState(false);
  const panel = useRef(null);
  const trigger = useRef(null);

  const drawable = !!qubits && !!ops?.length;
  // Derived rather than stored: if the encoding changes while the panel is
  // open, this closes with it. A separate flag could leave the page unable to
  // scroll with nothing on screen.
  const open = zoomed && drawable;

  useEffect(() => {
    if (!open) return;
    const onKey = (e) => {
      if (e.key === "Escape") setZoomed(false);
    };
    document.addEventListener("keydown", onKey);
    // The page behind must hold still while the panel is up, or a stray scroll
    // wheel moves the circuit the reader is looking at.
    const before = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const back = trigger.current;
    panel.current?.focus();
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = before;
      // Keyboard users land where they started, not on a dead body element.
      back?.focus();
    };
  }, [open]);

  if (!drawable) {
    return (
      <p className="text-[11.5px] text-muted">
        No circuit to draw for this encoding.
      </p>
    );
  }

  const placed = layout(ops, qubits);
  const cols = Math.max(...placed.map((o) => o.col)) + 1;

  const w = PAD_L + cols * COL + PAD_R;
  const h = PAD_Y * 2 + (qubits - 1) * ROW;
  const cx = (c) => PAD_L + c * COL + COL / 2;
  const cy = (r) => PAD_Y + r * ROW;

  const gates = `${twoQubitGates ?? "—"} two-qubit gate${
    twoQubitGates === 1 ? "" : "s"
  }`;

  return (
    <div>
      <div
        ref={trigger}
        onClick={zoomable ? () => setZoomed(true) : undefined}
        onKeyDown={
          zoomable
            ? (e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  setZoomed(true);
                }
              }
            : undefined
        }
        role={zoomable ? "button" : undefined}
        tabIndex={zoomable ? 0 : undefined}
        aria-label={zoomable ? "Enlarge the circuit" : undefined}
        className={zoomable ? "cursor-zoom-in rounded-[7px]" : undefined}
      >
        <div className="flex items-baseline justify-between gap-3">
          <p className="font-mono text-[11px] text-muted">{encodingLabel}</p>
          <p className="font-mono text-[10.5px] font-semibold text-quantum">
            depth {depth ?? "—"} · {gates}
          </p>
        </div>

        <div className="mt-2 overflow-x-auto">
          <Drawing
            qubits={qubits}
            placed={placed}
            w={w}
            h={h}
            cx={cx}
            cy={cy}
            depth={depth}
            className="max-w-full"
          />
        </div>
      </div>

      {truncated && (
        <p className="mt-2 text-[10.5px] text-muted">
          Showing the first {ops.length} gates — the full circuit is wider than
          this panel can usefully draw.
        </p>
      )}

      {open && (
        // Clicking the backdrop closes; the panel below stops that event, so
        // the circuit can be scrolled or read without dismissing it.
        <div
          onClick={() => setZoomed(false)}
          className="fixed inset-0 z-50 flex items-center justify-center bg-navy/40 p-6"
        >
          <div
            ref={panel}
            tabIndex={-1}
            role="dialog"
            aria-modal="true"
            aria-label={`${encodingLabel ?? "Quantum circuit"} enlarged`}
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-[1100px] rounded-[10px] border border-rule bg-white px-6 py-5 outline-none"
          >
            <div className="flex items-baseline justify-between gap-3">
              <p className="font-mono text-[12px] text-muted">{encodingLabel}</p>
              <p className="font-mono text-[11px] font-semibold text-quantum">
                depth {depth ?? "—"} · {gates}
              </p>
            </div>

            <div className="mt-3 max-h-[75vh] overflow-auto">
              <Drawing
                qubits={qubits}
                placed={placed}
                w={w}
                h={h}
                cx={cx}
                cy={cy}
                depth={depth}
                // Scales the whole drawing through its viewBox, so the labels
                // grow with it. max(100%, w) upscales a narrow circuit to the
                // panel and leaves a deep one at its natural size to scroll,
                // rather than shrinking either to unreadable.
                style={{
                  width: `max(100%, ${w}px)`,
                  maxWidth: w * MAX_ZOOM,
                  height: "auto",
                }}
              />
            </div>

            <p className="mt-4 border-t border-hairline pt-3 text-[11.5px] text-body">
              Click outside this panel to put it back, or press Escape.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
