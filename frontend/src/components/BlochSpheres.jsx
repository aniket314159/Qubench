/**
 * One patient, as qubits.
 *
 * Each sphere is a single qubit after the encoding's first data layer, with
 * the arrow at (theta, phi) as the backend computed it — including the model's
 * tuned bandwidth, so the rotation drawn is the rotation that ran.
 *
 * The projection is deliberately plain: a circle, an equator ellipse, a
 * vertical axis. It is not trying to be a 3D render. The honest thing this can
 * show is "these eight numbers became eight different orientations", and a
 * prettier sphere would not say more.
 *
 * Amplitude encoding returns no angles at all, and this renders the reason
 * rather than an empty box: its qubits hold one joint superposition, so eight
 * independent arrows would be a picture of something that is not happening.
 */
function Sphere({ theta, phi, index, r = 26 }) {
  const cx = r + 4,
    cy = r + 4,
    size = (r + 4) * 2;
  // Standard Bloch projection: polar angle tilts away from +Z, azimuth
  // foreshortens across the equator.
  const x = cx + r * 0.88 * Math.sin(theta) * Math.cos(phi);
  const y = cy - r * 0.88 * Math.cos(theta);
  const depth = Math.sin(theta) * Math.sin(phi); // −1 behind, +1 in front

  return (
    <li className="flex flex-col items-center">
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        aria-hidden="true"
      >
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill="white"
          stroke="var(--color-rule)"
          strokeWidth="1"
        />
        <ellipse
          cx={cx}
          cy={cy}
          rx={r}
          ry={r * 0.3}
          fill="none"
          stroke="var(--color-hairline)"
          strokeWidth="1"
        />
        <line
          x1={cx}
          y1={cy - r}
          x2={cx}
          y2={cy + r}
          stroke="var(--color-hairline)"
          strokeWidth="1"
        />
        <line
          x1={cx}
          y1={cy}
          x2={x}
          y2={y}
          stroke="var(--color-quantum)"
          strokeWidth="1.8"
          strokeLinecap="round"
          // A vector pointing away from the viewer is drawn faded rather than
          // hidden, so a sphere never looks broken.
          opacity={depth < 0 ? 0.45 : 1}
        />
        <circle
          cx={x}
          cy={y}
          r="2.8"
          fill="var(--color-quantum)"
          opacity={depth < 0 ? 0.45 : 1}
        />
      </svg>
      <p className="mt-1 font-mono text-[9.5px] text-muted">
        q{index} {Math.round((theta * 180) / Math.PI)}°
      </p>
    </li>
  );
}

export default function BlochSpheres({
  angles,
  encodingLabel,
  bandwidth,
  caption,
  columns = 4,
  missingHint = true,
}) {
  // Three states, deliberately distinguished. `null` is a real answer from the
  // backend; `undefined` means it never sent the field, which is a setup
  // problem and should say so rather than look like an encoding limitation.
  if (angles === undefined) {
    return (
      <p className="text-[11.5px] leading-snug text-caution">
        {missingHint
          ? "No qubit angles arrived. The backend may predate the field — restart it, and re-run scripts/gen_fixtures.py if you are in mock mode."
          : "No qubit angles available for this encoding yet."}
      </p>
    );
  }

  if (angles === null || angles.length === 0) {
    return (
      <p className="text-[11.5px] leading-snug text-muted">
        This encoding has no per-qubit reading. Its qubits hold one joint
        superposition rather than independent rotations, so a row of arrows
        would show something that is not happening.
      </p>
    );
  }

  return (
    <div>
      <ul
        className="grid gap-x-2 gap-y-3"
        style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
      >
        {angles.map((a, i) => (
          <Sphere key={i} theta={a.theta} phi={a.phi} index={i} />
        ))}
      </ul>
      <p className="mt-3 text-[11px] leading-snug text-body">
        {caption ?? (
          <>
            {angles.length} feature{angles.length === 1 ? "" : "s"} became{" "}
            {angles.length} rotation{angles.length === 1 ? "" : "s"} under{" "}
            <span className="font-mono">{encodingLabel}</span>
            {bandwidth != null && bandwidth !== 1 && (
              <>, scaled by the tuned bandwidth {bandwidth}</>
            )}
            .
          </>
        )}
      </p>
    </div>
  );
}
