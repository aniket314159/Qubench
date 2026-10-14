/**
 * Rows are the true class, columns the prediction.
 *
 * The matrix is the honest version of the headline number: an 0.76 accuracy
 * that never once predicts the rarest class looks fine in a table and obvious
 * here. Diagonal cells are tinted by how full they are; off-diagonal cells get
 * a warm tint so mistakes are findable without reading every number.
 */
export default function ConfusionMatrix({ result, classNames }) {
  const cm = result.confusion_matrix ?? [];
  if (!cm.length)
    return (
      <p className="text-[11.5px] text-muted">No matrix for this model.</p>
    );

  const total = cm.flat().reduce((a, b) => a + b, 0);
  const correct = cm.reduce((a, row, i) => a + (row[i] ?? 0), 0);
  const peak = Math.max(...cm.map((row, i) => row[i] ?? 0), 1);
  const short = (n) => (n.length > 6 ? `${n.slice(0, 5)}…` : n);

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="border-collapse">
          <thead>
            <tr>
              <th />
              {classNames.map((n) => (
                <th
                  key={n}
                  className="px-1 pb-1 text-center text-[10.5px] font-normal text-muted"
                >
                  {short(n)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {cm.map((row, i) => (
              <tr key={classNames[i] ?? i}>
                <th className="pr-2 text-right text-[10.5px] font-normal text-muted">
                  {short(classNames[i] ?? String(i))}
                </th>
                {row.map((v, j) => {
                  const diag = i === j;
                  const strength = diag ? v / peak : 0;
                  return (
                    <td key={j} className="p-[2px]">
                      <div
                        title={`${v} ${classNames[i]} predicted as ${classNames[j]}`}
                        className={`grid h-9 w-9 place-items-center rounded-[3px] font-mono text-[12px]
                          ${
                            diag
                              ? strength > 0.55
                                ? "text-white"
                                : "text-ink"
                              : v === 0
                                ? "bg-hairline text-muted"
                                : "bg-alert/10 text-ink"
                          }`}
                        style={
                          diag
                            ? {
                                backgroundColor: `rgba(91,43,217,${0.12 + strength * 0.8})`,
                              }
                            : undefined
                        }
                      >
                        {v}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-3 border-t border-hairline pt-3 text-[11.5px] text-body">
        {total - correct} mistakes out of {total} test images
      </p>
    </div>
  );
}
