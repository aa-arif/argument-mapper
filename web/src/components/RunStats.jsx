/**
 * What the last extraction cost.
 *
 * Shown in the UI rather than only in the results files because the whole
 * question this project asks is quality *per dollar and per second* -- a graph
 * with no price attached hides half the comparison.
 */
export default function RunStats({ meta }) {
  if (!meta) return null;

  const dollars = meta.usage?.dollars ?? 0;
  return (
    <dl className="run-stats">
      <div>
        <dt>Model</dt>
        <dd>{meta.model}</dd>
      </div>
      <div>
        <dt>Latency</dt>
        <dd>{meta.cached ? "cached" : `${(meta.latencyMs / 1000).toFixed(2)}s`}</dd>
      </div>
      <div>
        <dt>Cost</dt>
        <dd>{dollars > 0 ? `$${dollars.toFixed(4)}` : "—"}</dd>
      </div>
      {meta.escalated != null && (
        <div>
          <dt>Cascade</dt>
          <dd>{meta.escalated ? "escalated" : "handled locally"}</dd>
        </div>
      )}
    </dl>
  );
}
