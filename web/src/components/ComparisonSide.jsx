import { useMemo } from "react";
import Graph from "./Graph";
import TextInputPanel from "./TextInputPanel";

export default function ComparisonSide({
  side,
  text,
  onTextChange,
  onExtract,
  onFileUpload,
  processing,
  data,
  selectedNode,
  onNodeSelect,
  layoutMode,
  diff,
}) {
  const matchedSet = useMemo(() => {
    if (!diff) return null;
    return side === "left" ? diff.leftMatched : diff.rightMatched;
  }, [diff, side]);

  const nodeOpacityOverrides = useMemo(() => {
    if (!matchedSet || !data.claims.length) return undefined;
    const overrides = {};
    data.claims.forEach((c) => {
      overrides[c.id] = matchedSet.has(c.id) ? 1.0 : 0.5;
    });
    return overrides;
  }, [matchedSet, data.claims]);

  const hasData = data.claims.length > 0;

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <div style={{
        padding: "6px 12px", borderBottom: "1px solid #1A1A1A",
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <span style={{
          fontSize: "9px", textTransform: "uppercase",
          letterSpacing: "0.1em", color: "#555", fontWeight: 600,
        }}>
          {side === "left" ? "Text A" : "Text B"}
        </span>
      </div>

      {!hasData ? (
        <div style={{ display: "flex", flexDirection: "column", flex: 1 }}>
          <TextInputPanel
            text={text}
            onTextChange={onTextChange}
            onExtract={onExtract}
            onFileUpload={onFileUpload}
            processing={processing}
          />
        </div>
      ) : (
        <div style={{ flex: 1, position: "relative", overflow: "hidden" }}>
          <Graph
            data={data}
            selectedNode={selectedNode}
            onNodeSelect={onNodeSelect}
            layoutMode={layoutMode}
            nodeOpacityOverrides={nodeOpacityOverrides}
            matchedNodeIds={matchedSet}
          />
        </div>
      )}
    </div>
  );
}
