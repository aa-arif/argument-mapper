import { Handle, Position } from "reactflow";

import { componentAbbrev, componentColors, componentLabels } from "../constants";

// Long components are common: an essay premise is often a full sentence.
// Truncating in the node keeps the graph readable; the inspector shows the
// full text.
const PREVIEW_CHARS = 110;

function preview(text) {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length <= PREVIEW_CHARS ? clean : `${clean.slice(0, PREVIEW_CHARS - 1)}…`;
}

function ComponentNode({ data }) {
  const color = componentColors[data.type] ?? "#888";
  const classes = [
    "component-node",
    data.isSelected ? "is-selected" : "",
    data.isMatched ? "is-matched" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <>
      <Handle type="target" position={Position.Top} className="node-handle" />
      <div className={classes} style={{ "--node-color": color }}>
        <div className="component-node__header">
          <span className="component-node__badge" title={componentLabels[data.type] ?? data.type}>
            {componentAbbrev[data.type] ?? "?"}
          </span>
          <span className="component-node__id">{data.id}</span>
        </div>
        <p className="component-node__text">{preview(data.text ?? "")}</p>
      </div>
      <Handle type="source" position={Position.Bottom} className="node-handle" />
    </>
  );
}

export default ComponentNode;
