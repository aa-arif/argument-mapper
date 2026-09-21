import dagre from "@dagrejs/dagre";
import { useCallback, useEffect } from "react";
import ReactFlow, {
  Controls,
  MarkerType,
  MiniMap,
  useEdgesState,
  useNodesState,
} from "reactflow";
import "reactflow/dist/style.css";

import {
  componentColors,
  NODE_HEIGHT,
  NODE_WIDTH,
  relationColors,
} from "../constants";
import { nodeTypes } from "./nodeTypes";

// Bottom-to-top: premises sit below the claims they support, so the major
// claim rises to the top. That matches how the annotation scheme is taught
// and makes an inverted edge visually obvious.
const LAYOUT = { rankdir: "BT", nodesep: 70, ranksep: 90 };

function layout(components, relations) {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph(LAYOUT);

  components.forEach((component) => {
    g.setNode(component.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  });
  relations.forEach((relation) => {
    // dagre throws on an edge to a node it does not know about.
    if (g.hasNode(relation.src) && g.hasNode(relation.tgt)) {
      g.setEdge(relation.src, relation.tgt);
    }
  });

  dagre.layout(g);
  return g;
}

function buildNodes(components, relations, { selectedId, matchedIds }) {
  const positions = layout(components, relations);
  return components.map((component) => {
    const node = positions.node(component.id);
    return {
      id: component.id,
      type: "componentNode",
      position: {
        x: (node?.x ?? 0) - NODE_WIDTH / 2,
        y: (node?.y ?? 0) - NODE_HEIGHT / 2,
      },
      data: {
        ...component,
        isSelected: component.id === selectedId,
        isMatched: matchedIds?.has(component.id) ?? false,
      },
      draggable: true,
    };
  });
}

function buildEdges(relations, { selectedId }) {
  return relations.map((relation, index) => {
    const isAdjacent = selectedId === relation.src || selectedId === relation.tgt;
    const color = relationColors[relation.type] ?? "#666";
    return {
      id: relation.id ?? `e-${relation.src}-${relation.tgt}-${index}`,
      source: relation.src,
      target: relation.tgt,
      animated: isAdjacent,
      style: {
        stroke: color,
        strokeWidth: isAdjacent ? 2.5 : 1.5,
        opacity: selectedId && !isAdjacent ? 0.25 : 0.8,
      },
      markerEnd: { type: MarkerType.ArrowClosed, color },
    };
  });
}

export default function Graph({
  components,
  relations,
  selectedId,
  matchedIds,
  onSelect,
  fitViewKey,
}) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  useEffect(() => {
    setNodes(buildNodes(components, relations, { selectedId, matchedIds }));
    setEdges(buildEdges(relations, { selectedId }));
  }, [components, relations, selectedId, matchedIds, setNodes, setEdges]);

  const handleNodeClick = useCallback(
    (_event, node) => {
      onSelect?.(node.id === selectedId ? null : node.id);
    },
    [onSelect, selectedId],
  );

  if (components.length === 0) {
    return (
      <div className="graph-empty">
        <p>No argument components yet.</p>
      </div>
    );
  }

  return (
    <ReactFlow
      key={fitViewKey}
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeClick={handleNodeClick}
      onPaneClick={() => onSelect?.(null)}
      nodeTypes={nodeTypes}
      fitView
      minZoom={0.2}
      proOptions={{ hideAttribution: true }}
    >
      <Controls showInteractive={false} />
      <MiniMap
        pannable
        zoomable
        nodeColor={(node) => componentColors[node.data?.type] ?? "#888"}
        nodeStrokeWidth={0}
        // maskColor is the *unviewed* area; leaving it at the light default
        // puts a grey slab over a dark canvas.
        maskColor="rgba(10, 10, 12, 0.75)"
        style={{
          background: "#1b1b1f",
          border: "1px solid #2e2e36",
          borderRadius: 6,
        }}
      />
    </ReactFlow>
  );
}
