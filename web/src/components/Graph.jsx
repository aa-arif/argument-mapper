import { useEffect, useRef, useCallback } from "react";
import ReactFlow, {
  MiniMap,
  Controls,
  useNodesState,
  useEdgesState,
  MarkerType,
} from "reactflow";
import "reactflow/dist/style.css";
import dagre from "@dagrejs/dagre";
import { typeColors, relColors, relIcons, NODE_WIDTH, NODE_HEIGHT } from "../constants";
import { nodeTypes } from "./ClaimNode";

function getLayoutedNodes(claims, relationships) {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "BT", nodesep: 80, ranksep: 100 });

  claims.forEach((claim) => {
    g.setNode(claim.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  });
  relationships.forEach((rel) => {
    g.setEdge(rel.source, rel.target);
  });

  dagre.layout(g);

  return claims.map((claim) => {
    const pos = g.node(claim.id);
    return {
      id: claim.id,
      type: "claimNode",
      position: { x: pos.x - NODE_WIDTH / 2, y: pos.y - NODE_HEIGHT / 2 },
      data: { ...claim, isSelected: false, isMatched: false },
      draggable: false,
    };
  });
}

function buildEdges(relationships, opts = {}) {
  const { matchedNodeIds } = opts;
  return relationships.map((rel, i) => {
    const strength = rel.strength ?? 0.5;
    const strokeWidth = 1 + strength * 3;
    const isDashed =
      matchedNodeIds &&
      matchedNodeIds.has(rel.source) &&
      matchedNodeIds.has(rel.target);

    return {
      id: `e-${rel.source}-${rel.target}-${i}`,
      source: rel.source,
      target: rel.target,
      animated: rel.type === "supports",
      style: {
        stroke: relColors[rel.type] || "#555",
        strokeWidth,
        opacity: 0.7,
        ...(isDashed ? { strokeDasharray: "6 3" } : {}),
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: relColors[rel.type] || "#555",
        width: 16,
        height: 16,
      },
      label: `${relIcons[rel.type] || ""} ${rel.type}`,
      labelStyle: {
        fontSize: "8px",
        fill: relColors[rel.type] || "#888",
        fontFamily: "'DM Sans', sans-serif",
      },
      labelBgStyle: {
        fill: "#0D0D0D",
        fillOpacity: 0.85,
      },
      labelBgPadding: [4, 2],
      labelBgBorderRadius: 3,
    };
  });
}

export default function Graph({
  data,
  selectedNode,
  onNodeSelect,
  layoutMode,
  nodeOpacityOverrides,
  matchedNodeIds,
}) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const reactFlowRef = useRef(null);

  useEffect(() => {
    if (!data.claims.length) return;

    const newEdges = buildEdges(data.relationships, { matchedNodeIds });

    if (layoutMode === "hierarchical") {
      setNodes(getLayoutedNodes(data.claims, data.relationships));
    } else {
      const n = data.claims.length;
      const radius = Math.max(150, n * 40);
      setNodes(
        data.claims.map((claim, i) => {
          const angle = (2 * Math.PI * i) / n - Math.PI / 2;
          return {
            id: claim.id,
            type: "claimNode",
            position: {
              x: 400 + radius * Math.cos(angle) - NODE_WIDTH / 2,
              y: 300 + radius * Math.sin(angle) - NODE_HEIGHT / 2,
            },
            data: { ...claim, isSelected: false, isMatched: false },
            draggable: true,
          };
        })
      );
    }

    setEdges(newEdges);

    setTimeout(() => {
      reactFlowRef.current?.fitView({ padding: 0.2, duration: 300 });
    }, 50);
  }, [data, layoutMode, matchedNodeIds]);

  useEffect(() => {
    setNodes((nds) =>
      nds.map((n) => {
        const opacity = nodeOpacityOverrides?.[n.id];
        const isMatched = matchedNodeIds?.has(n.id) ?? false;
        return {
          ...n,
          data: { ...n.data, isSelected: selectedNode?.id === n.id, isMatched },
          style: opacity != null ? { opacity } : undefined,
        };
      })
    );
  }, [selectedNode, nodeOpacityOverrides, matchedNodeIds, setNodes]);

  const onNodeClick = useCallback(
    (_, node) => onNodeSelect(node.data),
    [onNodeSelect]
  );

  const onPaneClick = useCallback(
    () => onNodeSelect(null),
    [onNodeSelect]
  );

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      nodeTypes={nodeTypes}
      onNodeClick={onNodeClick}
      onPaneClick={onPaneClick}
      onInit={(instance) => { reactFlowRef.current = instance; }}
      nodesDraggable={layoutMode === "free"}
      fitView
      fitViewOptions={{ padding: 0.2 }}
      minZoom={0.25}
      maxZoom={4}
      proOptions={{ hideAttribution: true }}
      style={{ background: "transparent" }}
    >
      <MiniMap
        style={{
          background: "#141414",
          border: "1px solid #2A2A2A",
          borderRadius: "6px",
        }}
        nodeColor={(node) => typeColors[node.data?.type] || "#666"}
        maskColor="#0D0D0D80"
        position="bottom-left"
      />
      <Controls position="bottom-right" showInteractive={false} />
    </ReactFlow>
  );
}
