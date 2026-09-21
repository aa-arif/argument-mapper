// React Flow node-type registry.
//
// Kept out of ComponentNode.jsx so that file exports a component and nothing
// else: mixing a component export with a constant export breaks Fast Refresh.
import ComponentNode from "./ComponentNode";

export const nodeTypes = { componentNode: ComponentNode };
