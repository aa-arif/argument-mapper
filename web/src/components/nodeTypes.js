// React Flow node-type registry.
//
// Kept out of ClaimNode.jsx so that file exports a component and nothing else:
// mixing a component export with a constant export breaks Fast Refresh, which
// is what react-refresh/only-export-components warns about.
import ClaimNodeComponent from "./ClaimNode";

export const nodeTypes = { claimNode: ClaimNodeComponent };
