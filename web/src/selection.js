// Model-selection helpers.
//
// Kept out of ModelSelector.jsx so that file exports a component and nothing
// else: mixing component and non-component exports breaks Fast Refresh.
import { ROUTES } from "./constants";

/** Split a "route:model" key back into its parts. */
export function parseSelection(key) {
  const [route, model] = key.split(":");
  return { route, model };
}

export const DEFAULT_SELECTION = `${ROUTES[0].id}:${ROUTES[0].model}`;
