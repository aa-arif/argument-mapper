import { ROUTES } from "../constants";

/**
 * Pick which model produces the graph.
 *
 * Routes the server reports as unavailable are disabled rather than hidden:
 * seeing that a local model exists but is not running is more useful than
 * silently offering three options when only one works.
 */
export default function ModelSelector({ value, onChange, availability, disabled }) {
  return (
    <div className="model-selector" role="radiogroup" aria-label="Extraction model">
      {ROUTES.map((route) => {
        const key = `${route.id}:${route.model}`;
        const isAvailable = availability?.[route.id] ?? true;
        const isActive = value === key;
        return (
          <button
            key={key}
            type="button"
            role="radio"
            aria-checked={isActive}
            className={`model-option ${isActive ? "is-active" : ""}`}
            disabled={disabled || !isAvailable}
            title={isAvailable ? route.detail : `${route.detail} — not configured on this server`}
            onClick={() => onChange(key)}
          >
            <span className="model-option__label">{route.label}</span>
            <span className="model-option__detail">
              {isAvailable ? route.detail : "unavailable"}
            </span>
          </button>
        );
      })}
    </div>
  );
}
