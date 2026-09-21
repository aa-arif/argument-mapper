import { useMemo } from "react";

import { componentColors, GOLD_COLOR } from "../constants";

// Builds a flat list of non-overlapping segments so the source text can be
// rendered once with spans, rather than nesting highlights.
//
// Predictions and gold annotations can overlap each other freely, so when both
// layers are on, gold is drawn as an underline and the prediction as a
// background. Two backgrounds would be unreadable where they disagree, which
// is exactly where the reader is looking.
function buildSegments(text, spans) {
  if (spans.length === 0) return [{ start: 0, end: text.length, spans: [] }];

  const boundaries = new Set([0, text.length]);
  spans.forEach(({ start, end }) => {
    boundaries.add(Math.max(0, Math.min(start, text.length)));
    boundaries.add(Math.max(0, Math.min(end, text.length)));
  });

  const ordered = [...boundaries].sort((a, b) => a - b);
  const segments = [];
  for (let i = 0; i < ordered.length - 1; i += 1) {
    const start = ordered[i];
    const end = ordered[i + 1];
    if (end <= start) continue;
    segments.push({
      start,
      end,
      spans: spans.filter((span) => span.start < end && span.end > start),
    });
  }
  return segments;
}

export default function HighlightedText({
  text,
  components = [],
  gold = [],
  selectedId,
  onSelect,
}) {
  const segments = useMemo(() => {
    const spans = [
      ...components.map((c) => ({ ...c, layer: "prediction" })),
      ...gold.map((c) => ({ ...c, layer: "gold" })),
    ];
    return buildSegments(text, spans);
  }, [text, components, gold]);

  if (!text) {
    return <p className="source-text source-text--empty">Paste a passage to begin.</p>;
  }

  return (
    <div className="source-text">
      {segments.map((segment) => {
        const body = text.slice(segment.start, segment.end);
        const prediction = segment.spans.find((s) => s.layer === "prediction");
        const isGold = segment.spans.some((s) => s.layer === "gold");

        if (!prediction && !isGold) {
          return <span key={segment.start}>{body}</span>;
        }

        const isSelected = prediction && prediction.id === selectedId;
        const color = prediction ? (componentColors[prediction.type] ?? "#888") : null;

        return (
          <span
            key={segment.start}
            className={[
              "span",
              prediction ? "span--prediction" : "",
              isGold ? "span--gold" : "",
              isSelected ? "is-selected" : "",
            ]
              .filter(Boolean)
              .join(" ")}
            style={{
              ...(color ? { "--span-color": color } : {}),
              ...(isGold ? { "--gold-color": GOLD_COLOR } : {}),
            }}
            onClick={prediction ? () => onSelect?.(prediction.id) : undefined}
            role={prediction ? "button" : undefined}
            tabIndex={prediction ? 0 : undefined}
            onKeyDown={
              prediction
                ? (event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      onSelect?.(prediction.id);
                    }
                  }
                : undefined
            }
          >
            {body}
          </span>
        );
      })}
    </div>
  );
}
