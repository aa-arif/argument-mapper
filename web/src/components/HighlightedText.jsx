import { useEffect, useRef } from "react";
import { typeColors } from "../constants";

const wrapStyle = {
  fontFamily: "'Newsreader', 'Georgia', serif",
  fontSize: "14.5px",
  lineHeight: "1.8",
  letterSpacing: "0.01em",
};

function buildSegments(text, claimsList) {
  if (!claimsList.length) return [{ text, claim: null }];

  const sorted = [...claimsList]
    .filter((c) => c.sourceStart != null && c.sourceEnd != null)
    .sort(
      (a, b) =>
        a.sourceStart - b.sourceStart ||
        a.sourceEnd - a.sourceStart - (b.sourceEnd - b.sourceStart)
    );

  const ranges = [];
  sorted.forEach((claim) => {
    const start = Math.max(claim.sourceStart, 0);
    const end = Math.min(claim.sourceEnd, text.length);
    if (ranges.length > 0 && start <= ranges[ranges.length - 1].end) {
      const last = ranges[ranges.length - 1];
      last.end = Math.max(last.end, end);
      last.claims.push(claim);
    } else {
      ranges.push({ start, end, claims: [claim] });
    }
  });

  const segments = [];
  let cursor = 0;
  ranges.forEach((range) => {
    if (range.start > cursor) {
      segments.push({ text: text.slice(cursor, range.start), claim: null });
    }
    segments.push({ text: text.slice(range.start, range.end), claim: range.claims[0] });
    cursor = range.end;
  });
  if (cursor < text.length) {
    segments.push({ text: text.slice(cursor), claim: null });
  }
  return segments;
}

export default function HighlightedText({ text, selectedNode, claims, onClaimClick }) {
  const highlightRef = useRef(null);

  useEffect(() => {
    if (highlightRef.current) {
      highlightRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [selectedNode]);

  if (!text) return null;

  if (selectedNode) {
    const start = selectedNode.sourceStart;
    const end = selectedNode.sourceEnd;
    const color = typeColors[selectedNode.type];

    return (
      <div style={wrapStyle}>
        <span style={{ color: "#555" }}>{text.slice(0, start)}</span>
        <span
          ref={highlightRef}
          style={{
            color,
            backgroundColor: color + "1A",
            borderBottom: `2px solid ${color}`,
            padding: "2px 0",
            transition: "all 0.3s ease",
          }}
        >
          {text.slice(start, end)}
        </span>
        <span style={{ color: "#555" }}>{text.slice(end)}</span>
      </div>
    );
  }

  const segments = buildSegments(text, claims);

  return (
    <div style={wrapStyle}>
      {segments.map((seg, i) => {
        if (!seg.claim) {
          return (
            <span key={i} style={{ color: "#777" }}>
              {seg.text}
            </span>
          );
        }
        const color = typeColors[seg.claim.type];
        return (
          <span
            key={i}
            onClick={() => onClaimClick && onClaimClick(seg.claim)}
            style={{
              color: "#999",
              backgroundColor: color + "0D",
              borderBottom: `1px dashed ${color}40`,
              padding: "1px 0",
              cursor: onClaimClick ? "pointer" : "default",
              transition: "background 0.2s",
            }}
          >
            {seg.text}
          </span>
        );
      })}
    </div>
  );
}
