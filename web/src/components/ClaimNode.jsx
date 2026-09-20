import { Handle, Position } from "reactflow";
import { typeColors, typeLabels } from "../constants";

function ConfidenceArc({ confidence, color, size = 56 }) {
  const r = size / 2 + 3;
  const circumference = Math.PI * r;
  const filled = circumference * confidence;
  const cx = size / 2;
  const cy = size / 2;

  return (
    <svg
      width={size + 8}
      height={size + 8}
      style={{ position: "absolute", top: -4, left: -4, pointerEvents: "none" }}
    >
      <path
        d={`M ${cx + 4 - r} ${cy + 4} A ${r} ${r} 0 0 0 ${cx + 4 + r} ${cy + 4}`}
        fill="none"
        stroke={color}
        strokeOpacity={0.15}
        strokeWidth={2.5}
        strokeLinecap="round"
      />
      <path
        d={`M ${cx + 4 - r} ${cy + 4} A ${r} ${r} 0 0 0 ${cx + 4 + r} ${cy + 4}`}
        fill="none"
        stroke={color}
        strokeOpacity={0.4}
        strokeWidth={2.5}
        strokeLinecap="round"
        strokeDasharray={`${filled} ${circumference}`}
      />
    </svg>
  );
}

function ClaimNodeComponent({ data }) {
  const color = typeColors[data.type] || "#666";
  const isSelected = data.isSelected;
  const isMatched = data.isMatched;
  const confidence = data.confidence ?? 0.5;

  const maxChars = 60;
  const maxLineChars = 24;
  const maxLines = 3;
  const truncated = data.claim.length > maxChars ? data.claim.slice(0, maxChars) + "\u2026" : data.claim;
  const words = truncated.split(/\s+/);
  const lines = [];
  let currentLine = "";
  words.forEach((word) => {
    const testLine = currentLine ? currentLine + " " + word : word;
    if (testLine.length > maxLineChars && currentLine) {
      lines.push(currentLine);
      currentLine = word;
    } else {
      currentLine = testLine;
    }
  });
  if (currentLine) lines.push(currentLine);
  const displayLines = lines.slice(0, maxLines);
  if (lines.length > maxLines) {
    displayLines[maxLines - 1] = displayLines[maxLines - 1].slice(0, -1) + "\u2026";
  }

  const borderColor = isMatched ? "#ffffff90" : color;
  const glowShadow = isSelected
    ? `0 0 20px ${color}60`
    : isMatched
      ? "0 0 12px #ffffff40"
      : "none";

  return (
    <>
      <Handle type="target" position={Position.Bottom} style={{ opacity: 0, pointerEvents: "none" }} />
      <div style={{
        display: "flex", flexDirection: "column", alignItems: "center",
        padding: "8px 12px", minHeight: "120px", cursor: "pointer",
      }}>
        <div style={{ position: "relative" }}>
          <ConfidenceArc confidence={confidence} color={color} />
          <div style={{
            width: 56, height: 56, borderRadius: "50%",
            border: `2px solid ${borderColor}`,
            backgroundColor: isSelected ? `${color}47` : `${color}1F`,
            display: "flex", flexDirection: "column",
            alignItems: "center", justifyContent: "center",
            boxShadow: glowShadow,
            transition: "all 0.2s ease",
          }}>
            <div style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: "11px", fontWeight: 600,
              color, lineHeight: 1,
            }}>
              {data.id.toUpperCase()}
            </div>
            <div style={{
              fontFamily: "'DM Sans', sans-serif",
              fontSize: "7.5px", color,
              opacity: 0.6, marginTop: "2px",
            }}>
              {typeLabels[data.type] || data.type}
            </div>
          </div>
        </div>
        <div style={{
          marginTop: "6px", textAlign: "center",
          fontFamily: "'DM Sans', sans-serif",
          fontSize: "9px", color: "#555",
          lineHeight: "1.3", maxWidth: "140px",
        }}>
          {displayLines.map((line, i) => <div key={i}>{line}</div>)}
        </div>
      </div>
      <Handle type="source" position={Position.Top} style={{ opacity: 0, pointerEvents: "none" }} />
    </>
  );
}

export default ClaimNodeComponent;
