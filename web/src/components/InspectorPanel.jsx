import { useMemo } from "react";
import { typeColors, typeLabels, relColors, relIcons, framingColors } from "../constants";

function StrengthBar({ value, color }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "6px", marginTop: "4px" }}>
      <span style={{ fontSize: "8px", color: "#555", fontFamily: "'DM Sans', sans-serif", minWidth: "42px" }}>
        Strength
      </span>
      <div style={{
        flex: 1, height: "3px", background: "#2A2A2A", borderRadius: "2px", overflow: "hidden",
      }}>
        <div style={{
          width: `${Math.round(value * 100)}%`, height: "100%",
          background: color, borderRadius: "2px",
          transition: "width 0.3s ease",
        }} />
      </div>
      <span style={{ fontSize: "8px", color: "#555", fontFamily: "'JetBrains Mono', monospace", minWidth: "24px" }}>
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  );
}

function ReasoningBadge({ type }) {
  return (
    <span style={{
      fontSize: "7.5px", textTransform: "uppercase", letterSpacing: "0.06em",
      color: "#999", background: "#1E1E1E", padding: "2px 5px",
      borderRadius: "3px", fontFamily: "'JetBrains Mono', monospace",
    }}>
      {type}
    </span>
  );
}

function getDirectionLabel(relType, isSource) {
  if (isSource) {
    if (relType === "depends_on") return "This node depends on";
    return `This node ${relType}`;
  }
  const targetLabels = {
    supports: "Supported by",
    opposes: "Opposed by",
    refines: "Refined by",
    depends_on: "Depended on by",
  };
  return targetLabels[relType] || `${relType} by`;
}

function BiasAssessmentPanel({ bias }) {
  if (!bias) return null;

  return (
    <div style={{ marginTop: "16px" }}>
      <div style={{
        fontSize: "10px", textTransform: "uppercase",
        letterSpacing: "0.1em", color: "#555",
        fontWeight: 600, marginBottom: "8px",
      }}>
        Bias Assessment
      </div>

      <div style={{
        background: "#141414", borderRadius: "6px", padding: "10px",
        marginBottom: "6px",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "6px" }}>
          <span style={{ fontSize: "9px", color: "#555", textTransform: "uppercase", letterSpacing: "0.06em" }}>
            Framing
          </span>
          <span style={{
            fontSize: "8px", textTransform: "uppercase", letterSpacing: "0.06em",
            color: framingColors[bias.framing] || "#777",
            background: (framingColors[bias.framing] || "#777") + "20",
            padding: "2px 6px", borderRadius: "3px",
            fontWeight: 600, fontFamily: "'JetBrains Mono', monospace",
          }}>
            {bias.framing}
          </span>
        </div>

        {bias.dominant_perspective && (
          <div style={{
            fontSize: "11px", color: "#888", lineHeight: "1.4",
            fontFamily: "'Newsreader', serif", fontStyle: "italic",
            marginBottom: "6px",
          }}>
            {bias.dominant_perspective}
          </div>
        )}

        {bias.underrepresented_perspectives?.length > 0 && (
          <div>
            <div style={{
              fontSize: "8px", textTransform: "uppercase", letterSpacing: "0.06em",
              color: "#555", marginBottom: "4px",
            }}>
              Underrepresented
            </div>
            {bias.underrepresented_perspectives.map((p, i) => (
              <div key={i} style={{
                fontSize: "10px", color: "#777", lineHeight: "1.4",
                paddingLeft: "8px", borderLeft: "1px solid #2A2A2A",
                marginBottom: "3px", fontFamily: "'DM Sans', sans-serif",
              }}>
                {p}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function InspectorPanel({ selectedNode, data, onNodeSelect }) {
  const connectedRelationships = useMemo(() => {
    if (!selectedNode) return [];
    return data.relationships.filter(
      (r) => r.source === selectedNode.id || r.target === selectedNode.id
    );
  }, [selectedNode, data.relationships]);

  if (selectedNode) {
    return (
      <div style={{ padding: "16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
          <div style={{
            width: 32, height: 32, borderRadius: "50%",
            border: `2px solid ${typeColors[selectedNode.type]}`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "11px", fontFamily: "'JetBrains Mono', monospace",
            fontWeight: 600, color: typeColors[selectedNode.type],
          }}>
            {selectedNode.id.toUpperCase()}
          </div>
          <div>
            <div style={{
              fontSize: "11px", fontWeight: 600, textTransform: "uppercase",
              letterSpacing: "0.06em", color: typeColors[selectedNode.type],
            }}>
              {typeLabels[selectedNode.type]}
            </div>
            <div style={{
              fontSize: "9px", color: "#555",
              fontFamily: "'JetBrains Mono', monospace", marginTop: "2px",
            }}>
              Confidence: {Math.round((selectedNode.confidence ?? 0.5) * 100)}%
            </div>
          </div>
        </div>

        <div style={{
          fontSize: "13px", lineHeight: "1.6", color: "#CCC",
          fontFamily: "'Newsreader', serif", fontStyle: "italic",
          padding: "12px",
          borderLeft: `2px solid ${typeColors[selectedNode.type]}30`,
          marginBottom: "20px",
        }}>
          &ldquo;{selectedNode.claim}&rdquo;
        </div>

        {connectedRelationships.length > 0 && (
          <div>
            <div style={{
              fontSize: "10px", textTransform: "uppercase",
              letterSpacing: "0.1em", color: "#555",
              fontWeight: 600, marginBottom: "8px",
            }}>
              Relationships
            </div>
            {connectedRelationships.map((rel, i) => {
              const isSource = rel.source === selectedNode.id;
              const otherId = isSource ? rel.target : rel.source;
              const otherNode = data.claims.find((c) => c.id === otherId);
              const dirLabel = getDirectionLabel(rel.type, isSource);

              return (
                <div key={i} style={{
                  padding: "8px 10px", background: "#141414",
                  borderRadius: "6px", marginBottom: "6px",
                }}>
                  <div style={{
                    display: "flex", alignItems: "center", gap: "6px",
                    fontSize: "11px", marginBottom: "4px",
                  }}>
                    <span style={{
                      color: relColors[rel.type], fontWeight: 600,
                      fontFamily: "'JetBrains Mono', monospace", fontSize: "10px",
                    }}>
                      {relIcons[rel.type]} {rel.type.toUpperCase()}
                    </span>
                    <span style={{ color: "#444", fontSize: "9px" }}>{dirLabel}</span>
                    <span
                      onClick={() => otherNode && onNodeSelect(otherNode)}
                      style={{
                        color: typeColors[otherNode?.type] || "#999",
                        cursor: "pointer", fontWeight: 600,
                        fontFamily: "'JetBrains Mono', monospace", fontSize: "10px",
                      }}
                    >
                      {otherId.toUpperCase()}
                    </span>
                    <ReasoningBadge type={rel.reasoning_type || "deductive"} />
                  </div>
                  <StrengthBar value={rel.strength ?? 0.5} color={relColors[rel.type] || "#555"} />
                  {otherNode && (
                    <div style={{
                      fontSize: "10px", color: "#666", lineHeight: "1.3",
                      fontFamily: "'DM Sans', sans-serif", marginTop: "4px",
                    }}>
                      {otherNode.claim.length > 70 ? otherNode.claim.slice(0, 70) + "\u2026" : otherNode.claim}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  }

  if (data.claims.length > 0) {
    return (
      <div style={{ padding: "16px" }}>
        <div style={{
          display: "grid", gridTemplateColumns: "1fr 1fr",
          gap: "8px", marginBottom: "16px",
        }}>
          {[
            { label: "Claims", value: data.claims.length, color: "#E8E4DF" },
            { label: "Relations", value: data.relationships.length, color: "#E8E4DF" },
            { label: "Premises", value: data.claims.filter((c) => c.type === "premise").length, color: typeColors.premise },
            { label: "Conclusions", value: data.claims.filter((c) => c.type === "conclusion").length, color: typeColors.conclusion },
          ].map((stat, i) => (
            <div key={i} style={{
              background: "#141414", borderRadius: "6px", padding: "10px",
            }}>
              <div style={{
                fontSize: "18px", fontWeight: 600, color: stat.color,
                fontFamily: "'JetBrains Mono', monospace",
              }}>
                {stat.value}
              </div>
              <div style={{
                fontSize: "9px", textTransform: "uppercase",
                letterSpacing: "0.08em", color: "#555", marginTop: "2px",
              }}>
                {stat.label}
              </div>
            </div>
          ))}
        </div>

        <BiasAssessmentPanel bias={data.bias_assessment} />

        <div style={{
          fontSize: "10px", textTransform: "uppercase",
          letterSpacing: "0.1em", color: "#555",
          fontWeight: 600, marginBottom: "8px", marginTop: "16px",
        }}>
          All Claims
        </div>
        {data.claims.map((claim) => (
          <div
            key={claim.id}
            onClick={() => onNodeSelect(claim)}
            style={{
              padding: "8px 10px", background: "#141414",
              borderRadius: "6px", marginBottom: "6px",
              cursor: "pointer", borderLeft: `2px solid ${typeColors[claim.type]}`,
              transition: "background 0.15s",
            }}
            onMouseEnter={(e) => { e.currentTarget.style.background = "#1A1A1A"; }}
            onMouseLeave={(e) => { e.currentTarget.style.background = "#141414"; }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "4px" }}>
              <span style={{
                fontSize: "10px", fontFamily: "'JetBrains Mono', monospace",
                fontWeight: 600, color: typeColors[claim.type],
              }}>
                {claim.id.toUpperCase()}
              </span>
              <span style={{
                fontSize: "8px", textTransform: "uppercase",
                letterSpacing: "0.06em", color: "#555",
              }}>
                {typeLabels[claim.type]}
              </span>
            </div>
            <div style={{
              fontSize: "11px", color: "#888", lineHeight: "1.4",
              fontFamily: "'DM Sans', sans-serif",
            }}>
              {claim.claim.length > 80 ? claim.claim.slice(0, 80) + "\u2026" : claim.claim}
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div style={{
      height: "100%", display: "flex",
      alignItems: "center", justifyContent: "center",
    }}>
      <div style={{ textAlign: "center", color: "#333" }}>
        <div style={{ fontSize: "13px", marginBottom: "4px" }}>Inspector</div>
        <div style={{ fontSize: "11px", color: "#2A2A2A" }}>
          Extract arguments to view details here
        </div>
      </div>
    </div>
  );
}
