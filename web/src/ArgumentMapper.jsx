import { useState, useEffect, useRef, useMemo } from "react";
import { typeColors, relColors, relIcons } from "./constants";
import Graph from "./components/Graph";
import HighlightedText from "./components/HighlightedText";
import TextInputPanel from "./components/TextInputPanel";
import ResizeHandle from "./components/ResizeHandle";
import InspectorPanel from "./components/InspectorPanel";
import ComparisonSide from "./components/ComparisonSide";

// ─── Fuzzy Match for Comparison ─────────────────────────────────────────
function fuzzyMatch(a, b) {
  if (!a || !b) return 0;
  const al = a.toLowerCase();
  const bl = b.toLowerCase();
  if (al === bl) return 1;
  let common = 0;
  const aWords = new Set(al.split(/\s+/));
  const bWords = new Set(bl.split(/\s+/));
  for (const w of aWords) {
    if (bWords.has(w)) common++;
  }
  const union = new Set([...aWords, ...bWords]).size;
  return union > 0 ? common / union : 0;
}

function computeDiff(leftData, rightData) {
  if (!leftData.claims.length || !rightData.claims.length) return null;

  const matches = [];
  const leftMatched = new Set();
  const rightMatched = new Set();

  for (const lc of leftData.claims) {
    let bestMatch = null;
    let bestScore = 0;
    for (const rc of rightData.claims) {
      if (rightMatched.has(rc.id)) continue;
      const score = fuzzyMatch(lc.claim, rc.claim);
      if (score > bestScore) {
        bestScore = score;
        bestMatch = rc;
      }
    }
    if (bestMatch && bestScore > 0.8) {
      matches.push({ left: lc.id, right: bestMatch.id, score: bestScore });
      leftMatched.add(lc.id);
      rightMatched.add(bestMatch.id);
    }
  }

  return { matches, leftMatched, rightMatched };
}

// ─── Main App ───────────────────────────────────────────────────────────
const emptyData = { claims: [], relationships: [], bias_assessment: null };

export default function ArgumentMapper() {
  const [selectedNode, setSelectedNode] = useState(null);
  const [inputText, setInputText] = useState("");
  const [data, setData] = useState(emptyData);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState("input");
  const [leftWidth, setLeftWidth] = useState(340);
  const [rightWidth, setRightWidth] = useState(300);
  const [layoutMode, setLayoutMode] = useState("hierarchical");
  const [hasSeenOnboarding, setHasSeenOnboarding] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const leftBaseRef = useRef(340);
  const rightBaseRef = useRef(300);
  const abortRef = useRef(null);

  const [compareMode, setCompareMode] = useState(false);
  const [leftCompare, setLeftCompare] = useState({ text: "", data: emptyData, processing: false });
  const [rightCompare, setRightCompare] = useState({ text: "", data: emptyData, processing: false });
  const [diffResult, setDiffResult] = useState(null);
  const [backendHealthy, setBackendHealthy] = useState(true);
  const [progressStage, setProgressStage] = useState(0);

  const API_URL = "http://localhost:8000";

  // Health check on mount
  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch(`${API_URL}/health`);
        setBackendHealthy(r.ok);
      } catch {
        setBackendHealthy(false);
      }
    };
    check();
  }, []);

  // Rotating progress messages during extraction
  useEffect(() => {
    if (!processing) { setProgressStage(0); return; }
    const id = setInterval(() => setProgressStage((s) => (s + 1) % 4), 2200);
    return () => clearInterval(id);
  }, [processing]);

  // Keyboard: Escape to deselect / dismiss onboarding
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        setSelectedNode(null);
        setShowOnboarding(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const extractFromAPI = async (text, signal) => {
    const response = await fetch(`${API_URL}/extract`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Server error (${response.status})`);
    }

    const result = await response.json();
    if (!result.claims || result.claims.length === 0) {
      throw new Error("No arguments were extracted. Try a longer or more argumentative passage.");
    }
    return result;
  };

  const handleExtract = async () => {
    if (!inputText.trim()) return;
    setProcessing(true);
    setError(null);

    const controller = new AbortController();
    abortRef.current = controller;
    const timeoutId = setTimeout(() => controller.abort(), 90000);

    try {
      const result = await extractFromAPI(inputText, controller.signal);
      clearTimeout(timeoutId);
      setData(result);
      setViewMode("results");
      if (!hasSeenOnboarding) {
        setShowOnboarding(true);
        setHasSeenOnboarding(true);
      }
    } catch (err) {
      if (err.name === "AbortError") {
        setError("Request timed out \u2014 the model may be under load. Try again.");
      } else if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
        setError("Cannot reach the backend server. Make sure the ACCRE SSH tunnel is running (ssh -L 8000:localhost:8000 ...)");
      } else {
        setError(err.message);
      }
    } finally {
      clearTimeout(timeoutId);
      abortRef.current = null;
      setProcessing(false);
    }
  };

  const handleCancel = () => {
    abortRef.current?.abort();
  };

  const handleCompareExtract = async (side) => {
    const state = side === "left" ? leftCompare : rightCompare;
    const setState = side === "left" ? setLeftCompare : setRightCompare;
    if (!state.text.trim()) return;

    setState((prev) => ({ ...prev, processing: true }));
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 90000);

    try {
      const result = await extractFromAPI(state.text, controller.signal);
      clearTimeout(timeoutId);
      setState((prev) => ({ ...prev, data: result, processing: false }));
      setDiffResult(null);
    } catch (err) {
      clearTimeout(timeoutId);
      setState((prev) => ({ ...prev, processing: false }));
      setError(
        err.name === "AbortError"
          ? "Request timed out \u2014 the model may be under load. Try again."
          : err.message
      );
    }
  };

  const handleDiff = () => {
    setDiffResult(computeDiff(leftCompare.data, rightCompare.data));
  };

  const handleFileUpload = (file) => {
    const reader = new FileReader();
    reader.onload = (e) => setInputText(e.target.result);
    reader.onerror = () => setError("Failed to read file.");
    reader.readAsText(file);
  };

  const makeFileUploader = (setState) => (file) => {
    const reader = new FileReader();
    reader.onload = (e) => setState((prev) => ({ ...prev, text: e.target.result }));
    reader.readAsText(file);
  };

  const handleNewText = () => {
    setViewMode("input");
    setData(emptyData);
    setSelectedNode(null);
    setError(null);
  };

  const handleNodeSelect = (node) => {
    setSelectedNode(node);
    if (showOnboarding) setShowOnboarding(false);
  };

  const toggleCompareMode = () => {
    if (compareMode) {
      setCompareMode(false);
      setDiffResult(null);
    } else {
      setCompareMode(true);
      setSelectedNode(null);
    }
  };

  const memoizedData = useMemo(() => data, [
    data.claims, data.relationships, data.bias_assessment,
  ]);

  return (
    <div style={{
      width: "100%", height: "100vh",
      background: "#0D0D0D", color: "#E8E4DF",
      fontFamily: "'DM Sans', sans-serif",
      display: "flex", flexDirection: "column", overflow: "hidden",
    }}>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .react-flow__controls {
          background: #141414 !important;
          border: 1px solid #2A2A2A !important;
          border-radius: 6px !important;
          overflow: hidden;
        }
        .react-flow__controls button {
          background: #1A1A1A !important;
          border-bottom: 1px solid #2A2A2A !important;
          fill: #777 !important;
          color: #777 !important;
          width: 28px !important;
          height: 28px !important;
        }
        .react-flow__controls button:hover {
          background: #2A2A2A !important;
        }
        .react-flow__edge-textbg { rx: 3; }
      `}</style>

      {/* Header */}
      <div style={{
        padding: "14px 24px",
        borderBottom: "1px solid #1E1E1E",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: "12px" }}>
          <h1 style={{
            fontFamily: "'Instrument Serif', serif",
            fontSize: "22px", fontWeight: 400, color: "#E8E4DF",
            margin: 0, letterSpacing: "-0.02em",
          }}>
            Argument Mapper
          </h1>
          <span style={{
            fontSize: "11px", color: "#555",
            fontFamily: "'JetBrains Mono', monospace",
            textTransform: "uppercase", letterSpacing: "0.08em",
          }}>
            Philosophy
          </span>
        </div>
        <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
          {Object.entries(typeColors).map(([type, color]) => (
            <div key={type} style={{ display: "flex", alignItems: "center", gap: "4px" }}>
              <div style={{
                width: 7, height: 7, borderRadius: "50%",
                backgroundColor: color, opacity: 0.8,
              }} />
              <span style={{ fontSize: "9px", color: "#555", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                {type.replace("_", " ")}
              </span>
            </div>
          ))}
          <div style={{ width: "1px", height: "12px", background: "#2A2A2A" }} />
          {Object.entries(relColors).slice(0, 2).map(([type, color]) => (
            <div key={type} style={{ display: "flex", alignItems: "center", gap: "4px" }}>
              <span style={{ fontSize: "10px", color }}>{relIcons[type]}</span>
              <span style={{ fontSize: "9px", color: "#555", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                {type}
              </span>
            </div>
          ))}
          <div style={{ width: "1px", height: "12px", background: "#2A2A2A" }} />
          <button
            onClick={toggleCompareMode}
            style={{
              background: compareMode ? "#D4A57425" : "#141414",
              color: compareMode ? "#D4A574" : "#555",
              border: `1px solid ${compareMode ? "#D4A57440" : "#2A2A2A"}`,
              padding: "4px 10px", borderRadius: "4px",
              fontSize: "10px", fontWeight: 500,
              cursor: "pointer", fontFamily: "'DM Sans', sans-serif",
              transition: "all 0.15s",
            }}
          >
            Compare
          </button>
        </div>
      </div>

      {!backendHealthy && (
        <div style={{
          padding: "8px 24px",
          background: "#C47B7B15",
          borderBottom: "1px solid #C47B7B30",
          color: "#C47B7B",
          fontSize: "11px",
          fontFamily: "'JetBrains Mono', monospace",
          display: "flex", alignItems: "center", gap: "8px",
          flexShrink: 0,
        }}>
          <span>●</span>
          <span>Backend unreachable on port 8000 — run <code style={{ color: "#E8E4DF" }}>python server.py</code> in another terminal.</span>
        </div>
      )}

      {/* Main Content */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {compareMode ? (
          <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
            <ComparisonSide
              side="left"
              text={leftCompare.text}
              onTextChange={(t) => setLeftCompare((prev) => ({ ...prev, text: t }))}
              onExtract={() => handleCompareExtract("left")}
              onFileUpload={makeFileUploader(setLeftCompare)}
              processing={leftCompare.processing}
              data={leftCompare.data}
              selectedNode={selectedNode}
              onNodeSelect={handleNodeSelect}
              layoutMode={layoutMode}
              diff={diffResult}
            />

            <div style={{
              width: "40px", display: "flex", flexDirection: "column",
              alignItems: "center", justifyContent: "center",
              borderLeft: "1px solid #1E1E1E", borderRight: "1px solid #1E1E1E",
              flexShrink: 0, gap: "8px",
            }}>
              <button
                onClick={handleDiff}
                disabled={!leftCompare.data.claims.length || !rightCompare.data.claims.length}
                style={{
                  background: diffResult ? "#D4A57425" : "#1A1A1A",
                  color: diffResult ? "#D4A574" : "#555",
                  border: `1px solid ${diffResult ? "#D4A57440" : "#2A2A2A"}`,
                  padding: "6px 4px", borderRadius: "4px",
                  fontSize: "8px", fontWeight: 600,
                  cursor: leftCompare.data.claims.length && rightCompare.data.claims.length ? "pointer" : "default",
                  fontFamily: "'DM Sans', sans-serif",
                  writingMode: "vertical-lr", textOrientation: "mixed",
                  letterSpacing: "0.06em", textTransform: "uppercase",
                  transition: "all 0.15s",
                  opacity: leftCompare.data.claims.length && rightCompare.data.claims.length ? 1 : 0.4,
                }}
              >
                Diff
              </button>
              {diffResult && (
                <div style={{
                  fontSize: "8px", color: "#555",
                  fontFamily: "'JetBrains Mono', monospace",
                  writingMode: "vertical-lr",
                }}>
                  {diffResult.matches.length} shared
                </div>
              )}
            </div>

            <ComparisonSide
              side="right"
              text={rightCompare.text}
              onTextChange={(t) => setRightCompare((prev) => ({ ...prev, text: t }))}
              onExtract={() => handleCompareExtract("right")}
              onFileUpload={makeFileUploader(setRightCompare)}
              processing={rightCompare.processing}
              data={rightCompare.data}
              selectedNode={selectedNode}
              onNodeSelect={handleNodeSelect}
              layoutMode={layoutMode}
              diff={diffResult}
            />
          </div>
        ) : (
          <>
            {/* Left panel */}
            <div style={{
              width: `${leftWidth}px`, minWidth: "240px", maxWidth: "500px",
              borderRight: "1px solid #1E1E1E",
              display: "flex", flexDirection: "column", flexShrink: 0,
            }}>
              <div style={{
                padding: "10px 16px", borderBottom: "1px solid #1A1A1A",
                display: "flex", alignItems: "center", justifyContent: "space-between",
              }}>
                <span style={{
                  fontSize: "10px", textTransform: "uppercase",
                  letterSpacing: "0.1em", color: "#555", fontWeight: 600,
                }}>
                  {viewMode === "input" ? "Input Text" : "Source Text"}
                </span>
                {viewMode === "results" && (
                  <button
                    onClick={handleNewText}
                    style={{
                      background: "transparent", color: "#555",
                      border: "1px solid #2A2A2A", padding: "3px 8px",
                      borderRadius: "4px", fontSize: "9px", fontWeight: 600,
                      textTransform: "uppercase", letterSpacing: "0.06em",
                      cursor: "pointer", fontFamily: "'DM Sans', sans-serif",
                    }}
                  >
                    New Text
                  </button>
                )}
              </div>
              <div style={{ flex: 1, overflow: "auto" }}>
                {viewMode === "input" ? (
                  <TextInputPanel
                    text={inputText}
                    onTextChange={setInputText}
                    onExtract={handleExtract}
                    onFileUpload={handleFileUpload}
                    processing={processing}
                  />
                ) : (
                  <div style={{ padding: "16px" }}>
                    <HighlightedText
                      text={inputText}
                      selectedNode={selectedNode}
                      claims={memoizedData.claims}
                      onClaimClick={handleNodeSelect}
                    />
                  </div>
                )}
              </div>
            </div>

            <ResizeHandle
              onDragStart={() => { leftBaseRef.current = leftWidth; }}
              onDrag={(delta) => setLeftWidth(Math.min(500, Math.max(240, leftBaseRef.current + delta)))}
            />

            {/* Center -- Graph */}
            <div style={{ flex: 1, position: "relative", overflow: "hidden" }}>
              {memoizedData.claims.length > 0 ? (
                <Graph
                  data={memoizedData}
                  selectedNode={selectedNode}
                  onNodeSelect={handleNodeSelect}
                  layoutMode={layoutMode}
                />
              ) : (
                <div style={{
                  width: "100%", height: "100%",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  flexDirection: "column", gap: "12px",
                }}>
                  {processing ? (
                    <>
                      <div style={{
                        width: "40px", height: "40px", borderRadius: "50%",
                        border: "3px solid #1E1E1E", borderTopColor: "#D4A574",
                        animation: "spin 1s linear infinite",
                      }} />
                      <div style={{ color: "#555", fontSize: "13px" }}>
                        {[
                          "Reading the passage\u2026",
                          "Extracting claims\u2026",
                          "Mapping relationships\u2026",
                          "Assessing bias & assumptions\u2026",
                        ][progressStage]}
                      </div>
                      <div style={{ color: "#333", fontSize: "11px", fontFamily: "'JetBrains Mono', monospace" }}>
                        claude-sonnet-4-6
                      </div>
                      <button
                        onClick={handleCancel}
                        style={{
                          marginTop: "8px", background: "transparent",
                          color: "#555", border: "1px solid #2A2A2A",
                          padding: "6px 16px", borderRadius: "6px",
                          fontSize: "11px", cursor: "pointer",
                          fontFamily: "'DM Sans', sans-serif",
                          transition: "all 0.15s",
                        }}
                      >
                        Cancel
                      </button>
                    </>
                  ) : error ? (
                    <>
                      <div style={{
                        width: "60px", height: "60px", borderRadius: "50%",
                        border: "2px solid #C47B7B30",
                        display: "flex", alignItems: "center", justifyContent: "center",
                      }}>
                        <span style={{ fontSize: "24px", color: "#C47B7B" }}>!</span>
                      </div>
                      <div style={{ color: "#C47B7B", fontSize: "14px", fontWeight: 500 }}>
                        Extraction failed
                      </div>
                      <div style={{
                        color: "#888", fontSize: "12px", textAlign: "center",
                        maxWidth: "280px", lineHeight: "1.5",
                      }}>
                        {error}
                      </div>
                      <button
                        onClick={handleExtract}
                        style={{
                          marginTop: "8px", background: "#C47B7B20",
                          color: "#C47B7B", border: "1px solid #C47B7B30",
                          padding: "6px 16px", borderRadius: "6px",
                          fontSize: "11px", fontWeight: 600, cursor: "pointer",
                          fontFamily: "'DM Sans', sans-serif",
                        }}
                      >
                        Retry
                      </button>
                    </>
                  ) : (
                    <>
                      <div style={{
                        width: "60px", height: "60px", borderRadius: "50%",
                        border: "2px dashed #2A2A2A",
                        display: "flex", alignItems: "center", justifyContent: "center",
                      }}>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#333" strokeWidth="1.5">
                          <circle cx="8" cy="8" r="3" />
                          <circle cx="16" cy="16" r="3" />
                          <circle cx="16" cy="6" r="3" />
                          <line x1="10.5" y1="9" x2="13.5" y2="15" />
                          <line x1="10.5" y1="7" x2="13" y2="6.5" />
                        </svg>
                      </div>
                      <div style={{ color: "#444", fontSize: "14px", fontWeight: 500 }}>
                        No argument graph yet
                      </div>
                      <div style={{
                        color: "#333", fontSize: "12px", textAlign: "center",
                        maxWidth: "280px", lineHeight: "1.5",
                      }}>
                        Paste a philosophical text in the left panel and click{" "}
                        <span style={{ color: "#D4A574" }}>Extract Arguments</span>{" "}
                        to generate an interactive argument map.
                      </div>
                    </>
                  )}
                </div>
              )}

              {/* Graph controls overlay */}
              {memoizedData.claims.length > 0 && (
                <>
                  <div style={{
                    position: "absolute", top: 12, right: 12,
                    display: "flex", gap: "4px", zIndex: 10,
                  }}>
                    {[
                      { key: "hierarchical", label: "Hierarchy" },
                      { key: "free", label: "Free" },
                    ].map((opt) => (
                      <button
                        key={opt.key}
                        onClick={() => setLayoutMode(opt.key)}
                        style={{
                          background: layoutMode === opt.key ? "#D4A57425" : "#141414",
                          color: layoutMode === opt.key ? "#D4A574" : "#555",
                          border: `1px solid ${layoutMode === opt.key ? "#D4A57440" : "#2A2A2A"}`,
                          padding: "4px 10px", borderRadius: "4px",
                          fontSize: "10px", fontWeight: 500,
                          cursor: "pointer", fontFamily: "'DM Sans', sans-serif",
                          transition: "all 0.15s",
                        }}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                  <div style={{
                    position: "absolute", bottom: 16, left: "50%",
                    transform: "translateX(-50%)",
                    fontSize: "10px", color: "#333",
                    fontFamily: "'JetBrains Mono', monospace",
                    zIndex: 10, whiteSpace: "nowrap",
                  }}>
                    scroll to zoom · drag to pan · click nodes to inspect · esc to deselect
                  </div>
                </>
              )}

              {/* Onboarding overlay */}
              {showOnboarding && memoizedData.claims.length > 0 && (
                <div
                  onClick={() => setShowOnboarding(false)}
                  style={{
                    position: "absolute", inset: 0,
                    background: "#0D0D0DEE",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    zIndex: 20, cursor: "pointer",
                    animation: "fadeIn 0.3s ease",
                  }}
                >
                  <div style={{
                    maxWidth: "360px", padding: "28px",
                    background: "#141414", borderRadius: "12px",
                    border: "1px solid #2A2A2A",
                  }}>
                    <div style={{
                      fontFamily: "'Instrument Serif', serif",
                      fontSize: "18px", color: "#E8E4DF",
                      marginBottom: "16px",
                    }}>
                      Argument map ready
                    </div>
                    <div style={{
                      display: "flex", flexDirection: "column", gap: "12px",
                      fontSize: "12px", color: "#888",
                      fontFamily: "'DM Sans', sans-serif", lineHeight: "1.5",
                    }}>
                      {[
                        { icon: "\u25CF", color: typeColors.premise, text: "Premises sit at the bottom, conclusions flow upward" },
                        { icon: "\u2191", color: relColors.supports, text: "Arrows show how claims support or oppose each other" },
                        { icon: "\u25CE", color: "#D4A574", text: "Click any node to see the full claim and source passage" },
                        { icon: "\u21D4", color: "#666", text: "Drag panel edges to resize, scroll to zoom the graph" },
                      ].map((item, i) => (
                        <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: "10px" }}>
                          <span style={{
                            color: item.color, fontSize: "14px", lineHeight: "1.2",
                            flexShrink: 0, width: "16px", textAlign: "center",
                          }}>
                            {item.icon}
                          </span>
                          <span>{item.text}</span>
                        </div>
                      ))}
                    </div>
                    <div style={{
                      marginTop: "20px", fontSize: "10px", color: "#444",
                      textAlign: "center", fontFamily: "'JetBrains Mono', monospace",
                    }}>
                      click anywhere to dismiss
                    </div>
                  </div>
                </div>
              )}
            </div>

            <ResizeHandle
              onDragStart={() => { rightBaseRef.current = rightWidth; }}
              onDrag={(delta) => setRightWidth(Math.min(450, Math.max(220, rightBaseRef.current - delta)))}
            />

            {/* Right panel -- Inspector */}
            <div style={{
              width: `${rightWidth}px`, minWidth: "220px", maxWidth: "450px",
              borderLeft: "1px solid #1E1E1E",
              display: "flex", flexDirection: "column", flexShrink: 0,
            }}>
              <div style={{
                padding: "10px 16px", borderBottom: "1px solid #1A1A1A",
              }}>
                <span style={{
                  fontSize: "10px", textTransform: "uppercase",
                  letterSpacing: "0.1em", color: "#555", fontWeight: 600,
                }}>
                  {selectedNode ? "Node Detail" : memoizedData.claims.length > 0 ? "Overview" : "Inspector"}
                </span>
              </div>
              <div style={{ flex: 1, overflow: "auto" }}>
                <InspectorPanel
                  selectedNode={selectedNode}
                  data={memoizedData}
                  onNodeSelect={handleNodeSelect}
                />
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
