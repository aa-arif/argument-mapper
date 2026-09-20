import { useState, useRef } from "react";
import { SAMPLE_PASSAGE } from "../constants";

const supportedExts = [".txt", ".md"];

export default function TextInputPanel({ text, onTextChange, onExtract, onFileUpload, processing }) {
  const [dragOver, setDragOver] = useState(false);
  const [fileError, setFileError] = useState(null);
  const fileInputRef = useRef(null);

  const validateAndUpload = (file) => {
    const ext = "." + file.name.split(".").pop().toLowerCase();
    if (!supportedExts.includes(ext)) {
      setFileError(`Unsupported file type (${ext}). Use .txt or .md for now. PDF support coming soon.`);
      return;
    }
    if (file.size > 500000) {
      setFileError("File too large (max 500 KB). Try a shorter passage.");
      return;
    }
    setFileError(null);
    onFileUpload(file);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div
        style={{ flex: 1, position: "relative", borderBottom: "1px solid #1A1A1A" }}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          const file = e.dataTransfer.files[0];
          if (file) validateAndUpload(file);
        }}
      >
        {dragOver && (
          <div style={{
            position: "absolute", inset: 0,
            background: "#D4A57415", border: "2px dashed #D4A57460",
            borderRadius: "4px", display: "flex",
            alignItems: "center", justifyContent: "center",
            zIndex: 10, margin: "8px", pointerEvents: "none",
          }}>
            <span style={{ color: "#D4A574", fontSize: "13px", fontWeight: 500 }}>
              Drop file here
            </span>
          </div>
        )}
        <textarea
          value={text}
          onChange={(e) => { onTextChange(e.target.value); setFileError(null); }}
          placeholder={"Paste philosophical text here, or drag & drop a .txt file\u2026\n\nTry a passage from Kant's Groundwork, Descartes' Meditations, or any argumentative philosophical text."}
          style={{
            width: "100%", height: "100%",
            background: "transparent", border: "none", outline: "none", resize: "none",
            padding: "16px", color: "#CCC",
            fontFamily: "'Newsreader', 'Georgia', serif",
            fontSize: "14px", lineHeight: "1.75", letterSpacing: "0.01em",
            boxSizing: "border-box",
          }}
        />
      </div>

      {fileError && (
        <div style={{
          padding: "8px 16px", fontSize: "11px", color: "#C47B7B",
          background: "#C47B7B10", borderBottom: "1px solid #1A1A1A",
        }}>
          {fileError}
        </div>
      )}

      <div style={{
        padding: "12px 16px", display: "flex",
        alignItems: "center", gap: "8px", flexShrink: 0,
      }}>
        <button
          onClick={onExtract}
          disabled={processing || !text.trim()}
          style={{
            flex: 1,
            background: processing ? "#1A1A1A" : (!text.trim() ? "#1A1A1A" : "#D4A574"),
            color: processing || !text.trim() ? "#555" : "#0D0D0D",
            border: "none", padding: "10px 16px", borderRadius: "6px",
            fontSize: "12px", fontWeight: 600,
            textTransform: "uppercase", letterSpacing: "0.06em",
            cursor: processing || !text.trim() ? "default" : "pointer",
            fontFamily: "'DM Sans', sans-serif", transition: "all 0.2s",
          }}
        >
          {processing ? "Extracting arguments\u2026" : "Extract Arguments"}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".txt,.md"
          onChange={(e) => { const f = e.target.files[0]; if (f) validateAndUpload(f); }}
          style={{ display: "none" }}
        />
        <button
          onClick={() => { setFileError(null); onTextChange(SAMPLE_PASSAGE); }}
          disabled={processing}
          style={{
            background: "#141414", color: "#777",
            border: "1px solid #2A2A2A", padding: "10px 12px",
            borderRadius: "6px", fontSize: "11px",
            cursor: processing ? "default" : "pointer",
            fontFamily: "'DM Sans', sans-serif",
            transition: "all 0.2s", whiteSpace: "nowrap",
            opacity: processing ? 0.4 : 1,
          }}
        >
          Sample
        </button>
        <button
          onClick={() => fileInputRef.current?.click()}
          style={{
            background: "#141414", color: "#777",
            border: "1px solid #2A2A2A", padding: "10px 12px",
            borderRadius: "6px", fontSize: "11px",
            cursor: "pointer", fontFamily: "'DM Sans', sans-serif",
            transition: "all 0.2s", whiteSpace: "nowrap",
          }}
        >
          Upload
        </button>
      </div>
    </div>
  );
}
