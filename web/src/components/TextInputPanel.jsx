import { useRef, useState } from "react";

const ACCEPTED = [".txt", ".md"];

export default function TextInputPanel({
  text,
  onTextChange,
  onExtract,
  onSample,
  onLoadGold,
  processing,
}) {
  const [isDragging, setIsDragging] = useState(false);
  const fileRef = useRef(null);

  const readFile = (file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => onTextChange(String(reader.result ?? ""));
    reader.readAsText(file);
  };

  // A dataset document carries its gold annotations alongside the text, so the
  // overlay can show what the annotators marked. Plain .txt has none, and the
  // overlay toggle stays disabled rather than silently showing nothing.
  const readAnnotated = (file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const doc = JSON.parse(String(reader.result ?? ""));
        onTextChange(doc.text ?? "");
        onLoadGold?.(doc.components ?? []);
      } catch {
        onTextChange(String(reader.result ?? ""));
        onLoadGold?.([]);
      }
    };
    reader.readAsText(file);
  };

  return (
    <div
      className={`input-panel ${isDragging ? "is-dragging" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setIsDragging(false);
        const file = event.dataTransfer.files?.[0];
        if (file?.name.endsWith(".json")) readAnnotated(file);
        else readFile(file);
      }}
    >
      <textarea
        className="input-panel__textarea"
        value={text}
        onChange={(event) => onTextChange(event.target.value)}
        placeholder="Paste an argumentative passage, or drop a .txt file here."
        spellCheck={false}
        rows={10}
      />

      <div className="input-panel__actions">
        <button
          type="button"
          className="button button--primary"
          onClick={onExtract}
          disabled={processing || !text.trim()}
        >
          {processing ? "Extracting…" : "Extract"}
        </button>
        <button type="button" className="button" onClick={onSample} disabled={processing}>
          Sample passage
        </button>
        <button
          type="button"
          className="button"
          onClick={() => fileRef.current?.click()}
          disabled={processing}
        >
          Open file
        </button>
        <input
          ref={fileRef}
          type="file"
          accept={[...ACCEPTED, ".json"].join(",")}
          hidden
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file?.name.endsWith(".json")) readAnnotated(file);
            else readFile(file);
            event.target.value = "";
          }}
        />
        <span className="input-panel__count">{text.length.toLocaleString()} chars</span>
      </div>
    </div>
  );
}
