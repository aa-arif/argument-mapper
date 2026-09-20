import { useState } from "react";

export default function ResizeHandle({ onDragStart, onDrag }) {
  const [dragging, setDragging] = useState(false);

  const handleMouseDown = (e) => {
    e.preventDefault();
    setDragging(true);
    if (onDragStart) onDragStart();

    const handleMouseMove = (ev) => onDrag(ev.clientX - e.clientX);
    const handleMouseUp = () => {
      setDragging(false);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  };

  return (
    <div
      onMouseDown={handleMouseDown}
      style={{
        width: "6px", cursor: "col-resize",
        background: dragging ? "#D4A57440" : "transparent",
        transition: "background 0.15s", flexShrink: 0,
        position: "relative", zIndex: 5,
      }}
      onMouseEnter={(e) => { e.currentTarget.style.background = "#D4A57420"; }}
      onMouseLeave={(e) => { if (!dragging) e.currentTarget.style.background = "transparent"; }}
    >
      <div style={{
        position: "absolute", top: "50%", left: "50%",
        transform: "translate(-50%, -50%)",
        width: "2px", height: "24px", borderRadius: "1px",
        background: dragging ? "#D4A574" : "#333",
        transition: "background 0.15s",
      }} />
    </div>
  );
}
