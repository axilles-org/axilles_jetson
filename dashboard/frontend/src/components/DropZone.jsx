import React, { useCallback, useState } from "react";

/**
 * Generic drag-and-drop + click-to-browse file target.
 * onFile receives a single File; parent decides what to do with it (upload,
 * parse, etc). `accept` is a comma-separated extension hint (".csv", ".zip").
 */
export default function DropZone({ accept, label, onFile, disabled }) {
  const [dragOver, setDragOver] = useState(false);
  const inputId = React.useId();

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      if (disabled) return;
      const file = e.dataTransfer.files?.[0];
      if (file) onFile(file);
    },
    [onFile, disabled]
  );

  const handleChange = useCallback(
    (e) => {
      const file = e.target.files?.[0];
      if (file) onFile(file);
      e.target.value = ""; // allow re-selecting the same file
    },
    [onFile]
  );

  return (
    <label
      htmlFor={inputId}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      className="dropzone"
      style={{
        display: "block",
        border: `2px dashed ${dragOver ? "var(--accent)" : "var(--border)"}`,
        borderRadius: 10,
        padding: "28px 16px",
        textAlign: "center",
        cursor: disabled ? "not-allowed" : "pointer",
        background: dragOver ? "rgba(91,157,255,0.06)" : "transparent",
        opacity: disabled ? 0.5 : 1,
        transition: "border-color 0.15s, background 0.15s",
      }}
    >
      <input
        id={inputId}
        type="file"
        accept={accept}
        onChange={handleChange}
        disabled={disabled}
        style={{ display: "none" }}
      />
      <div style={{ color: "var(--text-dim)", fontSize: 13 }}>
        {label || `Drag & drop a ${accept || "file"} here, or click to browse`}
      </div>
    </label>
  );
}
