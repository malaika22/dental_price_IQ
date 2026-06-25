import { useCallback, useRef, useState } from "react";

interface UploadZoneProps {
  file: File | null;
  onFileSelect: (file: File | null) => void;
  onInvalidFile?: () => void;
  disabled?: boolean;
}

export function UploadZone({ file, onFileSelect, onInvalidFile, disabled }: UploadZoneProps) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (f: File | undefined) => {
      if (!f) return;
      if (f.type !== "application/pdf" && !f.name.toLowerCase().endsWith(".pdf")) {
        onInvalidFile?.();
        return;
      }
      onFileSelect(f);
    },
    [onFileSelect, onInvalidFile],
  );

  return (
    <div className="upload-zone">
      <div
        className={`dropzone ${dragOver ? "dropzone--drag" : ""} ${file ? "dropzone--ready" : ""}`}
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label="Upload PDF"
        aria-disabled={disabled}
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) => {
          if (!disabled && (e.key === "Enter" || e.key === " ")) {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          if (!disabled) handleFile(e.dataTransfer.files[0]);
        }}
      >
        <div className="dropzone__ring" aria-hidden>
          <div className="dropzone__icon-wrap">
            <svg viewBox="0 0 32 32" fill="none" className="dropzone__icon">
              <path
                d="M16 4v16M10 14l6-6 6 6"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M6 22v2a2 2 0 002 2h16a2 2 0 002-2v-2"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
          </div>
        </div>

        {file ? (
          <>
            <p className="dropzone__badge">Ready</p>
            <p className="dropzone__filename">{file.name}</p>
            <p className="dropzone__hint">{(file.size / 1024).toFixed(0)} KB · Tap to replace</p>
          </>
        ) : (
          <>
            <p className="dropzone__title">Drop your order PDF here</p>
            <p className="dropzone__hint">or click to browse · Henry Schein exports supported</p>
          </>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf,.pdf"
        className="sr-only"
        disabled={disabled}
        onChange={(e) => handleFile(e.target.files?.[0])}
      />
    </div>
  );
}
