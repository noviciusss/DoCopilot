"use client";

import { FormEvent, DragEvent, useRef, useState } from "react";
import { motion } from "motion/react";
import { UploadCloud, FileText, X, FileType2 } from "lucide-react";
import { IngestionStatus } from "../../../lib/hooks/useUpload";

type UploadType = "pdf" | "txt" | "text";

interface Props {
  uploadType: UploadType;
  setUploadType: (t: UploadType) => void;
  file: File | null;
  setFile: (f: File | null) => void;
  plainText: string;
  setPlainText: (t: string) => void;
  ingestionStatus: IngestionStatus;
  uploadError: string | null;
  onSubmit: (e: FormEvent<HTMLFormElement>) => void;
}

const TYPE_TABS: { id: UploadType; label: string }[] = [
  { id: "pdf",  label: "PDF" },
  { id: "txt",  label: "TXT" },
  { id: "text", label: "Paste" },
];

const SUBMIT_LABEL: Record<IngestionStatus, string> = {
  idle:      "Index document",
  uploading: "Uploading…",
  queued:    "Queued…",
  running:   "Processing…",
  succeeded: "Re-index",
  failed:    "Retry",
};

export default function UploadDropzone({
  uploadType,
  setUploadType,
  file,
  setFile,
  plainText,
  setPlainText,
  ingestionStatus,
  uploadError,
  onSubmit,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const busy = ingestionStatus === "uploading" || ingestionStatus === "running" || ingestionStatus === "queued";

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    setDragging(true);
  };
  const handleDragLeave = () => setDragging(false);
  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (!dropped) return;
    const isPdf = dropped.type === "application/pdf" || dropped.name.endsWith(".pdf");
    const isTxt = dropped.type === "text/plain" || dropped.name.endsWith(".txt");
    if (isPdf) { setUploadType("pdf"); setFile(dropped); }
    else if (isTxt) { setUploadType("txt"); setFile(dropped); }
  };

  const accept = uploadType === "pdf"
    ? ".pdf,application/pdf"
    : ".txt,text/plain";

  return (
    <form onSubmit={onSubmit} className="space-y-3">
      {/* Type selector */}
      <div
        className="flex rounded-md overflow-hidden"
        style={{ border: "1px solid var(--border)", background: "var(--ink)" }}
        role="group"
        aria-label="Upload format"
      >
        {TYPE_TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => { setUploadType(tab.id); setFile(null); }}
            className="flex-1 py-1.5 text-xs font-medium transition-colors"
            style={{
              background:   uploadType === tab.id ? "var(--cobalt)"    : "transparent",
              color:        uploadType === tab.id ? "#fff"              : "var(--text-2)",
              borderRight:  tab.id !== "text"     ? "1px solid var(--border)" : "none",
            }}
            aria-pressed={uploadType === tab.id}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* File drop zone */}
      {uploadType !== "text" ? (
        <div
          className="relative rounded-lg transition-colors"
          style={{
            border: `1.5px dashed ${dragging ? "var(--cobalt)" : file ? "var(--cobalt)" : "var(--border)"}`,
            background: dragging ? "var(--cobalt-dim)" : file ? "var(--cobalt-dim)" : "var(--surface-2)",
          }}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            id="file-upload-input"
            ref={inputRef}
            type="file"
            accept={accept}
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
            aria-label={`Choose ${uploadType === "pdf" ? "PDF" : "TXT"} file, or drag and drop`}
          />

          <div className="flex flex-col items-center justify-center py-8 px-4 text-center">
            {file ? (
              <div className="flex items-center gap-2.5 w-full px-2">
                <div className="p-2 rounded-md flex-shrink-0" style={{ background: "var(--cobalt-dim)", color: "var(--cobalt)" }}>
                  <FileText size={16} aria-hidden="true" />
                </div>
                <div className="flex-1 min-w-0 text-left">
                  <p className="text-xs font-medium truncate" style={{ color: "var(--text-1)" }}>{file.name}</p>
                  <p className="text-meta">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); setFile(null); if (inputRef.current) inputRef.current.value = ""; }}
                  className="relative z-20 p-1 rounded transition-colors flex-shrink-0"
                  style={{ color: "var(--text-3)" }}
                  aria-label="Remove file"
                >
                  <X size={14} aria-hidden="true" />
                </button>
              </div>
            ) : (
              <>
                {/* Paper stack — fans out on drag-over, collapses otherwise */}
                <div className="relative w-10 h-12 mb-3" aria-hidden="true">
                  {/* Back layers */}
                  <motion.div
                    className="absolute inset-0 rounded"
                    style={{ background: "var(--border)", originX: 0.5, originY: 1 }}
                    animate={dragging
                      ? { rotate: -10, x: -6, y: -4, opacity: 0.6 }
                      : { rotate: 0,   x: 0,  y: 0,  opacity: 0   }}
                    transition={{ duration: 0.18, ease: [0.16, 1, 0.3, 1] }}
                  />
                  <motion.div
                    className="absolute inset-0 rounded"
                    style={{ background: "var(--border-light)", originX: 0.5, originY: 1 }}
                    animate={dragging
                      ? { rotate: -5,  x: -3, y: -2, opacity: 0.8 }
                      : { rotate: 0,   x: 0,  y: 0,  opacity: 0   }}
                    transition={{ duration: 0.18, ease: [0.16, 1, 0.3, 1], delay: 0.03 }}
                  />
                  {/* Front page — always visible */}
                  <div
                    className="absolute inset-0 rounded flex items-center justify-center"
                    style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
                  >
                    <UploadCloud
                      size={18}
                      style={{ color: dragging ? "var(--cobalt)" : "var(--text-3)" }}
                      aria-hidden="true"
                    />
                  </div>
                </div>
                <p className="text-xs font-medium" style={{ color: "var(--text-1)" }}>
                  Drop {uploadType === "pdf" ? "PDF" : "TXT"} here or click to browse
                </p>
                <p className="text-meta mt-0.5">Max 20 MB</p>
              </>
            )}
          </div>
        </div>
      ) : (
        <div className="space-y-1">
          <label htmlFor="paste-text-input" className="text-label flex items-center gap-1.5">
            <FileType2 size={11} aria-hidden="true" />
            Paste text content
          </label>
          <textarea
            id="paste-text-input"
            value={plainText}
            onChange={(e) => setPlainText(e.target.value)}
            rows={8}
            placeholder="Paste your document text here…"
            className="w-full rounded-md px-3 py-2 text-xs resize-none transition-colors"
            style={{
              background: "var(--surface-2)",
              border: "1px solid var(--border)",
              color: "var(--text-1)",
              outline: "none",
              fontFamily: "var(--font-sans)",
            }}
            onFocus={(e) => (e.currentTarget.style.borderColor = "var(--cobalt)")}
            onBlur={(e) => (e.currentTarget.style.borderColor = "var(--border)")}
          />
        </div>
      )}

      {/* Error */}
      {uploadError && (
        <div
          className="text-xs rounded-md px-3 py-2 leading-snug"
          style={{ background: "var(--red-dim)", border: "1px solid var(--red)", color: "var(--red)" }}
          role="alert"
        >
          {uploadError}
        </div>
      )}

      {/* Submit */}
      <button
        id="submit-ingestion-button"
        type="submit"
        disabled={
          busy ||
          (uploadType !== "text" && !file) ||
          (uploadType === "text" && !plainText.trim())
        }
        className="w-full py-2 text-xs font-medium rounded-md transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        style={{
          background: "var(--cobalt)",
          color: "#fff",
        }}
        onMouseEnter={(e) => !e.currentTarget.disabled && (e.currentTarget.style.background = "var(--cobalt-hover)")}
        onMouseLeave={(e) => (e.currentTarget.style.background = "var(--cobalt)")}
      >
        {SUBMIT_LABEL[ingestionStatus]}
      </button>
    </form>
  );
}
