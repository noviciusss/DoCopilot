"use client";

import { FormEvent } from "react";
import { motion } from "motion/react";
import { ArrowRight, CheckCircle2, Clock, AlertTriangle } from "lucide-react";
import { DocumentLibraryItem } from "../../../lib/hooks/useDocuments";
import { IngestionStatus } from "../../../lib/hooks/useUpload";
import UploadDropzone from "./upload-dropzone";
import UploadStatus from "./upload-status";
import DocumentPipelineMap from "./document-pipeline-map";

type UploadType = "pdf" | "txt" | "text";

interface EmptyWorkspaceProps {
  uploadType: UploadType;
  setUploadType: (t: UploadType) => void;
  file: File | null;
  setFile: (f: File | null) => void;
  plainText: string;
  setPlainText: (t: string) => void;
  ingestionStatus: IngestionStatus;
  uploadError: string | null;
  onSubmit: (e: FormEvent<HTMLFormElement>) => void;
  recentDocuments?: DocumentLibraryItem[];
  onSelectDocument?: (doc: DocumentLibraryItem) => void;
}

function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 / 1024).toFixed(1)} MB`;
}

function getFileExtension(filename: string): string {
  const parts = filename.split(".");
  return parts.length > 1 ? parts.pop()?.toUpperCase() || "DOC" : "DOC";
}

export default function EmptyWorkspace({
  uploadType,
  setUploadType,
  file,
  setFile,
  plainText,
  setPlainText,
  ingestionStatus,
  uploadError,
  onSubmit,
  recentDocuments = [],
  onSelectDocument,
}: EmptyWorkspaceProps) {
  // Only display real documents from the database
  const realRecentDocs = recentDocuments.slice(0, 4);

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
      className="flex flex-col items-center justify-start px-4 py-8 max-w-xl mx-auto w-full gap-5"
    >
      {/* ── Editorial Header (Warm Paper Document Symbol) ──────────────────── */}
      <div className="flex flex-col items-center text-center space-y-2">
        {/* Minimal warm paper sheet icon */}
        <div
          className="relative w-12 h-14 rounded flex flex-col justify-between p-1.5 select-none shadow-sm mb-1"
          style={{
            background: "var(--paper)",
            border: "1px solid var(--paper-border)",
            color: "var(--ink-on-paper)",
          }}
          aria-hidden="true"
        >
          {/* Folded corner */}
          <div
            className="absolute top-0 right-0 w-3 h-3"
            style={{
              background: "var(--paper-2)",
              borderLeft: "1px solid var(--paper-border)",
              borderBottom: "1px solid var(--paper-border)",
              borderRadius: "0 2px 0 2px",
            }}
          />

          <span className="text-[9px] font-mono font-bold tracking-wider pt-0.5">
            INDEX
          </span>

          <div className="space-y-1 pb-0.5">
            <div className="h-0.5 rounded-full w-4/5" style={{ background: "var(--paper-border)" }} />
            <div className="h-0.5 rounded-full w-full" style={{ background: "var(--paper-border)" }} />
            <div className="h-0.5 rounded-full w-2/3" style={{ background: "var(--paper-border)" }} />
          </div>
        </div>

        <h1 className="text-base font-semibold" style={{ color: "var(--text-1)" }}>
          Start with a document.
        </h1>
        <p className="text-xs leading-relaxed max-w-md" style={{ color: "var(--text-2)" }}>
          Upload a PDF or TXT file, or paste raw text. Documents are parsed and indexed into vector
          context for grounded question-answering with verifiable source snippets.
        </p>
      </div>

      {/* ── Main Ingestion / Upload Dropzone Card ─────────────────────────── */}
      <div
        className="w-full rounded-xl p-5 space-y-4"
        style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
      >
        <UploadDropzone
          uploadType={uploadType}
          setUploadType={setUploadType}
          file={file}
          setFile={setFile}
          plainText={plainText}
          setPlainText={setPlainText}
          ingestionStatus={ingestionStatus}
          uploadError={uploadError}
          onSubmit={onSubmit}
        />
        <UploadStatus status={ingestionStatus} error={uploadError} />
      </div>

      {/* ── Pipeline System Map (Idle / Current Status) ───────────────────── */}
      <div className="w-full">
        <DocumentPipelineMap status={ingestionStatus} />
      </div>

      {/* ── Real Recent Documents List (Only if documents exist) ──────────── */}
      {realRecentDocs.length > 0 && (
        <div className="w-full space-y-2.5 pt-2">
          <div className="flex items-center justify-between">
            <h2 className="text-label" style={{ color: "var(--text-2)" }}>
              Recent documents in workspace
            </h2>
            <span className="text-[10px] font-mono" style={{ color: "var(--text-3)" }}>
              {realRecentDocs.length} {realRecentDocs.length === 1 ? "FILE" : "FILES"}
            </span>
          </div>

          <div className="space-y-1.5">
            {realRecentDocs.map((doc) => {
              const isReady = doc.ingestion_status === "succeeded";
              const isFailed = doc.ingestion_status === "failed";
              const ext = getFileExtension(doc.filename);

              return (
                <button
                  key={doc.id}
                  type="button"
                  onClick={() => onSelectDocument?.(doc)}
                  className="w-full flex items-center justify-between p-3 rounded-lg text-left transition-all group"
                  style={{
                    background: "var(--surface)",
                    border: "1px solid var(--border)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = "var(--cobalt)";
                    e.currentTarget.style.background = "var(--surface-2)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = "var(--border)";
                    e.currentTarget.style.background = "var(--surface)";
                  }}
                >
                  <div className="flex items-center gap-3 min-w-0 flex-1">
                    {/* Tiny warm paper badge */}
                    <div
                      className="w-7 h-8 rounded flex items-center justify-center flex-shrink-0 font-mono text-[9px] font-bold shadow-xs"
                      style={{
                        background: "var(--paper)",
                        border: "1px solid var(--paper-border)",
                        color: "var(--ink-on-paper)",
                      }}
                      aria-hidden="true"
                    >
                      {ext}
                    </div>

                    <div className="min-w-0 flex-1">
                      <p
                        className="text-xs font-medium truncate group-hover:text-[var(--text-1)] transition-colors"
                        style={{ color: "var(--text-1)" }}
                      >
                        {doc.filename}
                      </p>
                      <p className="text-meta mt-0.5">
                        {formatBytes(doc.file_size_bytes)} ·{" "}
                        {new Date(doc.created_at).toLocaleDateString("en-US", {
                          month: "short",
                          day: "numeric",
                        })}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 flex-shrink-0 ml-3">
                    {isReady ? (
                      <span
                        className="flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded"
                        style={{
                          background: "var(--green-dim)",
                          color: "var(--green)",
                          border: "1px solid var(--green)",
                        }}
                      >
                        <CheckCircle2 size={10} aria-hidden="true" />
                        READY
                      </span>
                    ) : isFailed ? (
                      <span
                        className="flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded"
                        style={{
                          background: "var(--red-dim)",
                          color: "var(--red)",
                          border: "1px solid var(--red)",
                        }}
                      >
                        <AlertTriangle size={10} aria-hidden="true" />
                        FAILED
                      </span>
                    ) : (
                      <span
                        className="flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded"
                        style={{
                          background: "var(--cobalt-dim)",
                          color: "var(--cobalt)",
                          border: "1px solid var(--cobalt)",
                        }}
                      >
                        <Clock size={10} aria-hidden="true" />
                        {doc.ingestion_status.toUpperCase()}
                      </span>
                    )}

                    <ArrowRight
                      size={13}
                      className="opacity-40 group-hover:opacity-100 group-hover:translate-x-0.5 transition-all"
                      style={{ color: "var(--text-3)" }}
                      aria-hidden="true"
                    />
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </motion.div>
  );
}
