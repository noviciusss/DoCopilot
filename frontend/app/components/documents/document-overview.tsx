"use client";

import { motion } from "motion/react";
import {
  Info,
  AlertTriangle,
  Loader2,
  CheckCircle2,
  ArrowUpRight,
} from "lucide-react";
import { DocumentLibraryItem } from "../../../lib/hooks/useDocuments";
import { IngestionStatus } from "../../../lib/hooks/useUpload";
import DocumentPipelineMap from "./document-pipeline-map";

interface DocOverviewProps {
  doc: DocumentLibraryItem;
  effectiveStatus: IngestionStatus;
  isAsking?: boolean;
  hasAnswer?: boolean;
  hasSources?: boolean;
  onSendQuestion: (q: string) => void;
}

const SHORTCUT_PROMPTS = [
  {
    label: "Summarize document",
    description: "Executive summary and core findings",
    prompt: "Summarize the key points and core findings of this document.",
  },
  {
    label: "Key concepts",
    description: "Main definitions, terms, and topics",
    prompt: "What are the main concepts, definitions, and topics defined in this document?",
  },
  {
    label: "Create study notes",
    description: "Structured takeaways and bullet review",
    prompt: "Create structured study notes highlighting the main takeaways and key learnings from this document.",
  },
];

function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 / 1024).toFixed(1)} MB`;
}

function getFileType(filename: string, mimeType?: string): string {
  const lower = filename.toLowerCase();
  if (lower.endsWith(".pdf") || mimeType === "application/pdf") return "PDF Document";
  if (lower.endsWith(".txt") || mimeType === "text/plain") return "Plain Text Document";
  if (lower.endsWith(".md")) return "Markdown Document";
  if (lower.endsWith(".doc") || lower.endsWith(".docx")) return "Word Document";
  return mimeType || "Document";
}

function getFileExtension(filename: string): string {
  const parts = filename.split(".");
  return parts.length > 1 ? parts.pop()?.toUpperCase() || "DOC" : "DOC";
}

export default function DocumentOverview({
  doc,
  effectiveStatus,
  isAsking = false,
  hasAnswer = false,
  hasSources = false,
  onSendQuestion,
}: DocOverviewProps) {
  const isReady = effectiveStatus === "succeeded";
  const isProcessing = ["uploading", "queued", "running"].includes(effectiveStatus);
  const isFailed = effectiveStatus === "failed";
  const fileType = getFileType(doc.filename, doc.mime_type);
  const fileExt = getFileExtension(doc.filename);

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
      className="flex flex-col gap-4 max-w-2xl mx-auto w-full px-4 py-6"
    >
      {/* ── Document Identity Header (Warm Paper Surface) ────────────────── */}
      <div
        className="rounded-xl p-5 space-y-4"
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
        }}
      >
        <div className="flex items-start gap-4">
          {/* Warm Paper Document Sheet Icon */}
          <div
            className="relative w-12 h-14 rounded flex-shrink-0 flex flex-col justify-between p-1.5 select-none shadow-sm"
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

            {/* Document extension tag */}
            <span className="text-[9px] font-mono font-bold tracking-wider pt-0.5">
              {fileExt}
            </span>

            {/* Subtle text lines */}
            <div className="space-y-1 pb-0.5">
              <div
                className="h-0.5 rounded-full w-4/5"
                style={{ background: "var(--paper-border)" }}
              />
              <div
                className="h-0.5 rounded-full w-full"
                style={{ background: "var(--paper-border)" }}
              />
              <div
                className="h-0.5 rounded-full w-3/5"
                style={{ background: "var(--paper-border)" }}
              />
            </div>
          </div>

          {/* Metadata */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap mb-1">
              <span
                className="text-[10px] font-mono font-medium px-2 py-0.5 rounded"
                style={{
                  background: "var(--surface-2)",
                  color: "var(--text-2)",
                  border: "1px solid var(--border)",
                }}
              >
                {fileType}
              </span>

              {/* Real Status Badge */}
              {isReady && (
                <span
                  className="flex items-center gap-1 text-[10px] font-mono font-medium px-2 py-0.5 rounded"
                  style={{
                    background: "var(--green-dim)",
                    color: "var(--green)",
                    border: "1px solid var(--green)",
                  }}
                >
                  <CheckCircle2 size={10} aria-hidden="true" />
                  INDEXED & READY
                </span>
              )}
              {isProcessing && (
                <span
                  className="flex items-center gap-1 text-[10px] font-mono font-medium px-2 py-0.5 rounded"
                  style={{
                    background: "var(--cobalt-dim)",
                    color: "var(--cobalt)",
                    border: "1px solid var(--cobalt)",
                  }}
                >
                  <Loader2 size={10} className="animate-spin" aria-hidden="true" />
                  {effectiveStatus.toUpperCase()}
                </span>
              )}
              {isFailed && (
                <span
                  className="flex items-center gap-1 text-[10px] font-mono font-medium px-2 py-0.5 rounded"
                  style={{
                    background: "var(--red-dim)",
                    color: "var(--red)",
                    border: "1px solid var(--red)",
                  }}
                >
                  <AlertTriangle size={10} aria-hidden="true" />
                  FAILED
                </span>
              )}
            </div>

            <h2
              className="text-base font-semibold leading-snug break-words"
              style={{ color: "var(--text-1)", wordBreak: "break-word" }}
            >
              {doc.filename}
            </h2>

            <p className="text-meta mt-1">
              {formatBytes(doc.file_size_bytes)} · Uploaded{" "}
              {new Date(doc.created_at).toLocaleDateString("en-US", {
                month: "short",
                day: "numeric",
                year: "numeric",
              })}
            </p>
          </div>
        </div>

        {/* Status explanation strip */}
        <div
          className="flex items-center gap-2.5 p-3 rounded-lg"
          style={{
            background: "var(--surface-2)",
            border: "1px solid var(--border)",
          }}
          role="status"
          aria-live="polite"
        >
          {isReady && (
            <>
              <CheckCircle2 size={13} style={{ color: "var(--green)", flexShrink: 0 }} aria-hidden="true" />
              <p className="text-xs" style={{ color: "var(--text-2)" }}>
                <span style={{ color: "var(--green)", fontWeight: 500 }}>
                  Ready to query.
                </span>{" "}
                Ask questions about this document in the assistant panel or use the shortcuts below.
              </p>
            </>
          )}
          {isProcessing && (
            <>
              <Loader2 size={13} className="animate-spin flex-shrink-0" style={{ color: "var(--cobalt)" }} aria-hidden="true" />
              <p className="text-xs" style={{ color: "var(--text-2)" }}>
                <span style={{ color: "var(--cobalt)", fontWeight: 500 }}>
                  Ingestion in progress.
                </span>{" "}
                Document is being parsed into vector chunks. Questions will be enabled once indexing completes.
              </p>
            </>
          )}
          {isFailed && (
            <>
              <AlertTriangle size={13} style={{ color: "var(--red)", flexShrink: 0 }} aria-hidden="true" />
              <p className="text-xs" style={{ color: "var(--red)" }}>
                Indexing encountered an error. Please re-upload this document.
              </p>
            </>
          )}
        </div>
      </div>

      {/* ── Document Pipeline Map (Real UI State) ────────────────────────── */}
      <DocumentPipelineMap
        status={effectiveStatus}
        isAsking={isAsking}
        hasAnswer={hasAnswer}
        hasSources={hasSources}
      />

      {/* ── Grounded Context Notice ──────────────────────────────────────── */}
      <div
        className="flex items-start gap-3 p-4 rounded-xl"
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
        }}
      >
        <Info
          size={14}
          style={{ color: "var(--text-3)", flexShrink: 0, marginTop: 2 }}
          aria-hidden="true"
        />
        <div className="space-y-1 text-xs leading-relaxed" style={{ color: "var(--text-2)" }}>
          <p className="font-medium" style={{ color: "var(--text-1)" }}>
            Grounded Retrieval Architecture
          </p>
          <p>
            Answers are generated strictly by retrieving the most relevant passages from this document
            and synthesizing them through the language model. Every factual response displays the{" "}
            <span style={{ color: "var(--amber)", fontWeight: 500 }}>
              retrieved context snippets
            </span>{" "}
            extracted from the document.
          </p>
        </div>
      </div>

      {/* ── Prompt Shortcuts (Sends through existing chat flow) ─────────── */}
      {isReady && (
        <div className="space-y-2.5">
          <div className="flex items-center justify-between">
            <h3 className="text-label" style={{ color: "var(--text-2)" }}>
              Prompt Shortcuts
            </h3>
            <span className="text-[10px] font-mono" style={{ color: "var(--text-3)" }}>
              GROUNDED SSE STREAM
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
            {SHORTCUT_PROMPTS.map((item) => (
              <button
                key={item.label}
                type="button"
                onClick={() => onSendQuestion(item.prompt)}
                disabled={isAsking}
                className="group flex flex-col justify-between text-left p-3.5 rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                style={{
                  background: "var(--surface)",
                  border: "1px solid var(--border)",
                }}
                onMouseEnter={(e) => {
                  if (!isAsking) {
                    e.currentTarget.style.borderColor = "var(--cobalt)";
                    e.currentTarget.style.background = "var(--surface-2)";
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = "var(--border)";
                  e.currentTarget.style.background = "var(--surface)";
                }}
              >
                <div className="flex items-start justify-between w-full mb-2">
                  <span
                    className="text-xs font-semibold leading-tight group-hover:text-[var(--text-1)] transition-colors"
                    style={{ color: "var(--text-1)" }}
                  >
                    {item.label}
                  </span>
                  <ArrowUpRight
                    size={12}
                    className="flex-shrink-0 opacity-40 group-hover:opacity-100 group-hover:text-[var(--cobalt)] transition-all"
                    style={{ color: "var(--text-3)" }}
                    aria-hidden="true"
                  />
                </div>
                <span
                  className="text-[11px] leading-snug line-clamp-2"
                  style={{ color: "var(--text-3)" }}
                >
                  {item.description}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );
}
