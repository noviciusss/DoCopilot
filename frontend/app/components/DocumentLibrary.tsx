"use client";

import { useState } from "react";
import { DocumentLibraryItem } from "../../lib/hooks/useDocuments";

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short", day: "numeric", year: "numeric"
  });
}

const STATUS_COLORS: Record<string, string> = {
  succeeded: "text-emerald-400 bg-emerald-400/10 border-emerald-400/20",
  running:   "text-amber-400 bg-amber-400/10 border-amber-400/20",
  queued:    "text-sky-400 bg-sky-400/10 border-sky-400/20",
  failed:    "text-rose-400 bg-rose-400/10 border-rose-400/20",
  unknown:   "text-zinc-400 bg-zinc-400/10 border-zinc-400/20",
};

const MIME_ICON: Record<string, string> = {
  "application/pdf": "📄",
  "text/plain": "📝",
};

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  documents: DocumentLibraryItem[];
  loading: boolean;
  error: string | null;
  showAllDocs: boolean;
  activeDocumentId: string | null;
  onSelect: (doc: DocumentLibraryItem) => void;
  onDelete: (docId: string) => void;
  onToggleScope: () => void;
  onRefresh: () => void;
}

export default function DocumentLibrary({
  documents,
  loading,
  error,
  showAllDocs,
  activeDocumentId,
  onSelect,
  onDelete,
  onToggleScope,
  onRefresh,
}: Props) {
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

  const handleDelete = (docId: string) => {
    if (confirmDelete === docId) {
      onDelete(docId);
      setConfirmDelete(null);
    } else {
      setConfirmDelete(docId);
      // Auto-cancel confirm after 3s
      setTimeout(() => setConfirmDelete(null), 3000);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
        <h2 className="text-sm font-semibold text-zinc-200 tracking-wide">
          📚 Document Library
        </h2>
        <div className="flex items-center gap-2">
          <button
            id="doc-library-refresh"
            onClick={onRefresh}
            title="Refresh"
            className="text-zinc-500 hover:text-zinc-300 transition-colors text-xs"
          >
            ↺
          </button>
          <button
            id="doc-library-scope-toggle"
            onClick={onToggleScope}
            className={`text-xs px-2 py-0.5 rounded border transition-colors ${
              showAllDocs
                ? "border-violet-500/40 text-violet-400 bg-violet-500/10"
                : "border-zinc-700 text-zinc-400 hover:text-zinc-200"
            }`}
          >
            {showAllDocs ? "All workspace" : "My docs"}
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1">
        {loading && (
          <div className="flex items-center justify-center py-8">
            <span className="text-xs text-zinc-500 animate-pulse">Loading library…</span>
          </div>
        )}

        {error && !loading && (
          <div className="text-xs text-rose-400 bg-rose-400/10 border border-rose-400/20 rounded px-3 py-2 mx-1">
            {error}
          </div>
        )}

        {!loading && !error && documents.length === 0 && (
          <div className="text-center py-8 text-zinc-600 text-xs">
            <p>No documents yet.</p>
            <p className="mt-1">Upload a PDF or text file to get started.</p>
          </div>
        )}

        {!loading && documents.map((doc) => {
          const isActive = doc.id === activeDocumentId;
          const canChat  = doc.ingestion_status === "succeeded" && doc.qdrant_collection;
          const icon     = MIME_ICON[doc.mime_type] ?? "📁";
          const statusCls = STATUS_COLORS[doc.ingestion_status] ?? STATUS_COLORS.unknown;

          return (
            <div
              key={doc.id}
              id={`doc-card-${doc.id.slice(0, 8)}`}
              className={`group relative rounded-lg border p-3 transition-all cursor-pointer ${
                isActive
                  ? "border-violet-500/50 bg-violet-500/10"
                  : "border-zinc-800 bg-zinc-900/50 hover:border-zinc-700 hover:bg-zinc-800/50"
              }`}
              onClick={() => canChat && onSelect(doc)}
              title={canChat ? "Click to chat with this document" : "Document not ready yet"}
            >
              {/* File info */}
              <div className="flex items-start gap-2">
                <span className="text-lg leading-none mt-0.5">{icon}</span>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-zinc-200 truncate">{doc.filename}</p>
                  <p className="text-[10px] text-zinc-500 mt-0.5">
                    {formatBytes(doc.file_size_bytes)} · {formatDate(doc.created_at)}
                  </p>
                </div>

                {/* Delete button */}
                <button
                  id={`doc-delete-${doc.id.slice(0, 8)}`}
                  onClick={(e) => { e.stopPropagation(); handleDelete(doc.id); }}
                  className={`opacity-0 group-hover:opacity-100 text-[10px] px-1.5 py-0.5 rounded border transition-all ${
                    confirmDelete === doc.id
                      ? "opacity-100 border-rose-500/50 text-rose-400 bg-rose-500/10"
                      : "border-zinc-700 text-zinc-500 hover:text-rose-400 hover:border-rose-500/30"
                  }`}
                >
                  {confirmDelete === doc.id ? "Confirm?" : "✕"}
                </button>
              </div>

              {/* Status badge */}
              <div className="mt-2 flex items-center gap-2">
                <span className={`text-[10px] px-1.5 py-0.5 rounded border font-medium ${statusCls}`}>
                  {doc.ingestion_status}
                </span>
                {isActive && (
                  <span className="text-[10px] text-violet-400 font-medium">● active</span>
                )}
                {!canChat && doc.ingestion_status !== "succeeded" && (
                  <span className="text-[10px] text-zinc-600 italic">not ready</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
