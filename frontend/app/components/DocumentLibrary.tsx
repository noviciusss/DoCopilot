"use client";

import { useState } from "react";
import { DocumentLibraryItem } from "../../lib/hooks/useDocuments";
import { 
  Library, 
  RotateCw, 
  User, 
  Users, 
  FileText, 
  File, 
  Trash2, 
  Check, 
  X, 
  Clock, 
  AlertCircle,
  CheckCircle2
} from "lucide-react";

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
      setTimeout(() => setConfirmDelete(null), 3000);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <Library className="w-4 h-4 text-violet-400" />
          <h2 className="text-xs font-semibold text-zinc-200 tracking-wide uppercase">
            Document Library
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <button
            id="doc-library-refresh"
            onClick={onRefresh}
            title="Refresh library"
            className="p-1 text-zinc-500 hover:text-zinc-300 transition-colors rounded hover:bg-zinc-800"
          >
            <RotateCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          </button>
          <button
            id="doc-library-scope-toggle"
            onClick={onToggleScope}
            className={`text-xs px-2 py-1 rounded border transition-all flex items-center gap-1 font-medium ${
              showAllDocs
                ? "border-violet-500/40 text-violet-300 bg-violet-500/10"
                : "border-zinc-800 text-zinc-400 hover:text-zinc-200"
            }`}
          >
            {showAllDocs ? (
              <>
                <Users className="w-3 h-3 text-violet-400" />
                <span>Workspace</span>
              </>
            ) : (
              <>
                <User className="w-3 h-3 text-zinc-400" />
                <span>My Docs</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1.5">
        {loading && documents.length === 0 && (
          <div className="flex flex-col items-center justify-center py-10 space-y-2">
            <RotateCw className="w-5 h-5 text-violet-400 animate-spin" />
            <span className="text-xs text-zinc-500">Loading library...</span>
          </div>
        )}

        {error && (
          <div className="text-xs text-rose-400 bg-rose-400/10 border border-rose-400/20 rounded-lg px-3 py-2.5 mx-1 flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
            <span className="flex-1 leading-snug">{error}</span>
          </div>
        )}

        {!loading && !error && documents.length === 0 && (
          <div className="text-center py-12 text-zinc-600 text-xs space-y-1">
            <File className="w-8 h-8 text-zinc-700 mx-auto mb-2 opacity-50" />
            <p className="font-medium text-zinc-400">No documents stored</p>
            <p className="text-zinc-600">Upload a PDF or TXT file to index</p>
          </div>
        )}

        {documents.map((doc) => {
          const isActive = doc.id === activeDocumentId;
          const canChat  = doc.ingestion_status === "succeeded" && doc.qdrant_collection;
          const statusCls = STATUS_COLORS[doc.ingestion_status] ?? STATUS_COLORS.unknown;

          return (
            <div
              key={doc.id}
              id={`doc-card-${doc.id.slice(0, 8)}`}
              className={`group relative rounded-xl border p-3 transition-all cursor-pointer ${
                isActive
                  ? "border-violet-500/50 bg-violet-500/10 shadow-sm"
                  : "border-zinc-800/80 bg-zinc-900/40 hover:border-zinc-700 hover:bg-zinc-800/50"
              }`}
              onClick={() => canChat && onSelect(doc)}
              title={canChat ? "Click to select document for chat" : "Indexing in progress"}
            >
              {/* Top Row: Icon + Name + Delete */}
              <div className="flex items-start gap-2.5">
                <FileText className={`w-4 h-4 mt-0.5 shrink-0 ${isActive ? "text-violet-400" : "text-zinc-400"}`} />
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-zinc-200 truncate leading-snug">
                    {doc.filename}
                  </p>
                  <p className="text-[10px] text-zinc-500 mt-0.5 font-mono">
                    {formatBytes(doc.file_size_bytes)} · {formatDate(doc.created_at)}
                  </p>
                </div>

                {/* Delete button */}
                <button
                  id={`doc-delete-${doc.id.slice(0, 8)}`}
                  onClick={(e) => { e.stopPropagation(); handleDelete(doc.id); }}
                  className={`p-1 rounded transition-all ${
                    confirmDelete === doc.id
                      ? "opacity-100 text-rose-400 bg-rose-500/10 border border-rose-500/30"
                      : "opacity-0 group-hover:opacity-100 text-zinc-500 hover:text-rose-400 hover:bg-zinc-800"
                  }`}
                  title={confirmDelete === doc.id ? "Click again to confirm delete" : "Delete document"}
                >
                  {confirmDelete === doc.id ? (
                    <span className="text-[10px] font-bold px-1 text-rose-400">Confirm?</span>
                  ) : (
                    <Trash2 className="w-3.5 h-3.5" />
                  )}
                </button>
              </div>

              {/* Bottom Row: Status Badge */}
              <div className="mt-2 flex items-center justify-between">
                <span className={`text-[10px] px-2 py-0.5 rounded border font-medium uppercase tracking-wider flex items-center gap-1 ${statusCls}`}>
                  {doc.ingestion_status === "succeeded" && <Check className="w-3 h-3 text-emerald-400" />}
                  {doc.ingestion_status === "running" && <RotateCw className="w-3 h-3 text-amber-400 animate-spin" />}
                  {doc.ingestion_status === "queued" && <Clock className="w-3 h-3 text-sky-400" />}
                  {doc.ingestion_status === "failed" && <X className="w-3 h-3 text-rose-400" />}
                  {doc.ingestion_status}
                </span>

                {isActive && (
                  <span className="text-[10px] text-violet-400 font-semibold flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" /> Active
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
