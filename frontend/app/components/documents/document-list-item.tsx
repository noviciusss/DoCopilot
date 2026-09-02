"use client";

import { useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { FileText, Trash2, Loader2, CheckCircle2, XCircle, Clock } from "lucide-react";
import { DocumentLibraryItem } from "../../../lib/hooks/useDocuments";

interface Props {
  doc: DocumentLibraryItem;
  isActive: boolean;
  onSelect: (doc: DocumentLibraryItem) => void;
  onDelete: (docId: string) => void;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
}

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case "succeeded":
      return <CheckCircle2 size={11} style={{ color: "var(--green)" }} aria-label="Indexed" />;
    case "running":
      return <Loader2 size={11} className="animate-spin" style={{ color: "var(--amber)" }} aria-label="Processing" />;
    case "queued":
      return <Clock size={11} style={{ color: "var(--amber)" }} aria-label="Queued" />;
    case "failed":
      return <XCircle size={11} style={{ color: "var(--red)" }} aria-label="Failed" />;
    default:
      return null;
  }
}

export default function DocumentListItem({ doc, isActive, onSelect, onDelete }: Props) {
  const [deleteOpen, setDeleteOpen] = useState(false);
  const canSelect = doc.ingestion_status === "succeeded" && !!doc.qdrant_collection;

  const handleDelete = () => {
    onDelete(doc.id);
    setDeleteOpen(false);
  };

  return (
    <div
      id={`doc-item-${doc.id.slice(0, 8)}`}
      role="listitem"
      className={[
        "group relative flex items-start gap-2.5 px-2.5 py-2 rounded-md cursor-pointer transition-colors",
        isActive
          ? "bg-[var(--cobalt-dim)]"
          : "hover:bg-[var(--surface-2)]",
        !canSelect && "opacity-70",
      ].filter(Boolean).join(" ")}
      onClick={() => canSelect && onSelect(doc)}
      tabIndex={canSelect ? 0 : -1}
      onKeyDown={(e) => {
        if ((e.key === "Enter" || e.key === " ") && canSelect) {
          e.preventDefault();
          onSelect(doc);
        }
      }}
      aria-current={isActive ? "true" : undefined}
      aria-label={`${doc.filename}${!canSelect ? " — not yet indexed" : ""}`}
      title={doc.filename}
    >
      {/* Icon */}
      <FileText
        size={14}
        style={{ color: isActive ? "var(--cobalt)" : "var(--text-2)", flexShrink: 0, marginTop: 1 }}
        aria-hidden="true"
      />

      {/* Content */}
      <div className="flex-1 min-w-0">
        <p
          className="text-xs font-medium leading-snug truncate"
          style={{ color: isActive ? "var(--text-1)" : "var(--text-1)" }}
        >
          {doc.filename}
        </p>
        <div className="flex items-center gap-1.5 mt-0.5">
          <StatusIcon status={doc.ingestion_status} />
          <span className="text-meta truncate">
            {formatBytes(doc.file_size_bytes)} · {formatDate(doc.created_at)}
          </span>
        </div>
      </div>

      {/* Delete — show on hover */}
      <Dialog.Root open={deleteOpen} onOpenChange={setDeleteOpen}>
        <Dialog.Trigger asChild>
          <button
            id={`doc-delete-${doc.id.slice(0, 8)}`}
            onClick={(e) => {
              e.stopPropagation();
              setDeleteOpen(true);
            }}
            className="opacity-0 group-hover:opacity-100 focus-visible:opacity-100 p-1 rounded transition-opacity"
            style={{ color: "var(--text-3)" }}
            aria-label={`Delete ${doc.filename}`}
            tabIndex={0}
          >
            <Trash2 size={12} aria-hidden="true" />
          </button>
        </Dialog.Trigger>

        <Dialog.Portal>
          <Dialog.Overlay
            className="fixed inset-0 z-50 bg-black/40"
            style={{ backdropFilter: "blur(2px)" }}
          />
          <Dialog.Content
            className="fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-xl p-5 w-80 shadow-xl"
            style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
            }}
            aria-describedby="delete-desc"
          >
            <Dialog.Title
              className="text-sm font-semibold mb-1"
              style={{ color: "var(--text-1)" }}
            >
              Delete document?
            </Dialog.Title>
            <Dialog.Description
              id="delete-desc"
              className="text-xs mb-4"
              style={{ color: "var(--text-2)" }}
            >
              <span className="font-medium" style={{ color: "var(--text-1)" }}>
                {doc.filename}
              </span>{" "}
              will be permanently removed along with its indexed data. This cannot be undone.
            </Dialog.Description>

            <div className="flex items-center justify-end gap-2">
              <Dialog.Close asChild>
                <button
                  className="px-3 py-1.5 text-xs rounded-md transition-colors"
                  style={{
                    background: "var(--surface-2)",
                    border: "1px solid var(--border)",
                    color: "var(--text-2)",
                  }}
                >
                  Cancel
                </button>
              </Dialog.Close>
              <button
                onClick={handleDelete}
                className="px-3 py-1.5 text-xs rounded-md font-medium transition-colors"
                style={{
                  background: "var(--red-dim)",
                  border: "1px solid var(--red)",
                  color: "var(--red)",
                }}
              >
                Delete
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}
