"use client";

import { useState, useEffect, useRef } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { motion, AnimatePresence } from "motion/react";
import { Search, FileText, X } from "lucide-react";
import { DocumentLibraryItem } from "../../../lib/hooks/useDocuments";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  documents: DocumentLibraryItem[];
  onSelectDocument: (doc: DocumentLibraryItem) => void;
}

export default function CommandPalette({
  open,
  onOpenChange,
  documents,
  onSelectDocument,
}: Props) {
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  // Filter only indexed documents by filename
  const filtered = documents.filter(
    (d) =>
      d.ingestion_status === "succeeded" &&
      d.filename.toLowerCase().includes(query.toLowerCase())
  );

  // Reset query when dialog closes — intentional UI sync on dialog open state change
  useEffect(() => {
    if (!open) setQuery(""); // eslint-disable-line react-hooks/set-state-in-effect
  }, [open]);

  const handleSelect = (doc: DocumentLibraryItem) => {
    onSelectDocument(doc);
    onOpenChange(false);
  };

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay
          className="fixed inset-0 z-50 bg-black/50"
          style={{ backdropFilter: "blur(3px)" }}
        />
        <Dialog.Content
          className="fixed z-50 top-[20%] left-1/2 -translate-x-1/2 rounded-xl shadow-2xl overflow-hidden"
          style={{
            width: "min(540px, calc(100vw - 2rem))",
            background: "var(--surface)",
            border: "1px solid var(--border)",
          }}
          aria-describedby="palette-desc"
        >
          <Dialog.Title className="sr-only">Document search</Dialog.Title>
          <Dialog.Description id="palette-desc" className="sr-only">
            Search your indexed documents by filename. Client-side filtering only.
          </Dialog.Description>

          {/* Search input */}
          <div
            className="flex items-center gap-2.5 px-3.5 py-3"
            style={{ borderBottom: "1px solid var(--border)" }}
          >
            <Search size={15} style={{ color: "var(--text-3)", flexShrink: 0 }} aria-hidden="true" />
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search documents…"
              className="flex-1 bg-transparent text-sm outline-none"
              style={{ color: "var(--text-1)", fontFamily: "var(--font-sans)" }}
              autoComplete="off"
              spellCheck={false}
              aria-label="Search documents"
              aria-controls="palette-results"
            />
            <Dialog.Close asChild>
              <button
                className="p-1 rounded transition-colors"
                style={{ color: "var(--text-3)" }}
                aria-label="Close search"
              >
                <X size={13} aria-hidden="true" />
              </button>
            </Dialog.Close>
          </div>

          {/* Results */}
          <div
            id="palette-results"
            role="listbox"
            aria-label="Document results"
            className="overflow-y-auto"
            style={{ maxHeight: "280px" }}
          >
            {filtered.length === 0 ? (
              <div className="px-4 py-8 text-center">
                <p className="text-xs" style={{ color: "var(--text-3)" }}>
                  {query
                    ? "No indexed documents match this search"
                    : "No indexed documents yet — upload a file to get started"}
                </p>
              </div>
            ) : (
              <AnimatePresence initial={false}>
                {filtered.map((doc) => (
                  <motion.button
                    key={doc.id}
                    layout="position"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.15 }}
                    role="option"
                    aria-selected="false"
                    onClick={() => handleSelect(doc)}
                    className="w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors"
                    style={{ color: "var(--text-1)" }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = "var(--cobalt-dim)")}
                    onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                    onFocus={(e) => (e.currentTarget.style.background = "var(--cobalt-dim)")}
                    onBlur={(e) => (e.currentTarget.style.background = "transparent")}
                  >
                    <FileText size={14} style={{ color: "var(--text-3)", flexShrink: 0 }} aria-hidden="true" />
                    <span className="flex-1 min-w-0 text-xs truncate">{doc.filename}</span>
                    <span className="text-meta flex-shrink-0">
                      {(doc.file_size_bytes / 1024).toFixed(0)} KB
                    </span>
                  </motion.button>
                ))}
              </AnimatePresence>
            )}
          </div>

          {/* Footer */}
          <div
            className="px-4 py-2 flex items-center justify-between"
            style={{ borderTop: "1px solid var(--border)" }}
          >
            <p className="text-meta" style={{ fontSize: "10px" }}>
              Filtering {documents.filter((d) => d.ingestion_status === "succeeded").length} indexed documents
            </p>
            <p className="text-meta" style={{ fontSize: "10px" }}>
              Esc to close
            </p>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
