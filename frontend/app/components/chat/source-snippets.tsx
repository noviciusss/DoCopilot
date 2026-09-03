"use client";

import { useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { motion } from "motion/react";
import { FileSearch, X, ChevronDown } from "lucide-react";

interface Props {
  sources: string[];
}

const PREVIEW_LEN = 90;

function SourceChip({ source, index }: { source: string; index: number }) {
  const [open, setOpen] = useState(false);
  const preview = source.length > PREVIEW_LEN
    ? source.slice(0, PREVIEW_LEN).trimEnd() + "…"
    : source;

  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Trigger asChild>
        {/* Subtle hover/tap lift — no fake citation behavior */}
        <motion.button
          className="group flex items-start gap-1.5 text-left w-full rounded-md px-2.5 py-2"
          style={{
            background: "var(--ink)",
            border: "1px solid var(--border)",
            color: "var(--text-2)",
            fontFamily: "var(--font-sans)",
            cursor: "pointer",
          }}
          whileHover={{ y: -1, borderColor: "var(--amber)" }}
          whileTap={{ scale: 0.985 }}
          transition={{ duration: 0.12, ease: [0.16, 1, 0.3, 1] }}
          aria-label={`Source snippet ${index + 1} — click to expand`}
        >
          <span
            className="flex-shrink-0 w-4 h-4 rounded text-center leading-4 text-xs font-mono font-medium mt-0.5"
            style={{ background: "var(--amber-dim)", color: "var(--amber)" }}
            aria-hidden="true"
          >
            {index + 1}
          </span>
          <span
            className="flex-1 text-xs leading-snug break-words min-w-0"
            style={{ color: "var(--text-2)", wordBreak: "break-word", overflowWrap: "anywhere" }}
          >
            {/* Preview is safe — rendered as text node, never innerHTML */}
            {preview}
          </span>
          {source.length > PREVIEW_LEN && (
            <ChevronDown
              size={11}
              className="flex-shrink-0 mt-1 group-hover:opacity-100 opacity-50 transition-opacity"
              style={{ color: "var(--amber)" }}
              aria-hidden="true"
            />
          )}
        </motion.button>
      </Dialog.Trigger>

      <Dialog.Portal>
        <Dialog.Overlay
          className="fixed inset-0 z-50 bg-black/40"
          style={{ backdropFilter: "blur(2px)" }}
        />
        <Dialog.Content
          className="fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-xl shadow-xl flex flex-col"
          style={{
            background: "var(--surface)",
            border: "1px solid var(--border)",
            width: "min(560px, calc(100vw - 2rem))",
            maxHeight: "70vh",
          }}
          aria-describedby={`source-full-${index}`}
        >
          <div
            className="flex items-center justify-between px-4 py-3 flex-shrink-0"
            style={{ borderBottom: "1px solid var(--border)" }}
          >
            <Dialog.Title
              className="text-xs font-semibold flex items-center gap-1.5"
              style={{ color: "var(--text-2)" }}
            >
              <span
                className="w-4 h-4 rounded text-center leading-4 text-xs font-mono font-medium"
                style={{ background: "var(--amber-dim)", color: "var(--amber)" }}
                aria-hidden="true"
              >
                {index + 1}
              </span>
              Source snippet · retrieved context
            </Dialog.Title>
            <Dialog.Close asChild>
              <button
                className="p-1 rounded transition-colors"
                style={{ color: "var(--text-3)" }}
                aria-label="Close source snippet"
              >
                <X size={14} aria-hidden="true" />
              </button>
            </Dialog.Close>
          </div>

          <div className="flex-1 overflow-y-auto px-4 py-3">
            {/* Rendered as <p> text — safe, no dangerouslySetInnerHTML */}
            <p
              id={`source-full-${index}`}
              className="text-xs leading-relaxed"
              style={{
                color: "var(--text-1)",
                fontFamily: "var(--font-sans)",
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
                overflowWrap: "anywhere",
              }}
            >
              {source}
            </p>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

export default function SourceSnippets({ sources }: Props) {
  if (!sources.length) return null;

  return (
    <div className="space-y-2" role="region" aria-label="Retrieved context snippets">
      <div className="flex items-center gap-1.5">
        <FileSearch size={12} style={{ color: "var(--amber)" }} aria-hidden="true" />
        <span className="text-label" style={{ color: "var(--amber)" }}>
          Retrieved context · {sources.length} {sources.length === 1 ? "snippet" : "snippets"}
        </span>
      </div>
      <div className="space-y-1.5">
        {sources.map((src, i) => (
          <SourceChip key={i} source={src} index={i} />
        ))}
      </div>
    </div>
  );
}
