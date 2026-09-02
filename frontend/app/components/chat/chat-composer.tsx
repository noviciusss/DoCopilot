"use client";

import { FormEvent, KeyboardEvent, useRef, useEffect } from "react";
import { Send, Loader2 } from "lucide-react";

interface Props {
  value: string;
  onChange: (v: string) => void;
  onSubmit: (e: FormEvent<HTMLFormElement>) => void;
  disabled: boolean;
  isAsking: boolean;
  hasDocument: boolean;
}

export default function ChatComposer({
  value,
  onChange,
  onSubmit,
  disabled,
  isAsking,
  hasDocument,
}: Props) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [value]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!disabled) {
        e.currentTarget.form?.requestSubmit();
      }
    }
  };

  const placeholder = hasDocument
    ? "Ask a question about this document… (Enter to send)"
    : "Select or upload a document to start asking questions";

  return (
    <div
      className="flex-shrink-0 px-3 py-3"
      style={{ borderTop: "1px solid var(--border)" }}
    >
      <form onSubmit={onSubmit} className="relative">
        <textarea
          id="chat-question-input"
          ref={textareaRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={!hasDocument || isAsking}
          placeholder={placeholder}
          rows={1}
          className="w-full rounded-lg px-3 py-2.5 pr-10 text-xs resize-none leading-relaxed transition-colors"
          style={{
            background:   "var(--surface-2)",
            border:       "1px solid var(--border)",
            color:        "var(--text-1)",
            outline:      "none",
            fontFamily:   "var(--font-sans)",
            minHeight:    "40px",
            maxHeight:    "160px",
            overflowY:    "auto",
          }}
          onFocus={(e) => (e.currentTarget.style.borderColor = "var(--cobalt)")}
          onBlur={(e) => (e.currentTarget.style.borderColor = "var(--border)")}
          aria-label="Question input"
        />

        <button
          id="send-chat-button"
          type="submit"
          disabled={disabled}
          className="absolute right-2 bottom-2 p-1.5 rounded-md transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          style={{
            background: disabled ? "transparent" : "var(--cobalt)",
            color: "#fff",
          }}
          aria-label={isAsking ? "Generating answer…" : "Send question"}
        >
          {isAsking ? (
            <Loader2 size={13} className="animate-spin" aria-hidden="true" />
          ) : (
            <Send size={13} aria-hidden="true" />
          )}
        </button>
      </form>

      <p className="text-meta mt-1.5 text-center" style={{ fontSize: "10px", color: "var(--text-3)" }}>
        Shift + Enter for new line · answers include retrieved context snippets
      </p>
    </div>
  );
}
