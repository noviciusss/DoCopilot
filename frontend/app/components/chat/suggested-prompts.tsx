"use client";

import { MessageSquare } from "lucide-react";

interface Props {
  onSelect: (prompt: string) => void;
  hasDocument: boolean;
}

const PROMPTS_WITH_DOC = [
  "Summarize the key points of this document",
  "What are the main conclusions or recommendations?",
  "List the most important terms or concepts defined here",
  "What problem does this document address?",
];

const PROMPTS_NO_DOC = [
  "Upload a PDF to ask questions about it",
  "Select a document from the sidebar to begin",
];

export default function SuggestedPrompts({ onSelect, hasDocument }: Props) {
  const prompts = hasDocument ? PROMPTS_WITH_DOC : PROMPTS_NO_DOC;

  return (
    <div className="flex flex-col items-center justify-center h-full px-4 py-8 text-center space-y-4">
      <div
        className="w-9 h-9 rounded-lg flex items-center justify-center"
        style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
      >
        <MessageSquare size={16} style={{ color: "var(--text-3)" }} aria-hidden="true" />
      </div>

      <div className="space-y-0.5">
        <p className="text-xs font-medium" style={{ color: "var(--text-1)" }}>
          {hasDocument ? "Ask about the document" : "No document selected"}
        </p>
        <p className="text-xs" style={{ color: "var(--text-3)" }}>
          {hasDocument
            ? "Type your question below or try a suggestion"
            : "Upload or select a document from the sidebar"}
        </p>
      </div>

      {hasDocument && (
        <ul className="w-full space-y-1.5 text-left" aria-label="Suggested questions">
          {prompts.map((p) => (
            <li key={p}>
              <button
                type="button"
                onClick={() => onSelect(p)}
                className="w-full text-left text-xs rounded-md px-3 py-2 transition-colors leading-snug"
                style={{
                  background: "var(--surface-2)",
                  border: "1px solid var(--border)",
                  color: "var(--text-2)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = "var(--cobalt)";
                  e.currentTarget.style.color = "var(--text-1)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = "var(--border)";
                  e.currentTarget.style.color = "var(--text-2)";
                }}
              >
                {p}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
