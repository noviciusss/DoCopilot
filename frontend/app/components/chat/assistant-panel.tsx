"use client";

import { FormEvent, useRef, useEffect } from "react";
import { motion, MotionConfig } from "motion/react";
import { Copy, Check, RotateCcw } from "lucide-react";
import ChatMessage from "./chat-message";
import ChatComposer from "./chat-composer";
import SourceSnippets from "./source-snippets";
import SuggestedPrompts from "./suggested-prompts";

interface Props {
  activeDocName: string | null;
  question: string;
  onQuestionChange: (v: string) => void;
  streamingAnswer: string;
  sources: string[];
  isAsking: boolean;
  hasDocument: boolean;
  copied: boolean;
  onSubmit: (e: FormEvent<HTMLFormElement>) => void;
  onCopy: () => void;
  onClear: () => void;
  onSuggestedPrompt: (p: string) => void;
}

const askDisabled = (hasDocument: boolean, question: string, isAsking: boolean) =>
  !hasDocument || !question.trim() || isAsking;

export default function AssistantPanel({
  activeDocName,
  question,
  onQuestionChange,
  streamingAnswer,
  sources,
  isAsking,
  hasDocument,
  copied,
  onSubmit,
  onCopy,
  onClear,
  onSuggestedPrompt,
}: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const hasConversation = !!(streamingAnswer || isAsking);

  // Auto-scroll while streaming
  useEffect(() => {
    if (streamingAnswer && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [streamingAnswer]);

  return (
    <MotionConfig reducedMotion="user">
      <div className="chat-panel" role="complementary" aria-label="Document assistant">
        {/* Header */}
        <div
          className="flex items-center justify-between px-4 py-3 flex-shrink-0"
          style={{ borderBottom: "1px solid var(--border)" }}
        >
          <div className="min-w-0 flex-1">
            <h2 className="text-xs font-semibold" style={{ color: "var(--text-1)" }}>
              Assistant
            </h2>
            {activeDocName && (
              <p
                className="text-meta truncate max-w-[220px]"
                title={activeDocName}
                style={{ color: "var(--text-3)" }}
              >
                {activeDocName}
              </p>
            )}
          </div>

          {hasConversation && (
            <button
              type="button"
              onClick={onClear}
              className="p-1.5 rounded-md transition-colors flex-shrink-0"
              style={{ color: "var(--text-3)" }}
              onMouseEnter={(e) => (e.currentTarget.style.color = "var(--text-1)")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-3)")}
              aria-label="Clear conversation"
              title="Clear conversation"
            >
              <RotateCcw size={13} aria-hidden="true" />
            </button>
          )}
        </div>

        {/* Messages area */}
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto px-4 py-4"
          aria-live="polite"
          aria-label="Conversation"
        >
          {!hasConversation ? (
            <SuggestedPrompts onSelect={onSuggestedPrompt} hasDocument={hasDocument} />
          ) : (
            <div className="space-y-4">
              {/* Answer */}
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
                className="space-y-3"
              >
                {/* Answer block */}
                <div
                  className="rounded-lg p-3 space-y-2"
                  style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
                >
                  <div
                    className="flex items-center justify-between pb-1.5"
                    style={{ borderBottom: "1px solid var(--border)" }}
                  >
                    <span className="text-label">Answer</span>
                    {streamingAnswer && !isAsking && (
                      <button
                        type="button"
                        onClick={onCopy}
                        className="flex items-center gap-1 text-xs rounded transition-colors px-1.5 py-0.5"
                        style={{ color: "var(--text-3)" }}
                        onMouseEnter={(e) => (e.currentTarget.style.color = "var(--text-1)")}
                        onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-3)")}
                        aria-label={copied ? "Copied!" : "Copy answer"}
                      >
                        {copied ? (
                          <Check size={12} style={{ color: "var(--green)" }} aria-hidden="true" />
                        ) : (
                          <Copy size={12} aria-hidden="true" />
                        )}
                        <span>{copied ? "Copied" : "Copy"}</span>
                      </button>
                    )}
                  </div>

                  {streamingAnswer ? (
                    <ChatMessage content={streamingAnswer} isStreaming={isAsking} />
                  ) : isAsking ? (
                    <div className="flex items-center gap-2 py-2">
                      <div className="flex gap-1" aria-hidden="true">
                        {[0, 1, 2].map((i) => (
                          <motion.div
                            key={i}
                            className="w-1.5 h-1.5 rounded-full"
                            style={{ background: "var(--text-3)" }}
                            animate={{ opacity: [0.3, 1, 0.3] }}
                            transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }}
                          />
                        ))}
                      </div>
                      <span className="text-xs" style={{ color: "var(--text-3)" }}>
                        Retrieving context and generating answer…
                      </span>
                    </div>
                  ) : null}
                </div>

                {/* Sources */}
                {sources.length > 0 && !isAsking && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.2, delay: 0.1 }}
                  >
                    <SourceSnippets sources={sources} />
                  </motion.div>
                )}
              </motion.div>

              <div ref={bottomRef} />
            </div>
          )}
        </div>

        {/* Composer — pinned bottom */}
        <ChatComposer
          value={question}
          onChange={onQuestionChange}
          onSubmit={onSubmit}
          disabled={askDisabled(hasDocument, question, isAsking)}
          isAsking={isAsking}
          hasDocument={hasDocument}
        />
      </div>
    </MotionConfig>
  );
}
