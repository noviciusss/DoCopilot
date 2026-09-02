"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Components } from "react-markdown";
import type { AnchorHTMLAttributes } from "react";

// Safe link renderer — all external links get rel + target
const SafeLink = ({
  href,
  children,
  ...props
}: AnchorHTMLAttributes<HTMLAnchorElement>) => (
  <a
    href={href}
    {...props}
    target="_blank"
    rel="noreferrer noopener"
    style={{ color: "var(--cobalt)" }}
  >
    {children}
  </a>
);

// Markdown component overrides — typed correctly
const components: Components = {
  a: SafeLink as Components["a"],
};

interface Props {
  content: string;
  isStreaming?: boolean;
}

export default function ChatMessage({ content, isStreaming }: Props) {
  return (
    <div
      className="prose prose-sm max-w-none prose-doc"
      style={{
        fontSize: "var(--text-base)",
        lineHeight: 1.65,
        color: "var(--text-1)",
      }}
    >
      {/* ReactMarkdown renders via VDOM — no dangerouslySetInnerHTML */}
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {content}
      </ReactMarkdown>
      {isStreaming && (
        <span
          className="inline-block w-0.5 h-3.5 ml-0.5 align-middle animate-pulse"
          style={{ background: "var(--cobalt)" }}
          aria-hidden="true"
        />
      )}
    </div>
  );
}
