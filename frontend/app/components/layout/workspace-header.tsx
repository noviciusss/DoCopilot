"use client";

import { Menu, FileText, Loader2, CheckCircle2, XCircle, Clock, X } from "lucide-react";
import { DocumentLibraryItem } from "../../../lib/hooks/useDocuments";
import { IngestionStatus } from "../../../lib/hooks/useUpload";

interface Props {
  activeDoc: DocumentLibraryItem | null | undefined;
  effectiveStatus: IngestionStatus;
  onClearSession: () => void;
  onOpenMobileSidebar: () => void;
  onOpenMobileChat: () => void;
}

function StatusBadge({ status }: { status: IngestionStatus | string }) {
  const config: Record<string, { label: string; color: string; icon: React.ReactNode }> = {
    succeeded: { label: "Indexed",    color: "var(--green)",  icon: <CheckCircle2 size={11} aria-hidden="true" /> },
    running:   { label: "Processing", color: "var(--amber)",  icon: <Loader2 size={11} className="animate-spin" aria-hidden="true" /> },
    queued:    { label: "Queued",     color: "var(--amber)",  icon: <Clock size={11} aria-hidden="true" /> },
    uploading: { label: "Uploading",  color: "var(--cobalt)", icon: <Loader2 size={11} className="animate-spin" aria-hidden="true" /> },
    failed:    { label: "Failed",     color: "var(--red)",    icon: <XCircle size={11} aria-hidden="true" /> },
    idle:      { label: "Not indexed",color: "var(--text-3)", icon: null },
  };

  const c = config[status] ?? config.idle;

  return (
    <span
      className="flex items-center gap-1 text-xs px-2 py-0.5 rounded"
      style={{
        background: `color-mix(in srgb, ${c.color} 12%, transparent)`,
        color: c.color,
        border: `1px solid color-mix(in srgb, ${c.color} 30%, transparent)`,
      }}
      aria-label={`Document status: ${c.label}`}
    >
      {c.icon}
      {c.label}
    </span>
  );
}

export default function WorkspaceHeader({
  activeDoc,
  effectiveStatus,
  onClearSession,
  onOpenMobileSidebar,
  onOpenMobileChat,
}: Props) {
  return (
    <header
      className="flex items-center gap-3 px-4 py-2.5 flex-shrink-0"
      style={{ borderBottom: "1px solid var(--border)", background: "var(--ink)" }}
    >
      {/* Mobile sidebar toggle */}
      <button
        id="mobile-sidebar-toggle"
        onClick={onOpenMobileSidebar}
        className="lg:hidden p-1.5 rounded-md transition-colors flex-shrink-0"
        style={{ color: "var(--text-3)" }}
        aria-label="Open sidebar navigation"
      >
        <Menu size={16} aria-hidden="true" />
      </button>

      {/* Document name */}
      <div className="flex-1 min-w-0 flex items-center gap-2.5">
        {activeDoc ? (
          <>
            <FileText
              size={14}
              style={{ color: "var(--text-3)", flexShrink: 0 }}
              aria-hidden="true"
            />
            <span
              className="text-xs font-medium truncate"
              style={{ color: "var(--text-1)" }}
              title={activeDoc.filename}
            >
              {activeDoc.filename}
            </span>
            <StatusBadge status={effectiveStatus} />
            <button
              onClick={onClearSession}
              className="hidden sm:flex items-center gap-1 px-2 py-0.5 rounded text-xs transition-colors flex-shrink-0"
              style={{
                color: "var(--text-3)",
                border: "1px solid var(--border)",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.color = "var(--text-1)";
                e.currentTarget.style.borderColor = "var(--border-light)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.color = "var(--text-3)";
                e.currentTarget.style.borderColor = "var(--border)";
              }}
              aria-label="Deselect current document"
            >
              <X size={11} aria-hidden="true" />
              Unload
            </button>
          </>
        ) : (
          <span className="text-xs" style={{ color: "var(--text-3)" }}>
            No document selected
          </span>
        )}
      </div>

      {/* Mobile chat toggle */}
      <button
        id="mobile-chat-toggle"
        onClick={onOpenMobileChat}
        className="lg:hidden px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors flex-shrink-0"
        style={{
          background: "var(--cobalt-dim)",
          color: "var(--cobalt)",
          border: "1px solid var(--cobalt-ring)",
        }}
        aria-label="Open chat assistant"
      >
        Chat
      </button>
    </header>
  );
}
