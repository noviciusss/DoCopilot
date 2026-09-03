"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { motion, MotionConfig } from "motion/react";
import {
  Plus,
  Search,
  RotateCw,
  User,
  Users,
  LogOut,
  X,
  FilePlus,
  PanelLeftClose,
  PanelLeftOpen,
} from "lucide-react";
import Logo from "../Logo";
import DocumentListItem from "../documents/document-list-item";
import { DocumentLibraryItem } from "../../../lib/hooks/useDocuments";

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
  onNewDocument: () => void;
  onOpenSearch: () => void;
  onLogout: () => void;
  // Desktop collapse state
  collapsed: boolean;
  onToggleCollapse: () => void;
  // Mobile drawer state
  mobileOpen: boolean;
  onMobileOpenChange: (v: boolean) => void;
}

// ── Shared inner content ───────────────────────────────────────────────────────

interface InnerProps extends Omit<Props, "collapsed" | "onToggleCollapse" | "mobileOpen" | "onMobileOpenChange"> {
  onClose?: () => void;
  showCollapseToggle?: boolean;
  collapsed?: boolean;
  onToggleCollapse?: () => void;
}

function SidebarInner({
  documents,
  loading,
  error,
  showAllDocs,
  activeDocumentId,
  onSelect,
  onDelete,
  onToggleScope,
  onRefresh,
  onNewDocument,
  onOpenSearch,
  onLogout,
  onClose,
  showCollapseToggle,
  collapsed,
  onToggleCollapse,
}: InnerProps) {
  return (
    <div className="flex flex-col h-full overflow-hidden">

      {/* ── Top: wordmark + collapse toggle ───────────────────────────────── */}
      <div
        className="flex items-center justify-between px-3 py-3 flex-shrink-0"
        style={{ borderBottom: "1px solid var(--border)" }}
      >
        {!collapsed && <Logo size="sm" />}

        <div className="flex items-center gap-1 ml-auto">
          {/* Desktop collapse toggle */}
          {showCollapseToggle && (
            <button
              onClick={onToggleCollapse}
              className="p-1.5 rounded-md transition-colors"
              style={{ color: "var(--text-3)" }}
              onMouseEnter={(e) => (e.currentTarget.style.color = "var(--text-1)")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-3)")}
              aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
              title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              {collapsed
                ? <PanelLeftOpen size={15} aria-hidden="true" />
                : <PanelLeftClose size={15} aria-hidden="true" />}
            </button>
          )}
          {/* Mobile close */}
          {onClose && (
            <button
              onClick={onClose}
              className="p-1.5 rounded-md transition-colors"
              style={{ color: "var(--text-3)" }}
              aria-label="Close sidebar"
            >
              <X size={15} aria-hidden="true" />
            </button>
          )}
        </div>
      </div>

      {/* ── Actions ─────────────────────────────────────────────────────────── */}
      <div className="px-2 py-2 space-y-0.5 flex-shrink-0">
        {collapsed ? (
          /* Icon-only rail */
          <>
            <button
              onClick={onNewDocument}
              className="w-full flex items-center justify-center p-2 rounded-md transition-colors"
              style={{ color: "var(--cobalt)", background: "var(--cobalt-dim)" }}
              onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(79,107,255,0.2)")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "var(--cobalt-dim)")}
              aria-label="Upload new document"
              title="New document"
            >
              <Plus size={15} aria-hidden="true" />
            </button>
            <button
              id="open-search-button"
              onClick={onOpenSearch}
              className="w-full flex items-center justify-center p-2 rounded-md transition-colors"
              style={{ color: "var(--text-2)" }}
              onMouseEnter={(e) => { e.currentTarget.style.background = "var(--surface-2)"; e.currentTarget.style.color = "var(--text-1)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "var(--text-2)"; }}
              aria-label="Search documents (⌘K)"
              title="Search (⌘K)"
            >
              <Search size={14} aria-hidden="true" />
            </button>
          </>
        ) : (
          /* Full labels */
          <>
            <button
              onClick={onNewDocument}
              className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-md text-xs font-medium transition-colors"
              style={{ color: "var(--cobalt)", background: "var(--cobalt-dim)" }}
              onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(79,107,255,0.2)")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "var(--cobalt-dim)")}
              aria-label="Upload new document"
            >
              <Plus size={14} aria-hidden="true" />
              New document
            </button>
            <button
              id="open-search-button"
              onClick={onOpenSearch}
              className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-md text-xs transition-colors"
              style={{ color: "var(--text-2)" }}
              onMouseEnter={(e) => { e.currentTarget.style.background = "var(--surface-2)"; e.currentTarget.style.color = "var(--text-1)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "var(--text-2)"; }}
              aria-label="Search documents (⌘K)"
            >
              <Search size={14} aria-hidden="true" />
              Search
              <span className="ml-auto text-meta" style={{ fontSize: "10px" }}>⌘K</span>
            </button>
          </>
        )}
      </div>

      {/* ── Library section ─────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-h-0" style={{ borderTop: "1px solid var(--border)" }}>

        {/* Library header */}
        <div className="flex items-center justify-between px-3 py-2 flex-shrink-0">
          {!collapsed && <span className="text-label">Documents</span>}
          <div className={`flex items-center gap-1 ${collapsed ? "w-full justify-center" : ""}`}>
            <button
              id="doc-library-scope-toggle"
              onClick={onToggleScope}
              className="p-1 rounded transition-colors"
              style={{ color: showAllDocs ? "var(--cobalt)" : "var(--text-3)" }}
              onMouseEnter={(e) => (e.currentTarget.style.color = "var(--text-1)")}
              onMouseLeave={(e) => (e.currentTarget.style.color = showAllDocs ? "var(--cobalt)" : "var(--text-3)")}
              aria-label={showAllDocs ? "Showing all workspace docs — switch to my docs" : "Showing my docs — switch to all workspace"}
              title={showAllDocs ? "All workspace" : "My docs"}
            >
              {showAllDocs ? <Users size={12} aria-hidden="true" /> : <User size={12} aria-hidden="true" />}
            </button>
            <button
              id="doc-library-refresh"
              onClick={onRefresh}
              className="p-1 rounded transition-colors"
              style={{ color: "var(--text-3)" }}
              onMouseEnter={(e) => (e.currentTarget.style.color = "var(--text-1)")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-3)")}
              aria-label="Refresh document list"
            >
              <RotateCw size={12} className={loading ? "animate-spin" : ""} aria-hidden="true" />
            </button>
          </div>
        </div>

        {/* Document list */}
        <div
          className="flex-1 overflow-y-auto px-2 pb-2"
          role="list"
          aria-label={showAllDocs ? "All workspace documents" : "My documents"}
        >
          {loading && documents.length === 0 && (
            <div className="flex flex-col items-center justify-center py-8 space-y-2">
              <RotateCw size={14} className="animate-spin" style={{ color: "var(--text-3)" }} aria-hidden="true" />
              {!collapsed && <span className="text-xs" style={{ color: "var(--text-3)" }}>Loading…</span>}
            </div>
          )}

          {error && !collapsed && (
            <div
              className="mx-1 my-2 px-3 py-2 rounded-md text-xs leading-snug"
              style={{ background: "var(--red-dim)", border: "1px solid var(--red)", color: "var(--red)" }}
              role="alert"
            >
              {error}
            </div>
          )}

          {!loading && !error && documents.length === 0 && !collapsed && (
            <div className="flex flex-col items-center justify-center py-10 text-center px-3">
              <FilePlus size={22} style={{ color: "var(--text-3)", opacity: 0.5, marginBottom: 8 }} aria-hidden="true" />
              <p className="text-xs font-medium" style={{ color: "var(--text-2)" }}>No documents</p>
              <p className="text-meta mt-0.5">Upload a file to get started</p>
            </div>
          )}

          {documents.map((doc) =>
            collapsed ? (
              /* Icon-only dot for each doc when collapsed */
              <button
                key={doc.id}
                onClick={() => doc.ingestion_status === "succeeded" && doc.qdrant_collection && onSelect(doc)}
                className="w-full flex items-center justify-center py-2 rounded-md transition-colors"
                style={{
                  background: doc.id === activeDocumentId ? "var(--cobalt-dim)" : "transparent",
                  color: doc.id === activeDocumentId ? "var(--cobalt)" : "var(--text-3)",
                }}
                onMouseEnter={(e) => { if (doc.id !== activeDocumentId) e.currentTarget.style.background = "var(--surface-2)"; }}
                onMouseLeave={(e) => { if (doc.id !== activeDocumentId) e.currentTarget.style.background = "transparent"; }}
                aria-label={doc.filename}
                title={doc.filename}
              >
                <div className="w-1.5 h-1.5 rounded-full" style={{
                  background: doc.id === activeDocumentId ? "var(--cobalt)"
                    : doc.ingestion_status === "succeeded" ? "var(--green)"
                    : doc.ingestion_status === "failed" ? "var(--red)"
                    : "var(--amber)"
                }} />
              </button>
            ) : (
              /* Full list item with Motion layoutId selection indicator */
              <div key={doc.id} className="relative">
                {doc.id === activeDocumentId && (
                  <motion.div
                    layoutId="active-doc-indicator"
                    className="absolute left-0 top-1 bottom-1 w-0.5 rounded-full"
                    style={{ background: "var(--cobalt)", zIndex: 1 }}
                    transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
                  />
                )}
                <DocumentListItem
                  doc={doc}
                  isActive={doc.id === activeDocumentId}
                  onSelect={(d) => { onSelect(d); onClose?.(); }}
                  onDelete={onDelete}
                />
              </div>
            )
          )}
        </div>
      </div>

      {/* ── Bottom: account ──────────────────────────────────────────────────── */}
      <div className="flex-shrink-0 px-2 py-2" style={{ borderTop: "1px solid var(--border)" }}>
        <button
          onClick={onLogout}
          className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-md text-xs transition-colors ${collapsed ? "justify-center" : ""}`}
          style={{ color: "var(--text-3)" }}
          onMouseEnter={(e) => { e.currentTarget.style.background = "var(--red-dim)"; e.currentTarget.style.color = "var(--red)"; }}
          onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "var(--text-3)"; }}
          aria-label="Sign out"
          title="Sign out"
        >
          <LogOut size={13} aria-hidden="true" />
          {!collapsed && "Sign out"}
        </button>
      </div>
    </div>
  );
}

// ── Main export ────────────────────────────────────────────────────────────────

export default function AppSidebar(props: Props) {
  const { collapsed, onToggleCollapse, mobileOpen, onMobileOpenChange, ...rest } = props;

  const desktopWidth = collapsed ? "48px" : "var(--sidebar-w)";

  return (
    <>
      {/* Desktop sidebar — animated width */}
      <MotionConfig reducedMotion="user">
        <motion.nav
          aria-label="Main navigation"
          style={{
            width: desktopWidth,
            background: "var(--surface)",
            borderRight: "1px solid var(--border)",
            display: "flex",
            flexDirection: "column",
            flexShrink: 0,
            height: "100vh",
            position: "fixed",
            top: 0,
            left: 0,
            zIndex: 40,
            overflow: "hidden",
          }}
          animate={{ width: collapsed ? 48 : 260 }}
          transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
          className="hidden lg:flex"
        >
          <SidebarInner
            {...rest}
            collapsed={collapsed}
            onToggleCollapse={onToggleCollapse}
            showCollapseToggle
          />
        </motion.nav>
      </MotionConfig>

      {/* Mobile drawer — Radix Dialog */}
      <Dialog.Root open={mobileOpen} onOpenChange={onMobileOpenChange}>
        <Dialog.Portal>
          <Dialog.Overlay
            className="fixed inset-0 z-50 bg-black/50 lg:hidden"
            style={{ backdropFilter: "blur(2px)" }}
          />
          <Dialog.Content
            className="fixed inset-y-0 left-0 z-50 lg:hidden flex flex-col shadow-2xl"
            style={{
              width: "280px",
              background: "var(--surface)",
              borderRight: "1px solid var(--border)",
            }}
            aria-describedby="mobile-sidebar-desc"
          >
            <Dialog.Title className="sr-only">Navigation sidebar</Dialog.Title>
            <Dialog.Description id="mobile-sidebar-desc" className="sr-only">
              Document library, navigation, and account actions
            </Dialog.Description>
            <SidebarInner {...rest} onClose={() => onMobileOpenChange(false)} />
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </>
  );
}
