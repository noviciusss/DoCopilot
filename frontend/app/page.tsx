"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { MotionConfig, motion } from "motion/react";

import { useAuth } from "../lib/hooks/useAuth";
import { useUpload, IngestionStatus } from "../lib/hooks/useUpload";
import { useDocuments, DocumentLibraryItem } from "../lib/hooks/useDocuments";
import { apiStreamChat } from "../lib/api";

import AppShell from "./components/layout/app-shell";
import AppSidebar from "./components/layout/app-sidebar";
import WorkspaceHeader from "./components/layout/workspace-header";
import UploadDropzone from "./components/documents/upload-dropzone";
import UploadStatus from "./components/documents/upload-status";
import AssistantPanel from "./components/chat/assistant-panel";
import CommandPalette from "./components/ui/command-palette";

import {
  FileText,
  Info,
  AlertTriangle,
  Loader2,
  CheckCircle2,
} from "lucide-react";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

type UploadType = "pdf" | "txt" | "text";

// ─────────────────────────────────────────────────────────────────────────────
// Document Overview center pane
// ─────────────────────────────────────────────────────────────────────────────

interface DocOverviewProps {
  doc: DocumentLibraryItem;
  effectiveStatus: IngestionStatus;
  onSendQuestion: (q: string) => void;
}

const SHORTCUT_QUESTIONS = [
  "Summarize the main points of this document",
  "What are the key conclusions or findings?",
  "What topics or concepts are defined here?",
];

function DocumentOverview({ doc, effectiveStatus, onSendQuestion }: DocOverviewProps) {
  function formatBytes(b: number): string {
    if (b < 1024) return `${b} B`;
    if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
    return `${(b / 1024 / 1024).toFixed(1)} MB`;
  }

  const isReady = effectiveStatus === "succeeded";
  const isProcessing = ["uploading", "queued", "running"].includes(effectiveStatus);
  const isFailed = effectiveStatus === "failed";

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
      className="flex flex-col gap-4 max-w-xl mx-auto w-full px-4 py-6"
    >
      {/* Document identity card */}
      <div
        className="rounded-xl p-5 space-y-3"
        style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
      >
        <div className="flex items-start gap-3">
          <div
            className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
            style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
          >
            <FileText size={18} style={{ color: "var(--text-2)" }} aria-hidden="true" />
          </div>
          <div className="flex-1 min-w-0">
            <h2
              className="text-sm font-semibold leading-snug break-words"
              style={{ color: "var(--text-1)", wordBreak: "break-word" }}
            >
              {doc.filename}
            </h2>
            <p className="text-meta mt-0.5">
              {formatBytes(doc.file_size_bytes)} · {doc.mime_type} ·{" "}
              {new Date(doc.created_at).toLocaleDateString("en-US", {
                month: "short",
                day: "numeric",
                year: "numeric",
              })}
            </p>
          </div>
        </div>

        {/* Status */}
        <div
          className="flex items-center gap-2 p-2.5 rounded-md"
          style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
          role="status"
          aria-live="polite"
        >
          {isReady && (
            <>
              <CheckCircle2 size={13} style={{ color: "var(--green)", flexShrink: 0 }} aria-hidden="true" />
              <p className="text-xs" style={{ color: "var(--text-2)" }}>
                <span style={{ color: "var(--green)", fontWeight: 500 }}>Ready to query.</span>{" "}
                Ask a question in the assistant panel on the right.
              </p>
            </>
          )}
          {isProcessing && (
            <>
              <Loader2 size={13} className="animate-spin flex-shrink-0" style={{ color: "var(--cobalt)" }} aria-hidden="true" />
              <p className="text-xs" style={{ color: "var(--text-2)" }}>
                Document is being processed and indexed — this may take a moment.
              </p>
            </>
          )}
          {isFailed && (
            <>
              <AlertTriangle size={13} style={{ color: "var(--red)", flexShrink: 0 }} aria-hidden="true" />
              <p className="text-xs" style={{ color: "var(--red)" }}>
                Indexing failed. Try re-uploading this document.
              </p>
            </>
          )}
        </div>
      </div>

      {/* How answers work */}
      <div
        className="flex items-start gap-2.5 px-3.5 py-3 rounded-lg"
        style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
      >
        <Info size={13} style={{ color: "var(--text-3)", flexShrink: 0, marginTop: 1 }} aria-hidden="true" />
        <p className="text-xs leading-relaxed" style={{ color: "var(--text-2)" }}>
          Answers are generated by retrieving the most relevant passages from this document
          and synthesizing them with an LLM. The{" "}
          <span style={{ color: "var(--amber)", fontWeight: 500 }}>source snippets</span>{" "}
          shown with each answer are the actual retrieved context used to produce the response.
        </p>
      </div>

      {/* Shortcut questions — only if indexed */}
      {isReady && (
        <div className="space-y-2">
          <p className="text-label" style={{ color: "var(--text-3)" }}>Quick questions</p>
          <div className="space-y-1.5">
            {SHORTCUT_QUESTIONS.map((q) => (
              <button
                key={q}
                type="button"
                onClick={() => onSendQuestion(q)}
                className="w-full text-left text-xs rounded-md px-3 py-2.5 leading-snug transition-colors"
                style={{
                  background: "var(--surface)",
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
                {q}
              </button>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Upload workspace (empty / no active doc)
// ─────────────────────────────────────────────────────────────────────────────

interface UploadWorkspaceProps {
  uploadType: UploadType;
  setUploadType: (t: UploadType) => void;
  file: File | null;
  setFile: (f: File | null) => void;
  plainText: string;
  setPlainText: (t: string) => void;
  ingestionStatus: IngestionStatus;
  uploadError: string | null;
  onSubmit: (e: FormEvent<HTMLFormElement>) => void;
}

function UploadWorkspace(props: UploadWorkspaceProps) {
  return (
    <div className="flex flex-col items-center justify-start px-4 py-8 max-w-lg mx-auto w-full gap-5">
      {/* Minimal stacked-paper illustration — CSS only */}
      <div className="relative w-14 h-16 mb-2 select-none" aria-hidden="true">
        <div
          className="absolute bottom-0 left-1 w-12 h-14 rounded-lg"
          style={{ background: "var(--paper-border)", transform: "rotate(-4deg)" }}
        />
        <div
          className="absolute bottom-0 left-0.5 w-12 h-14 rounded-lg"
          style={{ background: "var(--paper-2)", transform: "rotate(-1.5deg)" }}
        />
        <div
          className="absolute bottom-0 left-0 w-12 h-14 rounded-lg"
          style={{ background: "var(--paper)", border: "1px solid var(--paper-border)" }}
        />
        {/* Fold corner */}
        <div
          className="absolute top-0 right-0 w-0 h-0"
          style={{
            borderLeft: "10px solid transparent",
            borderBottom: "10px solid transparent",
            borderRight: "10px solid var(--paper-border)",
            borderTop: "10px solid var(--paper-border)",
            borderRadius: "0 2px 0 0",
          }}
        />
        {/* Lines */}
        <div className="absolute top-5 left-2 right-2 space-y-1.5">
          <div className="h-0.5 rounded-full" style={{ background: "var(--paper-border)" }} />
          <div className="h-0.5 rounded-full w-3/4" style={{ background: "var(--paper-border)" }} />
          <div className="h-0.5 rounded-full w-2/3" style={{ background: "var(--paper-border)" }} />
        </div>
      </div>

      <div className="text-center space-y-0.5">
        <h1 className="text-sm font-semibold" style={{ color: "var(--text-1)" }}>
          Upload a document
        </h1>
        <p className="text-xs" style={{ color: "var(--text-2)" }}>
          PDF, TXT, or paste text · Max 20 MB · Answers are grounded in retrieved passages
        </p>
      </div>

      <div
        className="w-full rounded-xl p-5 space-y-4"
        style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
      >
        <UploadDropzone {...props} />
        <UploadStatus status={props.ingestionStatus} error={props.uploadError} />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Root page component
// ─────────────────────────────────────────────────────────────────────────────

export default function Home() {
  const router = useRouter();
  const { isLoggedIn, isHydrated, logout } = useAuth();

  // Upload pipeline — hook unchanged
  const {
    status: ingestionStatus,
    documentId,
    error: uploadError,
    upload,
    clearDocument,
  } = useUpload();

  // Document library — hook unchanged
  const {
    documents,
    loading: libLoading,
    error: libError,
    refresh: refreshLib,
    deleteDoc,
    showAllDocs,
    toggleScope,
  } = useDocuments(isLoggedIn);

  // Active document selected for chat
  const [activeDocId, setActiveDocId] = useState<string | null>(null);

  // Upload form state
  const [uploadType, setUploadType] = useState<UploadType>("pdf");
  const [file, setFile] = useState<File | null>(null);
  const [plainText, setPlainText] = useState("");

  // Chat state
  const [question, setQuestion] = useState("");
  const [streamingAnswer, setStreamingAnswer] = useState("");
  const [sources, setSources] = useState<string[]>([]);
  const [isAsking, setIsAsking] = useState(false);
  const [copied, setCopied] = useState(false);
  const cancelStreamRef = useRef<(() => void) | null>(null);

  // UI state
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [mobileChatOpen, setMobileChatOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);

  // ── Side effects ────────────────────────────────────────────────────────────

  // Upload completes → auto-select new document & refresh library
  useEffect(() => {
    if (ingestionStatus === "succeeded" && documentId) {
      setActiveDocId(documentId); // eslint-disable-line react-hooks/set-state-in-effect
      refreshLib();
    }
  }, [ingestionStatus, documentId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Redirect to login if not authenticated
  useEffect(() => {
    if (isHydrated && !isLoggedIn) router.push("/login");
  }, [isLoggedIn, isHydrated, router]);

  // Global Cmd/Ctrl+K → command palette
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setPaletteOpen(true);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  // ── Derived state ──────────────────────────────────────────────────────────

  const activeLibDoc = documents.find((d) => d.id === activeDocId);

  const effectiveStatus: IngestionStatus =
    activeDocId === documentId
      ? ingestionStatus
      : activeLibDoc?.ingestion_status === "succeeded"
      ? "succeeded"
      : "idle";

  const askDisabled = useMemo(
    () => !activeDocId || !question.trim() || isAsking,
    [activeDocId, question, isAsking]
  );

  // ── Handlers ───────────────────────────────────────────────────────────────

  const handleUpload = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (uploadType !== "text" && !file) return;
    if (uploadType === "text" && !plainText.trim()) return;
    await upload(file, plainText, uploadType);
  };

  const handleSelectFromLibrary = useCallback((doc: DocumentLibraryItem) => {
    setActiveDocId(doc.id);
    setSources([]);
    setStreamingAnswer("");
    setQuestion("");
  }, []);

  const handlePaletteSelect = useCallback((doc: DocumentLibraryItem) => {
    handleSelectFromLibrary(doc);
  }, [handleSelectFromLibrary]);

  const handleChat = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (askDisabled || !activeDocId) return;

    setIsAsking(true);
    setStreamingAnswer("");
    setSources([]);

    cancelStreamRef.current = apiStreamChat(
      question,
      activeDocId,
      (token) => setStreamingAnswer((prev) => prev + token),
      (srcs, fullAns) => {
        setSources(srcs);
        if (fullAns) setStreamingAnswer(fullAns);
        setIsAsking(false);
      },
      (err) => {
        setStreamingAnswer("Error: " + err);
        setIsAsking(false);
      }
    );
  };

  // Send a suggested/shortcut question directly
  const handleSendQuestion = useCallback((q: string) => {
    setQuestion(q);
    // Defer so question state is set before submit
    setTimeout(() => {
      if (!activeDocId) return;
      setIsAsking(true);
      setStreamingAnswer("");
      setSources([]);

      cancelStreamRef.current = apiStreamChat(
        q,
        activeDocId,
        (token) => setStreamingAnswer((prev) => prev + token),
        (srcs, fullAns) => {
          setSources(srcs);
          if (fullAns) setStreamingAnswer(fullAns);
          setIsAsking(false);
        },
        (err) => {
          setStreamingAnswer("Error: " + err);
          setIsAsking(false);
        }
      );
    }, 0);
  }, [activeDocId]);

  const handleClearSession = useCallback(() => {
    cancelStreamRef.current?.();
    clearDocument();
    setActiveDocId(null);
    setFile(null);
    setPlainText("");
    setStreamingAnswer("");
    setSources([]);
    setQuestion("");
  }, [clearDocument]);

  const handleClearConversation = useCallback(() => {
    cancelStreamRef.current?.();
    setStreamingAnswer("");
    setSources([]);
    setQuestion("");
    setIsAsking(false);
  }, []);

  const handleCopyAnswer = useCallback(() => {
    if (!streamingAnswer) return;
    navigator.clipboard.writeText(streamingAnswer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [streamingAnswer]);

  const handleDelete = useCallback(async (id: string) => {
    await deleteDoc(id);
    if (id === activeDocId) handleClearSession();
  }, [deleteDoc, activeDocId, handleClearSession]);

  // ── Loading gate ───────────────────────────────────────────────────────────

  if (!isHydrated) {
    return (
      <div
        className="min-h-screen flex items-center justify-center"
        style={{ background: "var(--ink)" }}
      >
        <Loader2 size={20} className="animate-spin" style={{ color: "var(--text-3)" }} aria-label="Loading" />
      </div>
    );
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  const chatPanel = (
    <AssistantPanel
      activeDocName={activeLibDoc?.filename ?? null}
      question={question}
      onQuestionChange={setQuestion}
      streamingAnswer={streamingAnswer}
      sources={sources}
      isAsking={isAsking}
      hasDocument={!!activeDocId && effectiveStatus === "succeeded"}
      copied={copied}
      onSubmit={handleChat}
      onCopy={handleCopyAnswer}
      onClear={handleClearConversation}
      onSuggestedPrompt={handleSendQuestion}
    />
  );

  const workspace = (
    <>
      <WorkspaceHeader
        activeDoc={activeLibDoc}
        effectiveStatus={effectiveStatus}
        onClearSession={handleClearSession}
        onOpenMobileSidebar={() => setMobileSidebarOpen(true)}
        onOpenMobileChat={() => setMobileChatOpen(true)}
      />

      <MotionConfig reducedMotion="user">
        <div className="flex-1 overflow-y-auto">
          {activeDocId && activeLibDoc ? (
            <DocumentOverview
              doc={activeLibDoc}
              effectiveStatus={effectiveStatus}
              onSendQuestion={handleSendQuestion}
            />
          ) : (
            <UploadWorkspace
              uploadType={uploadType}
              setUploadType={setUploadType}
              file={file}
              setFile={setFile}
              plainText={plainText}
              setPlainText={setPlainText}
              ingestionStatus={ingestionStatus}
              uploadError={uploadError}
              onSubmit={handleUpload}
            />
          )}
        </div>
      </MotionConfig>
    </>
  );

  return (
    <MotionConfig reducedMotion="user">
      <AppShell
        sidebar={
          <AppSidebar
            documents={documents}
            loading={libLoading}
            error={libError}
            showAllDocs={showAllDocs}
            activeDocumentId={activeDocId}
            onSelect={handleSelectFromLibrary}
            onDelete={handleDelete}
            onToggleScope={toggleScope}
            onRefresh={refreshLib}
            onNewDocument={handleClearSession}
            onOpenSearch={() => setPaletteOpen(true)}
            onLogout={logout}
            collapsed={sidebarCollapsed}
            onToggleCollapse={() => setSidebarCollapsed((prev) => !prev)}
            mobileOpen={mobileSidebarOpen}
            onMobileOpenChange={setMobileSidebarOpen}
          />
        }
        workspace={workspace}
        chatPanel={chatPanel}
        sidebarCollapsed={sidebarCollapsed}
        mobileChatOpen={mobileChatOpen}
        onMobileChatOpenChange={setMobileChatOpen}
      />

      {/* Global command palette */}
      <CommandPalette
        open={paletteOpen}
        onOpenChange={setPaletteOpen}
        documents={documents}
        onSelectDocument={handlePaletteSelect}
      />
    </MotionConfig>
  );
}
