"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { MotionConfig } from "motion/react";

import { useAuth } from "../lib/hooks/useAuth";
import { useUpload, IngestionStatus } from "../lib/hooks/useUpload";
import { useDocuments, DocumentLibraryItem } from "../lib/hooks/useDocuments";
import { apiStreamChat } from "../lib/api";

import AppShell from "./components/layout/app-shell";
import AppSidebar from "./components/layout/app-sidebar";
import WorkspaceHeader from "./components/layout/workspace-header";
import AssistantPanel from "./components/chat/assistant-panel";
import CommandPalette from "./components/ui/command-palette";
import DocumentOverview from "./components/documents/document-overview";
import EmptyWorkspace from "./components/documents/empty-workspace";

import { Loader2 } from "lucide-react";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

type UploadType = "pdf" | "txt" | "text";

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
              isAsking={isAsking}
              hasAnswer={Boolean(streamingAnswer)}
              hasSources={sources.length > 0}
              onSendQuestion={handleSendQuestion}
            />
          ) : (
            <EmptyWorkspace
              uploadType={uploadType}
              setUploadType={setUploadType}
              file={file}
              setFile={setFile}
              plainText={plainText}
              setPlainText={setPlainText}
              ingestionStatus={ingestionStatus}
              uploadError={uploadError}
              onSubmit={handleUpload}
              recentDocuments={documents}
              onSelectDocument={handleSelectFromLibrary}
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
