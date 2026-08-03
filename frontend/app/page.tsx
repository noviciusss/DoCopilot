"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { 
  Library, 
  UploadCloud, 
  MessageSquare, 
  LogOut, 
  FileText, 
  Sparkles, 
  Send, 
  RotateCw, 
  FileCheck,
  Zap,
  Copy,
  Check,
  X,
  FileUp,
  Cpu,
  Layers
} from "lucide-react";

import { useAuth } from "../lib/hooks/useAuth";
import { useUpload, UPLOAD_STATUS_LABELS, IngestionStatus } from "../lib/hooks/useUpload";
import { useDocuments, DocumentLibraryItem } from "../lib/hooks/useDocuments";
import { apiStreamChat } from "../lib/api";
import DocumentLibrary from "./components/DocumentLibrary";
import Logo from "./components/Logo";

type UploadType = "pdf" | "txt" | "text";

const TABS: { id: UploadType; label: string; icon: any }[] = [
  { id: "pdf",  label: "PDF File", icon: FileText },
  { id: "txt",  label: "Plain Text", icon: FileUp },
  { id: "text", label: "Paste Text", icon: Sparkles },
];

const UPLOAD_BTN: Record<IngestionStatus, string> = {
  idle:      "Index Document",
  uploading: "Sending file...",
  queued:    "Queued in pipeline...",
  running:   "Generating vector embeddings...",
  succeeded: "Re-index Document",
  failed:    "Retry Indexing",
};

export default function Home() {
  const router = useRouter();
  const { isLoggedIn, isHydrated, logout } = useAuth();

  // Single upload pipeline state
  const { status: ingestionStatus, documentId, error: uploadError, upload, clearDocument } = useUpload();

  // Document Library (persisted docs from DB)
  const { documents, loading: libLoading, error: libError, refresh: refreshLib, deleteDoc, showAllDocs, toggleScope } = useDocuments(isLoggedIn);

  // The active document selected for chat (fresh upload OR selected from library)
  const [activeDocId, setActiveDocId]   = useState<string | null>(null);
  const [libOpen, setLibOpen]           = useState(true);

  const [uploadType, setUploadType]     = useState<UploadType>("pdf");
  const [file, setFile]                 = useState<File | null>(null);
  const [plainText, setPlainText]       = useState("");

  const [question, setQuestion]         = useState("");
  const [streamingAnswer, setStreamingAnswer] = useState("");
  const [sources, setSources]           = useState<string[]>([]);
  const [isAsking, setIsAsking]         = useState(false);
  const [copied, setCopied]             = useState(false);
  const cancelStreamRef                 = useRef<(() => void) | null>(null);
  const answerRef                       = useRef<HTMLDivElement>(null);

  // When upload completes → auto-select the new document & refresh library
  useEffect(() => {
    if (ingestionStatus === "succeeded" && documentId) {
      setActiveDocId(documentId);
      refreshLib();
    }
  }, [ingestionStatus, documentId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Redirect to login if not authenticated
  useEffect(() => {
    if (isHydrated && !isLoggedIn) router.push("/login");
  }, [isLoggedIn, isHydrated, router]);

  // Auto-scroll while streaming
  useEffect(() => {
    if (streamingAnswer && answerRef.current) {
      answerRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [streamingAnswer]);

  const askDisabled = useMemo(
    () => !activeDocId || !question.trim() || isAsking,
    [activeDocId, question, isAsking]
  );

  const handleUpload = async (evt: FormEvent<HTMLFormElement>) => {
    evt.preventDefault();
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

  const handleChat = async (evt: FormEvent<HTMLFormElement>) => {
    evt.preventDefault();
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
        setStreamingAnswer("Warning: " + err);
        setIsAsking(false);
      }
    );
  };

  const handleClearSession = () => {
    cancelStreamRef.current?.();
    clearDocument();
    setActiveDocId(null);
    setFile(null);
    setPlainText("");
    setStreamingAnswer("");
    setSources([]);
  };

  const handleCopyAnswer = () => {
    if (!streamingAnswer) return;
    navigator.clipboard.writeText(streamingAnswer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!isHydrated) {
    return (
      <div className="min-h-screen bg-zinc-950 flex items-center justify-center">
        <RotateCw className="w-6 h-6 text-indigo-400 animate-spin" />
      </div>
    );
  }

  const activeLibDoc = documents.find(d => d.id === activeDocId);
  const effectiveStatus: IngestionStatus =
    activeDocId === documentId ? ingestionStatus :
    activeLibDoc?.ingestion_status === "succeeded" ? "succeeded" : "idle";

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col selection:bg-indigo-500/30 selection:text-indigo-200">

      {/* ── Header / Navigation ──────────────────────────────────── */}
      <nav className="sticky top-0 z-20 border-b border-zinc-800/80 bg-zinc-950/80 backdrop-blur-xl">
        <div className="mx-auto flex max-w-screen-xl items-center justify-between px-6 py-3.5">
          <Logo size="md" />

          <div className="flex items-center gap-3">
            {/* Processing badge */}
            {(ingestionStatus === "running" || ingestionStatus === "queued") && (
              <span className="flex items-center gap-2 text-xs text-amber-400 bg-amber-400/10 border border-amber-400/20 px-3 py-1 rounded-full font-mono">
                <RotateCw className="w-3 h-3 animate-spin text-amber-400" />
                {ingestionStatus === "queued" ? "Queued" : "Chunking Vectors..."}
              </span>
            )}

            {/* Active document indicator */}
            {activeDocId ? (
              <div className="flex items-center gap-2.5">
                <span className="flex h-2 w-2 rounded-full bg-emerald-400 animate-pulse shadow-sm shadow-emerald-400/50" />
                <span className="hidden sm:inline-flex items-center gap-2 font-mono text-xs text-zinc-200 bg-zinc-900/90 border border-zinc-800 px-3 py-1 rounded-lg">
                  <FileText className="w-3.5 h-3.5 text-indigo-400" />
                  <span className="max-w-[180px] truncate">{activeLibDoc?.filename ?? `ID: ${activeDocId.slice(0, 8)}...`}</span>
                </span>
                <button
                  type="button"
                  onClick={handleClearSession}
                  className="text-xs text-zinc-400 hover:text-zinc-200 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 px-2.5 py-1 rounded-lg transition-colors"
                >
                  Unload
                </button>
              </div>
            ) : (
              <span className="text-xs text-zinc-500 flex items-center gap-2 bg-zinc-900/50 border border-zinc-800/50 px-3 py-1 rounded-lg">
                <span className="h-2 w-2 rounded-full bg-zinc-700" />
                No active document
              </span>
            )}

            {/* Library Toggle */}
            <button
              id="toggle-library"
              onClick={() => setLibOpen(o => !o)}
              className={`text-xs px-3 py-1.5 rounded-lg border transition-all flex items-center gap-2 font-medium ${
                libOpen
                  ? "border-indigo-500/40 text-indigo-300 bg-indigo-500/10 shadow-sm shadow-indigo-500/10"
                  : "border-zinc-800 text-zinc-400 hover:text-zinc-200 hover:border-zinc-700 bg-zinc-900/50"
              }`}
            >
              <Library className="w-3.5 h-3.5" />
              <span>Library</span>
            </button>

            {/* Sign out */}
            <button
              onClick={logout}
              className="text-xs text-zinc-400 hover:text-zinc-200 bg-zinc-900/50 hover:bg-zinc-800 border border-zinc-800 hover:border-zinc-700 px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5"
            >
              <LogOut className="w-3.5 h-3.5" />
              <span>Sign out</span>
            </button>
          </div>
        </div>
      </nav>

      {/* ── Main Layout ─────────────────────────────────────────────── */}
      <main className="flex-1 flex overflow-hidden">

        {/* Document Library Sidebar Drawer */}
        {libOpen && (
          <aside className="w-80 flex-shrink-0 border-r border-zinc-800/80 bg-zinc-950/60 h-[calc(100vh-65px)] sticky top-[65px] overflow-hidden flex flex-col">
            <DocumentLibrary
              documents={documents}
              loading={libLoading}
              error={libError}
              showAllDocs={showAllDocs}
              activeDocumentId={activeDocId}
              onSelect={handleSelectFromLibrary}
              onDelete={async (id) => { await deleteDoc(id); if (id === activeDocId) handleClearSession(); }}
              onToggleScope={toggleScope}
              onRefresh={refreshLib}
            />
          </aside>
        )}

        {/* Workspace & Chat Panel */}
        <div className="flex-1 overflow-y-auto bg-gradient-to-b from-zinc-950 via-zinc-950/95 to-zinc-900/20">
          <div className="mx-auto max-w-5xl px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">

            {/* Left Column — Upload Workspace */}
            <section className="lg:col-span-5 space-y-5">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <div className="p-1.5 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
                    <UploadCloud className="w-4 h-4" />
                  </div>
                  <h1 className="text-base font-semibold tracking-tight text-zinc-100">Document Ingestion</h1>
                </div>
                <p className="text-xs text-zinc-400">
                  Upload a PDF/TXT or select an indexed document from your library to chat.
                </p>
              </div>

              <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/40 backdrop-blur-xl p-5 space-y-5 shadow-xl hover:border-zinc-700/80 transition-all duration-300">
                {/* Format tabs */}
                <div className="flex gap-1 rounded-xl bg-zinc-950 p-1 border border-zinc-800/80">
                  {TABS.map((tab) => {
                    const IconComponent = tab.icon;
                    return (
                      <button
                        key={tab.id}
                        type="button"
                        onClick={() => { setUploadType(tab.id); setFile(null); }}
                        className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs font-medium transition-all duration-200 ${
                          uploadType === tab.id
                            ? "bg-gradient-to-r from-indigo-600/90 to-violet-600/90 text-white shadow-md shadow-indigo-500/20 border border-indigo-400/30"
                            : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900/50"
                        }`}
                      >
                        <IconComponent className="w-3.5 h-3.5" />
                        <span>{tab.label}</span>
                      </button>
                    );
                  })}
                </div>

                <form onSubmit={handleUpload} className="space-y-4">
                  {uploadType === "text" ? (
                    <div className="space-y-1.5">
                      <label className="text-xs font-medium text-zinc-300 flex items-center justify-between">
                        <span>Raw Content Snippet</span>
                        <span className="text-[10px] text-zinc-500">Plain text format</span>
                      </label>
                      <textarea
                        value={plainText}
                        onChange={(e) => setPlainText(e.target.value)}
                        rows={7}
                        placeholder="Paste plain text document content here..."
                        className="w-full rounded-xl border border-zinc-800 bg-zinc-950/70 px-4 py-3 text-xs text-zinc-100 placeholder:text-zinc-600 outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/50 transition-all resize-none shadow-inner"
                      />
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <label className="text-xs font-medium text-zinc-300">
                          {uploadType === "pdf" ? "PDF Document File" : "TXT Plain Text File"}
                        </label>
                        <span className="text-[10px] text-zinc-500">Max 20MB</span>
                      </div>
                      <div className="relative group">
                        <input
                          id="file-upload-input"
                          type="file"
                          accept={uploadType === "pdf" ? ".pdf,application/pdf" : ".txt,text/plain"}
                          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                          className="absolute inset-0 z-10 w-full h-full opacity-0 cursor-pointer"
                        />
                        <div className={`flex flex-col items-center justify-center rounded-xl border-2 border-dashed px-4 py-7 text-center transition-all duration-200 ${
                          file 
                            ? "border-indigo-500/50 bg-indigo-500/5" 
                            : "border-zinc-800 hover:border-indigo-500/40 bg-zinc-950/50 group-hover:bg-zinc-950/80"
                        }`}>
                          {file ? (
                            <div className="flex items-center gap-3 text-left w-full px-2">
                              <div className="p-2 rounded-lg bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                                <FileCheck className="w-5 h-5" />
                              </div>
                              <div className="flex-1 min-w-0">
                                <p className="text-xs font-medium text-zinc-200 truncate">{file.name}</p>
                                <p className="text-[10px] text-zinc-400">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                              </div>
                              <button
                                type="button"
                                onClick={(e) => { e.stopPropagation(); setFile(null); }}
                                className="p-1 text-zinc-500 hover:text-zinc-300 rounded hover:bg-zinc-800"
                              >
                                <X className="w-4 h-4" />
                              </button>
                            </div>
                          ) : (
                            <>
                              <UploadCloud className="w-8 h-8 text-zinc-600 mb-2 group-hover:text-indigo-400 group-hover:scale-110 transition-all duration-300" />
                              <p className="text-xs font-medium text-zinc-300">Click to choose or drag & drop</p>
                              <p className="text-[10px] text-zinc-500 mt-0.5">
                                {uploadType === "pdf" ? "PDF format up to 20MB" : "Plain text .txt files"}
                              </p>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                  )}

                  {uploadError && (
                    <div className="rounded-xl bg-rose-950/40 border border-rose-900/60 p-3 text-xs text-rose-300 flex items-start gap-2">
                      <span className="font-semibold text-rose-400">Error:</span>
                      <span className="flex-1">{uploadError}</span>
                    </div>
                  )}

                  <button
                    id="submit-ingestion-button"
                    type="submit"
                    disabled={
                      (uploadType !== "text" && !file) ||
                      (uploadType === "text" && !plainText.trim()) ||
                      ingestionStatus === "uploading" ||
                      ingestionStatus === "running"
                    }
                    className="w-full rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white font-medium py-2.5 text-xs transition-all shadow-lg shadow-indigo-600/20 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2 border border-indigo-400/20"
                  >
                    {ingestionStatus === "uploading" || ingestionStatus === "running" ? (
                      <>
                        <RotateCw className="w-3.5 h-3.5 animate-spin" />
                        <span>{UPLOAD_BTN[ingestionStatus]}</span>
                      </>
                    ) : (
                      <>
                        <Zap className="w-3.5 h-3.5 text-indigo-200" />
                        <span>{UPLOAD_BTN[ingestionStatus]}</span>
                      </>
                    )}
                  </button>
                </form>
              </div>

              {/* Status / Active Document Card */}
              {activeDocId && (
                <div className="rounded-2xl border border-indigo-500/30 bg-indigo-500/5 p-4 space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-indigo-300 font-medium flex items-center gap-1.5">
                      <Layers className="w-3.5 h-3.5 text-indigo-400" />
                      Active Search Target
                    </span>
                    <span className="text-[10px] font-mono text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded border border-emerald-400/20">
                      Ready for RAG
                    </span>
                  </div>
                  <p className="text-xs font-mono text-zinc-300 truncate">
                    {activeLibDoc?.filename ?? activeDocId}
                  </p>
                </div>
              )}
            </section>

            {/* Right Column — Copilot Chat Interface */}
            <section className="lg:col-span-7 flex flex-col space-y-5">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <div className="p-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
                    <MessageSquare className="w-4 h-4" />
                  </div>
                  <h2 className="text-base font-semibold tracking-tight text-zinc-100">Hybrid RAG Copilot</h2>
                </div>
                <p className="text-xs text-zinc-400">
                  Ask natural language questions to perform Qdrant hybrid retrieval + Cohere reranking + LLM synthesis.
                </p>
              </div>

              {/* Chat Container */}
              <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/40 backdrop-blur-xl p-5 flex flex-col min-h-[460px] shadow-2xl">
                
                {/* Response area */}
                <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-1">
                  {!streamingAnswer && !isAsking && (
                    <div className="flex flex-col items-center justify-center h-64 text-center text-zinc-500 space-y-3">
                      <div className="p-4 rounded-2xl bg-zinc-950 border border-zinc-800/80 shadow-inner">
                        <Cpu className="w-8 h-8 text-indigo-400 opacity-80" />
                      </div>
                      <div className="space-y-1 max-w-xs">
                        <p className="text-xs font-medium text-zinc-300">Ready for Document QA</p>
                        <p className="text-[11px] text-zinc-500">
                          {activeDocId 
                            ? "Type a question below to query the active document."
                            : "Upload a document or select one from your library to start chatting."}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Streaming Output */}
                  {(streamingAnswer || isAsking) && (
                    <div className="space-y-4">
                      <div className="rounded-xl border border-zinc-800 bg-zinc-950/80 p-4 space-y-3 shadow-inner">
                        <div className="flex items-center justify-between border-b border-zinc-900 pb-2.5">
                          <span className="text-xs font-medium text-indigo-400 flex items-center gap-1.5">
                            <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                            Copilot Response
                          </span>
                          {streamingAnswer && (
                            <button
                              type="button"
                              onClick={handleCopyAnswer}
                              className="text-[10px] text-zinc-400 hover:text-zinc-200 flex items-center gap-1 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 px-2 py-0.5 rounded transition-colors"
                            >
                              {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                              <span>{copied ? "Copied" : "Copy"}</span>
                            </button>
                          )}
                        </div>

                        {streamingAnswer ? (
                          <div className="prose prose-invert prose-xs max-w-none text-zinc-200 leading-relaxed space-y-2">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                              {streamingAnswer}
                            </ReactMarkdown>
                            <div ref={answerRef} />
                          </div>
                        ) : (
                          <div className="flex items-center gap-2 py-4 text-xs text-zinc-500 font-mono">
                            <RotateCw className="w-3.5 h-3.5 text-indigo-400 animate-spin" />
                            <span>Retrieving vector chunks and generating answer...</span>
                          </div>
                        )}
                      </div>

                      {/* Source Citations */}
                      {sources.length > 0 && (
                        <div className="rounded-xl border border-zinc-800/80 bg-zinc-950/40 p-3.5 space-y-2">
                          <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider block">
                            Retrieved Sources ({sources.length})
                          </span>
                          <div className="flex flex-wrap gap-2">
                            {sources.map((src, idx) => (
                              <span
                                key={idx}
                                className="text-[10px] font-mono text-zinc-300 bg-zinc-900 border border-zinc-800 px-2.5 py-1 rounded-md"
                              >
                                {src}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Input form */}
                <form onSubmit={handleChat} className="mt-auto space-y-2">
                  <div className="relative flex items-center">
                    <input
                      id="chat-question-input"
                      type="text"
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      disabled={!activeDocId || isAsking}
                      placeholder={
                        activeDocId
                          ? "Ask a question about the active document..."
                          : "Select or upload a document to enable chat..."
                      }
                      className="w-full rounded-xl border border-zinc-800 bg-zinc-950 px-4 py-3 pr-12 text-xs text-zinc-100 placeholder:text-zinc-600 outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-inner"
                    />
                    <button
                      id="send-chat-button"
                      type="submit"
                      disabled={askDisabled}
                      className="absolute right-2 p-2 rounded-lg bg-gradient-to-r from-indigo-600 to-violet-600 text-white hover:from-indigo-500 hover:to-violet-500 disabled:opacity-30 disabled:cursor-not-allowed transition-all shadow-md shadow-indigo-600/20"
                    >
                      {isAsking ? (
                        <RotateCw className="w-3.5 h-3.5 animate-spin" />
                      ) : (
                        <Send className="w-3.5 h-3.5" />
                      )}
                    </button>
                  </div>
                </form>

              </div>
            </section>

          </div>
        </div>

      </main>
    </div>
  );
}
