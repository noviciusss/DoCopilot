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
  AlertTriangle,
  FileCheck,
  Zap
} from "lucide-react";

import { useAuth } from "../lib/hooks/useAuth";
import { useUpload, UPLOAD_STATUS_LABELS, IngestionStatus } from "../lib/hooks/useUpload";
import { useDocuments, DocumentLibraryItem } from "../lib/hooks/useDocuments";
import { apiStreamChat } from "../lib/api";
import DocumentLibrary from "./components/DocumentLibrary";

type UploadType = "pdf" | "txt" | "text";

const TABS: { id: UploadType; label: string }[] = [
  { id: "pdf",  label: "PDF Document" },
  { id: "txt",  label: "Plain Text File" },
  { id: "text", label: "Paste Text" },
];

const UPLOAD_BTN: Record<IngestionStatus, string> = {
  idle:      "Index Source",
  uploading: "Sending file...",
  queued:    "Queued in pipeline...",
  running:   "Embedding vector chunks...",
  succeeded: "Re-index Source",
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

  if (!isHydrated) {
    return (
      <div className="min-h-screen bg-zinc-950 flex items-center justify-center">
        <RotateCw className="w-6 h-6 text-zinc-500 animate-spin" />
      </div>
    );
  }

  const activeLibDoc = documents.find(d => d.id === activeDocId);
  const effectiveStatus: IngestionStatus =
    activeDocId === documentId ? ingestionStatus :
    activeLibDoc?.ingestion_status === "succeeded" ? "succeeded" : "idle";

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col selection:bg-zinc-800 selection:text-white">

      {/* ── Header / Navigation ──────────────────────────────────── */}
      <nav className="sticky top-0 z-20 border-b border-zinc-900 bg-zinc-950/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-screen-xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-zinc-100 text-sm font-bold text-zinc-950 select-none shadow-sm shadow-white/10">
              D
            </span>
            <div className="flex flex-col">
              <span className="text-sm font-semibold tracking-tight text-zinc-100 leading-none">DoCopilot</span>
              <span className="text-[10px] text-zinc-500 tracking-wider">Enterprise RAG</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Processing badge */}
            {(ingestionStatus === "running" || ingestionStatus === "queued") && (
              <span className="flex items-center gap-1.5 text-xs text-amber-400 font-mono">
                <RotateCw className="w-3 h-3 animate-spin" />
                {ingestionStatus === "queued" ? "Queued" : "Indexing..."}
              </span>
            )}

            {/* Active document indicator */}
            {activeDocId ? (
              <div className="flex items-center gap-2.5">
                <span className="flex h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="hidden sm:inline-flex items-center gap-1.5 font-mono text-xs text-zinc-300 bg-zinc-900 border border-zinc-800 px-2.5 py-1 rounded-md">
                  <FileText className="w-3 h-3 text-violet-400" />
                  {activeLibDoc?.filename ?? `ID: ${activeDocId.slice(0, 8)}...`}
                </span>
                <button
                  type="button"
                  onClick={handleClearSession}
                  className="text-xs text-zinc-500 hover:text-zinc-300 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 px-2 py-1 rounded-md transition-colors"
                >
                  Unload
                </button>
              </div>
            ) : (
              <span className="text-xs text-zinc-500 flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-zinc-700" />
                No active document
              </span>
            )}

            {/* Library Toggle */}
            <button
              id="toggle-library"
              onClick={() => setLibOpen(o => !o)}
              className={`text-xs px-3 py-1.5 rounded-lg border transition-all flex items-center gap-1.5 font-medium ${
                libOpen
                  ? "border-violet-500/40 text-violet-300 bg-violet-500/10"
                  : "border-zinc-800 text-zinc-400 hover:text-zinc-200 hover:border-zinc-700"
              }`}
            >
              <Library className="w-3.5 h-3.5" />
              <span>Library</span>
            </button>

            {/* Sign out */}
            <button
              onClick={logout}
              className="text-xs text-zinc-500 hover:text-zinc-300 border border-zinc-800 hover:border-zinc-700 px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5"
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
          <aside className="w-80 flex-shrink-0 border-r border-zinc-900 bg-zinc-950/60 h-[calc(100vh-65px)] sticky top-[65px] overflow-hidden flex flex-col">
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
        <div className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-5xl px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">

            {/* Left Column — Upload Workspace */}
            <section className="lg:col-span-5 space-y-5">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <UploadCloud className="w-4 h-4 text-violet-400" />
                  <h1 className="text-lg font-medium tracking-tight text-zinc-200">Document Ingestion</h1>
                </div>
                <p className="text-xs text-zinc-500">
                  Upload a PDF/TXT document or select an indexed document from your library.
                </p>
              </div>

              <div className="rounded-2xl border border-zinc-900 bg-zinc-900/40 backdrop-blur-sm p-5 space-y-5 transition-all duration-300 hover:border-zinc-800">
                {/* Format tabs */}
                <div className="flex gap-1 rounded-xl bg-zinc-950 p-1 border border-zinc-900">
                  {TABS.map((tab) => (
                    <button
                      key={tab.id}
                      type="button"
                      onClick={() => { setUploadType(tab.id); setFile(null); }}
                      className={`flex-1 text-center py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${
                        uploadType === tab.id
                          ? "bg-zinc-900 text-zinc-100 shadow-sm border border-zinc-800/80"
                          : "text-zinc-500 hover:text-zinc-300"
                      }`}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>

                <form onSubmit={handleUpload} className="space-y-4">
                  {uploadType === "text" ? (
                    <div className="space-y-1.5">
                      <label className="text-xs font-medium text-zinc-400">Raw Content Text</label>
                      <textarea
                        value={plainText}
                        onChange={(e) => setPlainText(e.target.value)}
                        rows={7}
                        placeholder="Paste plain text content here..."
                        className="w-full rounded-xl border border-zinc-800/80 bg-zinc-950/60 px-3.5 py-2.5 text-xs text-zinc-100 placeholder:text-zinc-600 outline-none focus:border-zinc-700 focus:bg-zinc-950 transition-all resize-none shadow-inner"
                      />
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <label className="text-xs font-medium text-zinc-400">
                          {uploadType === "pdf" ? "PDF Document File" : "TXT Plain Text File"}
                        </label>
                        <span className="text-[10px] text-zinc-600 font-mono">Max 20 MB</span>
                      </div>
                      <div className="border border-dashed border-zinc-800 hover:border-zinc-700 transition-colors rounded-xl bg-zinc-950/40 p-5 flex flex-col items-center justify-center text-center cursor-pointer group relative">
                        <input
                          type="file"
                          accept={uploadType === "pdf" ? "application/pdf" : "text/plain,.txt"}
                          onChange={(e) => { setFile(e.target.files?.[0] ?? null); }}
                          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                        />
                        <UploadCloud className="w-7 h-7 text-zinc-600 group-hover:text-zinc-400 transition-colors mb-2" />
                        <span className="text-xs text-zinc-400 group-hover:text-zinc-200 transition-colors font-medium">
                          {file ? file.name : `Select or drag ${uploadType.toUpperCase()} file`}
                        </span>
                        {file && (
                          <span className="text-[10px] text-zinc-500 mt-1 font-mono">
                            {(file.size / 1024 / 1024).toFixed(2)} MB
                          </span>
                        )}
                      </div>
                    </div>
                  )}

                  <div className="space-y-2 pt-1">
                    <button
                      type="submit"
                      disabled={ingestionStatus === "uploading" || ingestionStatus === "queued" || ingestionStatus === "running"}
                      className="w-full rounded-xl bg-zinc-100 hover:bg-white text-zinc-950 font-semibold py-2.5 text-xs transition-colors shadow-sm disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                      {(ingestionStatus === "uploading" || ingestionStatus === "queued" || ingestionStatus === "running") ? (
                        <>
                          <RotateCw className="w-3.5 h-3.5 animate-spin text-zinc-950" />
                          <span>{UPLOAD_BTN[ingestionStatus]}</span>
                        </>
                      ) : (
                        <>
                          <Zap className="w-3.5 h-3.5 text-zinc-950" />
                          <span>{UPLOAD_BTN[ingestionStatus]}</span>
                        </>
                      )}
                    </button>

                    {ingestionStatus !== "idle" && (
                      <div className={`p-3 rounded-xl border text-xs flex items-start gap-2 ${
                        ingestionStatus === "succeeded"
                          ? "bg-emerald-950/20 border-emerald-900/60 text-emerald-350"
                          : ingestionStatus === "failed"
                          ? "bg-rose-950/20 border-rose-900/60 text-rose-350"
                          : "bg-amber-950/20 border-amber-900/60 text-amber-300"
                      }`}>
                        {ingestionStatus === "succeeded" && <FileCheck className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />}
                        {ingestionStatus === "failed" && <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />}
                        {(ingestionStatus === "running" || ingestionStatus === "queued" || ingestionStatus === "uploading") && (
                          <RotateCw className="w-4 h-4 text-amber-400 animate-spin shrink-0 mt-0.5" />
                        )}
                        <span className="flex-1 leading-snug">
                          {uploadError ?? UPLOAD_STATUS_LABELS[ingestionStatus]}
                        </span>
                      </div>
                    )}
                  </div>
                </form>
              </div>
            </section>

            {/* Right Column — Copilot Chat */}
            <section className="lg:col-span-7 flex flex-col space-y-5">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <MessageSquare className="w-4 h-4 text-violet-400" />
                  <h1 className="text-lg font-medium tracking-tight text-zinc-200">Copilot Chat</h1>
                </div>
                <p className="text-xs text-zinc-500">
                  Ground-truth Q&A backed by Qdrant hybrid vector search, BM25, and Cohere reranking.
                </p>
              </div>

              <div className="rounded-2xl border border-zinc-900 bg-zinc-900/40 backdrop-blur-sm p-5 space-y-4 transition-all duration-300 hover:border-zinc-800">
                <form onSubmit={handleChat} className="space-y-3">
                  <div className="relative">
                    <textarea
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      rows={3}
                      placeholder={
                        effectiveStatus === "succeeded"
                          ? "Ask any question about your document..."
                          : "Select or index a document from your library to unlock chat"
                      }
                      disabled={effectiveStatus !== "succeeded"}
                      className="w-full rounded-xl border border-zinc-800 bg-zinc-950/70 px-3.5 py-2.5 text-xs text-zinc-100 placeholder:text-zinc-600 outline-none focus:border-zinc-700 transition-colors resize-none disabled:opacity-40 disabled:cursor-not-allowed shadow-inner"
                    />
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-[10px] text-zinc-600 font-mono">Rate Limit: 20 / min</span>
                    <button
                      type="submit"
                      disabled={askDisabled}
                      className="rounded-xl bg-zinc-100 hover:bg-white text-zinc-950 font-semibold px-4 py-2 text-xs transition-colors shadow-sm disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5"
                    >
                      {isAsking ? (
                        <>
                          <RotateCw className="w-3.5 h-3.5 animate-spin text-zinc-950" />
                          <span>Synthesizing...</span>
                        </>
                      ) : (
                        <>
                          <Send className="w-3.5 h-3.5 text-zinc-950" />
                          <span>Submit Query</span>
                        </>
                      )}
                    </button>
                  </div>
                </form>
              </div>

              {/* Answer Output Stream */}
              {(streamingAnswer || isAsking) && (
                <div ref={answerRef} className="rounded-2xl border border-zinc-900 bg-zinc-900/40 backdrop-blur-sm p-5 space-y-4 transition-all duration-300 hover:border-zinc-800">
                  <div className="flex items-center justify-between border-b border-zinc-900 pb-3">
                    <div className="flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-violet-400" />
                      <h2 className="text-xs font-semibold uppercase tracking-widest text-zinc-400">
                        Synthesized Answer
                      </h2>
                    </div>
                    <span className="text-[10px] font-mono text-zinc-500 bg-zinc-950 px-2 py-0.5 rounded border border-zinc-900 flex items-center gap-1">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" /> SSE Streamed
                    </span>
                  </div>

                  <div className="space-y-4">
                    <article className="
                      prose prose-sm prose-invert max-w-none
                      prose-p:text-zinc-300 prose-p:leading-relaxed prose-p:my-2 prose-p:text-[13px]
                      prose-headings:text-zinc-100 prose-headings:font-medium prose-headings:tracking-tight
                      prose-strong:text-zinc-200 prose-strong:font-semibold
                      prose-em:text-zinc-400
                      prose-code:text-zinc-200 prose-code:bg-zinc-950 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-[11px] prose-code:before:content-none prose-code:after:content-none prose-code:border prose-code:border-zinc-900
                      prose-pre:bg-zinc-950 prose-pre:border prose-pre:border-zinc-900 prose-pre:rounded-xl prose-pre:p-4
                      prose-ul:text-zinc-300 prose-ol:text-zinc-300 prose-ul:my-2 prose-ol:my-2 prose-ul:list-disc
                      prose-li:my-0.5 prose-li:marker:text-zinc-700 prose-li:text-[13px]
                      prose-blockquote:border-zinc-700 prose-blockquote:text-zinc-400 prose-blockquote:pl-4 prose-blockquote:italic
                      prose-hr:border-zinc-900
                      prose-table:text-xs prose-table:my-3
                      prose-th:text-zinc-250 prose-th:font-semibold prose-th:pb-2 prose-th:border-b prose-th:border-zinc-800
                      prose-td:text-zinc-350 prose-td:py-2 prose-td:border-b prose-td:border-zinc-900
                    ">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {streamingAnswer}
                      </ReactMarkdown>
                      {isAsking && (
                        <span className="inline-block w-1.5 h-3.5 bg-zinc-400 animate-pulse ml-0.5 align-text-bottom" />
                      )}
                    </article>

                    {sources.length > 0 && (
                      <div className="border-t border-zinc-900 pt-3 space-y-2">
                        <p className="text-[10px] font-semibold uppercase tracking-widest text-zinc-500 font-mono">
                          Referenced Sources
                        </p>
                        <div className="flex flex-wrap gap-1.5">
                          {sources.map((src, i) => (
                            <span
                              key={`${src}-${i}`}
                              className="inline-flex items-center gap-1 rounded-lg border border-zinc-800 bg-zinc-950 px-2.5 py-1 text-xs text-zinc-400 font-mono"
                            >
                              <FileText className="w-3 h-3 text-zinc-500" />
                              {src || "document_chunk"}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </section>

          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-900/60 py-4 bg-zinc-950 text-center">
        <p className="text-[10px] text-zinc-600 tracking-wider">
          DoCopilot &copy; 2026. Powered by FastAPI, Next.js, and Qdrant Hybrid Search.
        </p>
      </footer>
    </div>
  );
}
