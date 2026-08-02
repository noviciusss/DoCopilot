"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useAuth } from "../lib/hooks/useAuth";
import { useUpload, UPLOAD_STATUS_LABELS, IngestionStatus } from "../lib/hooks/useUpload";
import { apiStreamChat } from "../lib/api";

type UploadType = "pdf" | "txt" | "text";

const TABS: { id: UploadType; label: string; hint: string }[] = [
  { id: "pdf",  label: "PDF",        hint: "Upload a PDF document" },
  { id: "txt",  label: "TXT",        hint: "Upload a plain text file" },
  { id: "text", label: "Paste Text", hint: "Paste text directly" },
];

// Maps ingestion status to button label
const UPLOAD_BTN: Record<IngestionStatus, string> = {
  idle:      "Index Source",
  uploading: "Sending…",
  queued:    "Queued…",
  running:   "Embedding…",
  succeeded: "Re-index",
  failed:    "Retry",
};

export default function Home() {
  const router  = useRouter();
  const { isLoggedIn, loading: authLoading, logout } = useAuth();

  const { status: ingestionStatus, documentId, error: uploadError, upload, clearDocument } = useUpload();

  const [uploadType, setUploadType] = useState<UploadType>("pdf");
  const [file, setFile]             = useState<File | null>(null);
  const [plainText, setPlainText]   = useState("");

  const [question, setQuestion]           = useState("");
  const [streamingAnswer, setStreamingAnswer] = useState("");
  const [sources, setSources]             = useState<string[]>([]);
  const [isAsking, setIsAsking]           = useState(false);
  const cancelStreamRef = useRef<(() => void) | null>(null);
  const answerRef = useRef<HTMLDivElement>(null);

  // Redirect to /login if not authenticated (after auth state resolves)
  useEffect(() => {
    if (!authLoading && !isLoggedIn) router.push("/login");
  }, [isLoggedIn, authLoading, router]);

  // Auto-scroll answer into view while streaming
  useEffect(() => {
    if (streamingAnswer && answerRef.current) {
      answerRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [streamingAnswer]);

  const askDisabled = useMemo(
    () => !documentId || ingestionStatus !== "succeeded" || !question.trim() || isAsking,
    [documentId, ingestionStatus, question, isAsking]
  );

  const handleUpload = async (evt: FormEvent<HTMLFormElement>) => {
    evt.preventDefault();
    if (uploadType !== "text" && !file) return;
    if (uploadType === "text" && !plainText.trim()) return;
    await upload(file, plainText, uploadType);
  };

  const handleChat = async (evt: FormEvent<HTMLFormElement>) => {
    evt.preventDefault();
    if (askDisabled || !documentId) return;

    setIsAsking(true);
    setStreamingAnswer("");
    setSources([]);

    cancelStreamRef.current = apiStreamChat(
      question,
      documentId,
      (token) => setStreamingAnswer((prev) => prev + token),
      (srcs, fullAns) => {
        setSources(srcs);
        if (fullAns) setStreamingAnswer(fullAns);
        setIsAsking(false);
      },
      (err) => {
        setStreamingAnswer("⚠ " + err);
        setIsAsking(false);
      }
    );
  };

  const handleClearSession = () => {
    cancelStreamRef.current?.();
    clearDocument();
    setFile(null);
    setPlainText("");
    setStreamingAnswer("");
    setSources([]);
  };

  // Show loading state while auth resolves to prevent flash of redirect
  if (authLoading) {
    return (
      <div className="min-h-screen bg-zinc-950 flex items-center justify-center">
        <span className="inline-block h-6 w-6 rounded-full border-2 border-zinc-500 border-t-transparent animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col selection:bg-zinc-800 selection:text-white">

      {/* ── Header / Navigation ──────────────────────────────────── */}
      <nav className="sticky top-0 z-20 border-b border-zinc-900 bg-zinc-950/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-zinc-100 text-sm font-bold text-zinc-950 select-none shadow-sm shadow-white/10">
              D
            </span>
            <div className="flex flex-col">
              <span className="text-sm font-semibold tracking-tight text-zinc-100 leading-none">DoCopilot</span>
              <span className="text-[10px] text-zinc-500 tracking-wider">Enterprise RAG</span>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Ingestion status badge */}
            {ingestionStatus === "running" || ingestionStatus === "queued" ? (
              <span className="flex items-center gap-1.5 text-xs text-amber-400 font-mono">
                <span className="h-1.5 w-1.5 rounded-full bg-amber-400 animate-pulse" />
                {ingestionStatus === "queued" ? "Queued" : "Indexing…"}
              </span>
            ) : null}

            {documentId && ingestionStatus === "succeeded" ? (
              <div className="flex items-center gap-3">
                <span className="flex h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="hidden sm:inline-block font-mono text-xs text-zinc-400 bg-zinc-900 border border-zinc-800 px-2.5 py-1 rounded-md">
                  Active ID: {documentId.slice(0, 8)}…{documentId.slice(-8)}
                </span>
                <button
                  type="button"
                  onClick={handleClearSession}
                  className="text-xs text-zinc-500 hover:text-zinc-300 bg-zinc-900 hover:bg-zinc-850 border border-zinc-850 px-2.5 py-1 rounded-md transition-colors"
                >
                  Unload Document
                </button>
              </div>
            ) : (
              <span className="text-xs text-zinc-500 flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-zinc-700" />
                No active document
              </span>
            )}

            {/* Logout */}
            <button
              onClick={logout}
              className="text-xs text-zinc-500 hover:text-zinc-300 border border-zinc-800 px-3 py-1.5 rounded-lg transition-colors"
            >
              Sign out
            </button>
          </div>
        </div>
      </nav>

      {/* ── Main Layout ────────────────────────────────────────────── */}
      <main className="mx-auto max-w-6xl w-full px-6 py-10 flex-1 grid grid-cols-1 lg:grid-cols-12 gap-8">

        {/* Left Column — Upload & Settings */}
        <section className="lg:col-span-5 space-y-6">
          <div className="space-y-1">
            <h1 className="text-xl font-medium tracking-tight text-zinc-200">Document Workspace</h1>
            <p className="text-xs text-zinc-500">
              Select or paste document sources to index them into Qdrant hybrid vector store.
            </p>
          </div>

          <div className="rounded-2xl border border-zinc-900 bg-zinc-900/40 backdrop-blur-sm p-6 space-y-6 transition-all duration-300 hover:border-zinc-800">
            {/* Type tabs */}
            <div className="flex gap-1.5 rounded-xl bg-zinc-950 p-1 border border-zinc-900">
              {TABS.map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => { setUploadType(tab.id); setFile(null); }}
                  className={`flex-1 text-center py-2 rounded-lg text-xs font-medium transition-all duration-200 ${
                    uploadType === tab.id
                      ? "bg-zinc-900 text-zinc-100 shadow-sm border border-zinc-800/80"
                      : "text-zinc-500 hover:text-zinc-300"
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            <form onSubmit={handleUpload} className="space-y-5">
              {uploadType === "text" ? (
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-zinc-400">Raw Content</label>
                  <textarea
                    value={plainText}
                    onChange={(e) => setPlainText(e.target.value)}
                    rows={8}
                    placeholder="Paste text source here..."
                    className="w-full rounded-xl border border-zinc-800/80 bg-zinc-950/60 px-4 py-3 text-xs text-zinc-100 placeholder:text-zinc-650 outline-none focus:border-zinc-700 focus:bg-zinc-950 transition-all resize-none shadow-inner"
                  />
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <label className="text-xs font-medium text-zinc-400">
                      {uploadType === "pdf" ? "Select PDF Document" : "Select TXT File"}
                    </label>
                    <span className="text-[10px] text-zinc-600 uppercase font-mono">Max 20MB</span>
                  </div>

                  <div className="border border-dashed border-zinc-800 hover:border-zinc-700 transition-colors rounded-xl bg-zinc-950/40 p-6 flex flex-col items-center justify-center text-center cursor-pointer group relative">
                    <input
                      type="file"
                      accept={uploadType === "pdf" ? "application/pdf" : "text/plain,.txt"}
                      onChange={(e) => { setFile(e.target.files?.[0] ?? null); }}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    />
                    <svg className="w-8 h-8 text-zinc-600 group-hover:text-zinc-500 transition-colors mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    <span className="text-xs text-zinc-400 group-hover:text-zinc-300 transition-colors font-medium">
                      {file ? file.name : `Click or drag your ${uploadType.toUpperCase()} file here`}
                    </span>
                    {file && (
                      <span className="text-[10px] text-zinc-600 mt-1 font-mono">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </span>
                    )}
                  </div>
                </div>
              )}

              <div className="space-y-3 pt-2">
                <button
                  type="submit"
                  disabled={ingestionStatus === "uploading" || ingestionStatus === "queued" || ingestionStatus === "running"}
                  className="w-full rounded-xl bg-zinc-100 hover:bg-white text-zinc-950 font-semibold py-2.5 text-xs transition-colors shadow-sm disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {(ingestionStatus === "uploading" || ingestionStatus === "queued" || ingestionStatus === "running") ? (
                    <>
                      <span className="inline-block h-3.5 w-3.5 rounded-full border-2 border-zinc-950 border-t-transparent animate-spin" />
                      {UPLOAD_BTN[ingestionStatus]}
                    </>
                  ) : UPLOAD_BTN[ingestionStatus]}
                </button>

                {/* Ingestion status message */}
                {ingestionStatus !== "idle" && (
                  <div className={`p-3 rounded-lg border text-xs flex items-start gap-2 ${
                    ingestionStatus === "succeeded"
                      ? "bg-emerald-950/20 border-emerald-900/60 text-emerald-350"
                      : ingestionStatus === "failed"
                      ? "bg-red-950/20 border-red-900/60 text-red-350"
                      : "bg-amber-950/20 border-amber-900/60 text-amber-300"
                  }`}>
                    <span className={`h-1.5 w-1.5 rounded-full mt-1.5 flex-shrink-0 ${
                      ingestionStatus === "succeeded" ? "bg-emerald-400" :
                      ingestionStatus === "failed" ? "bg-red-400" : "bg-amber-400 animate-pulse"
                    }`} />
                    <span className="flex-1">
                      {uploadError ?? UPLOAD_STATUS_LABELS[ingestionStatus]}
                    </span>
                  </div>
                )}
              </div>
            </form>
          </div>
        </section>

        {/* Right Column — Chat */}
        <section className="lg:col-span-7 flex flex-col space-y-6">
          <div className="space-y-1">
            <h1 className="text-xl font-medium tracking-tight text-zinc-200">Copilot Chat</h1>
            <p className="text-xs text-zinc-500">
              Ask questions backed by semantic context extraction, BM25, and hybrid re-ranking.
            </p>
          </div>

          <div className="rounded-2xl border border-zinc-900 bg-zinc-900/40 backdrop-blur-sm p-6 space-y-6 transition-all duration-300 hover:border-zinc-800">
            <form onSubmit={handleChat} className="space-y-4">
              <div className="relative">
                <textarea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  rows={3}
                  placeholder={
                    ingestionStatus === "succeeded"
                      ? "Ask a question about the document..."
                      : "Upload and index a document to unlock chat"
                  }
                  disabled={ingestionStatus !== "succeeded"}
                  className="w-full rounded-xl border border-zinc-800 bg-zinc-950/70 px-4 py-3 text-xs text-zinc-150 placeholder:text-zinc-600 outline-none focus:border-zinc-700 transition-colors resize-none disabled:opacity-40 disabled:cursor-not-allowed shadow-inner"
                />
              </div>

              <div className="flex justify-between items-center">
                <span className="text-[10px] text-zinc-600 font-mono">Rate Limit: 20 / min</span>
                <button
                  type="submit"
                  disabled={askDisabled}
                  className="rounded-lg bg-zinc-100 hover:bg-white text-zinc-950 font-semibold px-5 py-2 text-xs transition-colors shadow-sm disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  {isAsking ? (
                    <>
                      <span className="inline-block h-3.5 w-3.5 rounded-full border-2 border-zinc-950 border-t-transparent animate-spin" />
                      Synthesizing…
                    </>
                  ) : "Submit Query"}
                </button>
              </div>
            </form>
          </div>

          {/* Answer Area */}
          {(streamingAnswer || isAsking) && (
            <div ref={answerRef} className="rounded-2xl border border-zinc-900 bg-zinc-900/40 backdrop-blur-sm p-6 space-y-6 transition-all duration-300 hover:border-zinc-800">
              <div className="flex items-center justify-between border-b border-zinc-900 pb-3">
                <div className="flex items-center gap-2">
                  <h2 className="text-xs font-semibold uppercase tracking-widest text-zinc-500">
                    Response Output
                  </h2>
                  {isAsking && <span className="h-1.5 w-1.5 rounded-full bg-zinc-400 animate-pulse" />}
                </div>
                <span className="text-[10px] font-mono text-zinc-600 bg-zinc-950 px-2 py-0.5 rounded border border-zinc-900">SSE Streamed</span>
              </div>

              <div className="space-y-6">
                <article className="
                  prose prose-sm prose-invert max-w-none
                  prose-p:text-zinc-300 prose-p:leading-relaxed prose-p:my-2.5 prose-p:text-[13px]
                  prose-headings:text-zinc-100 prose-headings:font-medium prose-headings:tracking-tight
                  prose-strong:text-zinc-200 prose-strong:font-semibold
                  prose-em:text-zinc-400
                  prose-code:text-zinc-200 prose-code:bg-zinc-950 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-[11px] prose-code:before:content-none prose-code:after:content-none prose-code:border prose-code:border-zinc-900
                  prose-pre:bg-zinc-950 prose-pre:border prose-pre:border-zinc-900 prose-pre:rounded-xl prose-pre:p-4
                  prose-ul:text-zinc-300 prose-ol:text-zinc-300 prose-ul:my-2.5 prose-ol:my-2.5 prose-ul:list-disc
                  prose-li:my-1 prose-li:marker:text-zinc-700 prose-li:text-[13px]
                  prose-blockquote:border-zinc-700 prose-blockquote:text-zinc-400 prose-blockquote:pl-4 prose-blockquote:italic
                  prose-hr:border-zinc-900
                  prose-table:text-xs prose-table:my-4
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
                  <div className="border-t border-zinc-900 pt-4 space-y-2">
                    <p className="text-[10px] font-semibold uppercase tracking-widest text-zinc-600 font-mono">
                      Referenced Documents
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {sources.map((src, i) => (
                        <span
                          key={`${src}-${i}`}
                          className="inline-flex items-center rounded-lg border border-zinc-800 bg-zinc-950 px-2.5 py-1 text-xs text-zinc-400 font-mono"
                        >
                          {src || "unnamed_source"}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </section>

      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-950/60 py-6 bg-zinc-950 text-center">
        <p className="text-[10px] text-zinc-650 tracking-wider">
          DoCopilot &copy; 2026. Built with FastAPI, Next.js, and Qdrant.
        </p>
      </footer>
    </div>
  );
}
