"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

type UploadType = "pdf" | "txt" | "text";
type UploadResponse = { document_id: string };

const TABS: { id: UploadType; label: string; hint: string }[] = [
  { id: "pdf",  label: "PDF",        hint: "Upload a PDF document" },
  { id: "txt",  label: "TXT",        hint: "Upload a plain text file" },
  { id: "text", label: "Paste Text", hint: "Paste text directly" },
];

export default function Home() {
  const [uploadType, setUploadType]       = useState<UploadType>("pdf");
  const [file, setFile]                   = useState<File | null>(null);
  const [plainText, setPlainText]         = useState("");
  const [uploadStatus, setUploadStatus]   = useState<{ msg: string; ok: boolean } | null>(null);
  const [documentId, setDocumentId]       = useState("");
  const [question, setQuestion]           = useState("");
  const [streamingAnswer, setStreamingAnswer] = useState("");
  const [sources, setSources]             = useState<string[]>([]);
  const [isUploading, setIsUploading]     = useState(false);
  const [isAsking, setIsAsking]           = useState(false);
  const answerRef = useRef<HTMLDivElement>(null);

  const askDisabled = useMemo(
    () => !documentId || !question.trim() || isAsking,
    [documentId, question, isAsking]
  );

  // Restore document_id from sessionStorage on mount (Task 4.2)
  useEffect(() => {
    const saved = sessionStorage.getItem("document_id");
    if (saved) setDocumentId(saved);
  }, []);

  // Auto-scroll answer section into view while streaming
  useEffect(() => {
    if (streamingAnswer && answerRef.current) {
      answerRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [streamingAnswer]);

  const handleUpload = async (evt: FormEvent<HTMLFormElement>) => {
    evt.preventDefault();
    const body = new FormData();

    if (uploadType === "pdf" || uploadType === "txt") {
      if (!file) {
        setUploadStatus({ msg: `Please choose a ${uploadType.toUpperCase()} file first.`, ok: false });
        return;
      }
      body.append(uploadType === "pdf" ? "pdf_file" : "txt_file", file);
    } else {
      if (!plainText.trim()) {
        setUploadStatus({ msg: "Please paste some text first.", ok: false });
        return;
      }
      body.append("plain_text", plainText);
    }

    setIsUploading(true);
    setUploadStatus({ msg: "Uploading & indexing…", ok: true });

    try {
      const res = await fetch(`${API_BASE}/upload`, { method: "POST", body });
      if (!res.ok) throw new Error((await res.text()) || "Upload failed");
      const data = (await res.json()) as UploadResponse;
      sessionStorage.setItem("document_id", data.document_id);
      setDocumentId(data.document_id);
      setUploadStatus({ msg: "Indexed successfully — ready to chat.", ok: true });
    } catch (e) {
      setUploadStatus({ msg: String(e) || "Upload failed.", ok: false });
    } finally {
      setIsUploading(false);
    }
  };

  // Streaming chat using fetch + ReadableStream (Task 4.1)
  const handleChat = async (evt: FormEvent<HTMLFormElement>) => {
    evt.preventDefault();
    if (askDisabled) return;

    setIsAsking(true);
    setStreamingAnswer("");
    setSources([]);

    try {
      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, document_id: documentId || null }),
      });

      if (!res.ok) throw new Error("Chat request failed");
      if (!res.body)  throw new Error("No response body");

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() ?? "";

        for (const part of parts) {
          if (!part.startsWith("data: ")) continue;
          try {
            const ev = JSON.parse(part.slice(6));
            if (ev.done) {
              setSources(ev.sources ?? []);
              if (ev.answer) setStreamingAnswer(ev.answer);
            } else if (ev.token) {
              setStreamingAnswer((prev) => prev + ev.token);
            } else if (ev.error) {
              setStreamingAnswer("⚠ " + ev.error);
            }
          } catch {
            // malformed SSE chunk — ignore
          }
        }
      }
    } catch (e) {
      setStreamingAnswer("Something went wrong — check the backend server.");
    } finally {
      setIsAsking(false);
    }
  };

  const handleClearSession = () => {
    sessionStorage.removeItem("document_id");
    setDocumentId("");
    setUploadStatus(null);
    setFile(null);
    setPlainText("");
    setStreamingAnswer("");
    setSources([]);
  };

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
            {documentId ? (
              <div className="flex items-center gap-3">
                <span className="flex h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="hidden sm:inline-block font-mono text-xs text-zinc-400 bg-zinc-900 border border-zinc-800 px-2.5 py-1 rounded-md">
                  Active ID: {documentId.slice(0, 8)}...{documentId.slice(-8)}
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
          </div>
        </div>
      </nav>

      {/* ── Main Layout (Dashboard Grid) ─────────────────────────── */}
      <main className="mx-auto max-w-6xl w-full px-6 py-10 flex-1 grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left Column (Upload & Settings Control Panel) - 5 Cols */}
        <section className="lg:col-span-5 space-y-6">
          <div className="space-y-1">
            <h1 className="text-xl font-medium tracking-tight text-zinc-200">
              Document Workspace
            </h1>
            <p className="text-xs text-zinc-500">
              Select or paste document sources to index them into Qdrant hybrid vector store.
            </p>
          </div>

          {/* Upload Settings Card */}
          <div className="rounded-2xl border border-zinc-900 bg-zinc-900/40 backdrop-blur-sm p-6 space-y-6 transition-all duration-300 hover:border-zinc-800">
            {/* Tab Swappers */}
            <div className="flex gap-1.5 rounded-xl bg-zinc-950 p-1 border border-zinc-900">
              {TABS.map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => { 
                    setUploadType(tab.id); 
                    setFile(null); 
                    setUploadStatus(null); 
                  }}
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

            {/* Input fields based on selection */}
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
                  
                  {/* File zone styling */}
                  <div className="border border-dashed border-zinc-800 hover:border-zinc-700 transition-colors rounded-xl bg-zinc-950/40 p-6 flex flex-col items-center justify-center text-center cursor-pointer group relative">
                    <input
                      type="file"
                      accept={uploadType === "pdf" ? "application/pdf" : "text/plain,.txt"}
                      onChange={(e) => { 
                        setFile(e.target.files?.[0] ?? null); 
                        setUploadStatus(null); 
                      }}
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

              {/* Action area */}
              <div className="space-y-3 pt-2">
                <button
                  type="submit"
                  disabled={isUploading}
                  className="w-full rounded-xl bg-zinc-100 hover:bg-white text-zinc-950 font-semibold py-2.5 text-xs transition-colors shadow-sm disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isUploading ? (
                    <>
                      <span className="inline-block h-3.5 w-3.5 rounded-full border-2 border-zinc-950 border-t-transparent animate-spin" />
                      Indexing Data...
                    </>
                  ) : "Index Source"}
                </button>

                {uploadStatus && (
                  <div className={`p-3 rounded-lg border text-xs flex items-start gap-2 ${
                    uploadStatus.ok 
                      ? "bg-emerald-950/20 border-emerald-900/60 text-emerald-350" 
                      : "bg-red-950/20 border-red-900/60 text-red-350"
                  }`}>
                    <span className={`h-1.5 w-1.5 rounded-full mt-1.5 ${uploadStatus.ok ? "bg-emerald-400" : "bg-red-400"}`} />
                    <span className="flex-1">{uploadStatus.msg}</span>
                  </div>
                )}
              </div>
            </form>
          </div>
        </section>

        {/* Right Column (Ask & Answers Workspace) - 7 Cols */}
        <section className="lg:col-span-7 flex flex-col space-y-6">
          <div className="space-y-1">
            <h1 className="text-xl font-medium tracking-tight text-zinc-200">
              Copilot Chat
            </h1>
            <p className="text-xs text-zinc-500">
              Ask questions backed by semantic context extraction, BM25, and hybrid re-ranking.
            </p>
          </div>

          {/* Chat card */}
          <div className="rounded-2xl border border-zinc-900 bg-zinc-900/40 backdrop-blur-sm p-6 space-y-6 transition-all duration-300 hover:border-zinc-800">
            <form onSubmit={handleChat} className="space-y-4">
              <div className="relative">
                <textarea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  rows={3}
                  placeholder={documentId ? "Ask a question about the document..." : "Please upload a document to unlock the chat console"}
                  disabled={!documentId}
                  className="w-full rounded-xl border border-zinc-800 bg-zinc-950/70 px-4 py-3 text-xs text-zinc-150 placeholder:text-zinc-600 outline-none focus:border-zinc-700 transition-colors resize-none disabled:opacity-40 disabled:cursor-not-allowed shadow-inner"
                />
              </div>

              <div className="flex justify-between items-center">
                <span className="text-[10px] text-zinc-600 font-mono">Rate Limit: 10 / min</span>
                <button
                  type="submit"
                  disabled={askDisabled}
                  className="rounded-lg bg-zinc-100 hover:bg-white text-zinc-950 font-semibold px-5 py-2 text-xs transition-colors shadow-sm disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  {isAsking ? (
                    <>
                      <span className="inline-block h-3.5 w-3.5 rounded-full border-2 border-zinc-950 border-t-transparent animate-spin" />
                      Synthesizing...
                    </>
                  ) : "Submit Query"}
                </button>
              </div>
            </form>
          </div>

          {/* Answer Area - Rendered conditionally */}
          {(streamingAnswer || isAsking) && (
            <div ref={answerRef} className="rounded-2xl border border-zinc-900 bg-zinc-900/40 backdrop-blur-sm p-6 space-y-6 transition-all duration-300 hover:border-zinc-800">
              <div className="flex items-center justify-between border-b border-zinc-900 pb-3">
                <div className="flex items-center gap-2">
                  <h2 className="text-xs font-semibold uppercase tracking-widest text-zinc-500">
                    Response Output
                  </h2>
                  {isAsking && (
                    <span className="h-1.5 w-1.5 rounded-full bg-zinc-400 animate-pulse" />
                  )}
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
