"use client";

import { useState } from "react";

const DECISIONS = [
  {
    title: "1. Why Qdrant for Vector Storage & Search?",
    badge: "Vector DB",
    color: "text-violet-400 bg-violet-400/10 border-violet-400/20",
    text: "Qdrant provides native built-in hybrid search (dense embeddings + sparse BM25 + Reciprocal Rank Fusion) in a single API call. This eliminated over 130 lines of complex manual BM25 and ranking code while providing disk persistence and tenant payload filtering.",
  },
  {
    title: "2. Why Groq (llama-3.3-70b) for LLM Inference?",
    badge: "Inference Engine",
    color: "text-emerald-400 bg-emerald-400/10 border-emerald-400/20",
    text: "Groq's custom LPU hardware delivers ~500 tokens/second — nearly 10x faster than traditional cloud GPUs. This makes streamed SSE token generation feel instantaneous and truly real-time for the user.",
  },
  {
    title: "3. Why Async Ingestion (HTTP 202 Accepted)?",
    badge: "Job State Machine",
    color: "text-amber-400 bg-amber-400/10 border-amber-400/20",
    text: "Indexing a 20-page PDF involves text extraction, chunking into 200+ fragments, and neural embedding, taking 15–45 seconds. Synchronous HTTP requests would cause gateway timeouts. Returning 202 immediately with a job_id lets the client poll status asynchronously.",
  },
  {
    title: "4. Why Two-Stage Retrieval (Hybrid → Rerank)?",
    badge: "RAG Precision",
    color: "text-sky-400 bg-sky-400/10 border-sky-400/20",
    text: "Stage 1 (Hybrid top-20) casts a wide net using both meaning and exact keywords. Stage 2 (Cohere Cross-Encoder top-5) re-evaluates candidate relevance using deep cross-attention, boosting answer correctness by +1.5% in evaluation benchmarks.",
  },
  {
    title: "5. Why JWT Auth & Multi-Tenancy Scoping?",
    badge: "Security & SaaS Isolation",
    color: "text-rose-400 bg-rose-400/10 border-rose-400/20",
    text: "JWTs are stateless — every API request verifies cryptographic signatures without querying the database. Claims embed tenant_id and user_id, ensuring every Qdrant query strictly filters payload metadata so workspace data never leaks.",
  },
  {
    title: "6. Why SSE (Server-Sent Events) over WebSockets?",
    badge: "Streaming Protocol",
    color: "text-cyan-400 bg-cyan-400/10 border-cyan-400/20",
    text: "SSE is lightweight and unidirectional (server to browser), working over standard HTTP/2. Unlike WebSockets, standard fetch() with readable streams allows sending HTTP Authorization headers with JWT tokens seamlessly.",
  },
];

export default function DesignDecisionsModal() {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        id="open-architecture-modal"
        onClick={() => setOpen(true)}
        className="text-xs px-2.5 py-1.5 rounded-lg border border-zinc-800 text-zinc-400 hover:text-zinc-200 hover:border-zinc-700 transition-colors flex items-center gap-1.5"
      >
        <span>⚙</span> Architecture & Decisions
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-zinc-950/80 backdrop-blur-md animate-fade-in">
          <div className="relative w-full max-w-2xl max-h-[85vh] flex flex-col rounded-2xl border border-zinc-800 bg-zinc-900 shadow-2xl overflow-hidden">
            {/* Modal Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800 bg-zinc-950/40">
              <div>
                <h3 className="text-base font-semibold text-zinc-100">
                  DoCopilot — System Design Decisions
                </h3>
                <p className="text-xs text-zinc-500 mt-0.5">
                  Architectural choices, trade-offs, and technical rationale
                </p>
              </div>
              <button
                onClick={() => setOpen(false)}
                className="text-zinc-500 hover:text-zinc-300 p-1 text-sm rounded-lg hover:bg-zinc-800 transition-colors"
              >
                ✕
              </button>
            </div>

            {/* Modal Body */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              {DECISIONS.map((item, idx) => (
                <div
                  key={idx}
                  className="rounded-xl border border-zinc-800/80 bg-zinc-950/50 p-4 space-y-2"
                >
                  <div className="flex items-center justify-between">
                    <h4 className="text-xs font-semibold text-zinc-200">{item.title}</h4>
                    <span className={`text-[10px] px-2 py-0.5 rounded border font-medium ${item.color}`}>
                      {item.badge}
                    </span>
                  </div>
                  <p className="text-xs text-zinc-400 leading-relaxed">{item.text}</p>
                </div>
              ))}
            </div>

            {/* Modal Footer */}
            <div className="flex items-center justify-between px-6 py-3 border-t border-zinc-800 bg-zinc-950/40">
              <span className="text-[10px] font-mono text-zinc-600">
                Built with FastAPI + Qdrant + Groq + Next.js
              </span>
              <button
                onClick={() => setOpen(false)}
                className="px-4 py-1.5 text-xs font-medium text-zinc-300 bg-zinc-800 hover:bg-zinc-700 rounded-lg transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
