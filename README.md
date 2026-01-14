# DoCopilot

A Next.js + FastAPI RAG app: upload PDFs or text, embed them into Qdrant (hybrid search), and chat with sourced answers. LangSmith is available for tracing.

## Project Structure
- **backend/** FastAPI API, RAG pipeline, Qdrant vector store
- **frontend/** Next.js UI (app router) with PDF upload + chat

## Prerequisites
- Python 3.10+ with pip or conda
- Node.js 18+
- Groq API key for LLM
- (Optional) LangSmith API key for tracing

## Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Environment (place in `.env`, do not commit):
```
GROQ_API_KEY=your_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=DoCopilot
ALLOWED_ORIGINS=http://localhost:3000
```

## Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
The UI calls the backend at `http://localhost:8000` by default. Override with `NEXT_PUBLIC_API_BASE`.

## Usage
1) Upload a PDF/TXT/plain text → receives `document_id`.
2) Ask a question referencing that `document_id` → answer with `sources` is returned.

---

##  Latest Evaluation Results (Qdrant Hybrid)

```
============================================================
EVALUATION SUMMARY
============================================================
Total Questions:      40
Successful:           40/40

--- LLM-Based Scores (Semantic) ---
Avg Correctness:      89.2%
Avg Relevance:        90.5%

--- Keyword-Based Scores (Baseline) ---
Avg Correctness:      57.7%
Avg Relevance:        57.8%

--- Other Metrics ---
Has Sources Rate:     100.0%
Avg Latency:          2.86s
============================================================
```

| Metric | Score |
|--------|-------|
| **Total Questions** | 40 |
| **Success Rate** | 100% (40/40) |
| **Avg Correctness (LLM)** | 89.2% |
| **Avg Relevance (LLM)** | 90.5% |
| **Avg Correctness (Keyword)** | 57.7% |
| **Avg Relevance (Keyword)** | 57.8% |
| **Has Sources Rate** | 100% |
| **Avg Latency** | 2.86s |

>  **LLM-based evaluation** uses semantic understanding to judge answer quality.  
> **Keyword-based** is a baseline using exact string matching.

---

##  Current Architecture (Qdrant Hybrid)

```
Question
    │
    ▼
┌─────────────────────────────────────┐
│      QDRANT HYBRID SEARCH           │
│  ┌───────────┐    ┌───────────┐     │
│  │  Dense    │    │  Sparse   │     │
│  │ (Vector)  │    │  (BM25)   │     │
│  └───────────┘    └───────────┘     │
│         │              │            │
│         └──────┬───────┘            │
│                ▼                    │
│         RRF Fusion (built-in)       │
└─────────────────────────────────────┘
    │
    ▼ Top 20
┌─────────────────────────────────────┐
│      CROSS-ENCODER RERANK           │
└─────────────────────────────────────┘
    │
    ▼ Top 5
┌─────────────────────────────────────┐
│      LLM (Llama-4-Scout)            │
└─────────────────────────────────────┘
    │
    ▼
Answer + Citations
```

---

##  Known Issues & Solutions

> ###  Qdrant Local Storage Lock
> 
> **Error:**
> ```
> RuntimeError: Storage folder ./qdrant_data is already accessed 
> by another instance of Qdrant client
> ```
> 
> **Cause:** Qdrant local mode (`path=`) uses file locking. Only ONE client can access at a time.
> 
> **Solutions:**
> | Option | Code | Persistence |
> |--------|------|-------------|
> | In-Memory | `location=":memory:"` |  No |
> | Docker Server | `url="http://localhost:6333"` |  Yes |
> | Qdrant Cloud | `url="https://xxx.cloud.qdrant.io"` |  Yes |
> 
> **Current:** Using `:memory:` for development (no persistence).

---

> ###  Version Mismatch Error
> 
> **Error:**
> ```
> TypeError: Client.__init__() got an unexpected keyword argument 'client'
> ```
> 
> **Cause:** `langchain-qdrant` version incompatible with `qdrant-client`.
> 
> **Solution:** Use `location=` or `url=` instead of `client=` parameter.

---

## Historical Evaluation Results

### Ablation Study - Chunk Size Comparison (LLM Judge)

| Config | Chunk Size | Overlap | Correctness | Relevance | Sources | Latency |
|--------|------------|---------|-------------|-----------|---------|---------|
| Small | 500 | 100 | 88.5% | 88.7% | 100% | 6.9s |
| Medium | 1000 | 200 | 85.5% | 86.5% | 100% | 9.8s |
| **Large** | **2000** | **400** | **87.7%** | **89.0%** | **100%** | **2.1s** |

### Keyword-Based Scores (for reference)

| Config | Chunk Size | Overlap | Correctness | Relevance |
|--------|------------|---------|-------------|-----------|
| Small | 500 | 100 | 48.2% | 39.4% |
| Medium | 1000 | 200 | 47.3% | 36.3% |
| Large | 2000 | 400 | 52.3% | 57.1% |

### Ablation Study - Retrieval Methods (LLM Judge, 2000/400 chunks)

| Config | Method | Correctness | Relevance | Latency |
|--------|--------|-------------|-----------|---------|
| Baseline | Vector only | 87.7% | 89.0% | 2.1s |
| + Rerank | Vector + Rerank | 87.7% | 89.0% | 3.0s |
| + Hybrid (FAISS) | BM25 + Vector + RRF + Rerank | 88.7% | 90.7% | 2.2s |
| **+ Qdrant Hybrid** | **Qdrant built-in + Rerank** | **89.2%** | **90.5%** | **2.86s** |

### Best Config: Qdrant Hybrid + Rerank

```json
{
  "chunk_size": 2000,
  "chunk_overlap": 400,
  "retrieval": "Qdrant hybrid (dense + sparse + RRF)",
  "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "initial_k": 20,
  "final_k": 5,
  "eval_method": "LLM-as-Judge (Groq llama-3.1-8b)",
  "llm_correctness": 0.892,
  "llm_relevance": 0.905,
  "keyword_correctness": 0.577,
  "keyword_relevance": 0.578,
  "has_sources_rate": 1.0,
  "avg_latency": 2.86
}
```

---

##  FAISS + BM25 vs Qdrant Comparison

| Aspect | FAISS + Manual BM25 | Qdrant Hybrid |
|--------|---------------------|---------------|
| Lines of Code | ~150 | ~20 |
| Hybrid Search | Manual RRF fusion | Built-in |
| Persistence | In-memory only | Disk/Cloud |
| Correctness | 88.7% | **89.2%** |
| Relevance | 90.7% | 90.5% |
| Latency | 2.2s | 2.86s |
| Maintenance | Two indexes | Single system |

> Qdrant slightly higher correctness, similar relevance, slightly slower due to sparse embedding computation.

---

<details>
<summary><strong>Why Hybrid Helped</strong></summary>

| Reason | Explanation |
|--------|-------------|
| Exact keyword matches | BM25 finds "EC2", "S3" even if embeddings differ |
| Semantic understanding | Vector finds synonyms and paraphrases |
| RRF combines both | Docs appearing in both lists get highest scores |
| Complementary strengths | Each method covers the other's weaknesses |

</details>
<details>
<summary><strong>How Qdrant Hybrid Search Works</strong></summary>

```
Query: "What is EC2 pricing?"
         │
         ▼
┌─────────────────────────────────────────────────┐
│           QDRANT (Single Index)                 │
│                                                 │
│  ┌─────────────────┐  ┌─────────────────┐      │
│  │  Dense Vectors  │  │ Sparse Vectors  │      │
│  │  (MiniLM-L6)    │  │ (Qdrant/bm25)   │      │
│  └────────┬────────┘  └────────┬────────┘      │
│           │                    │                │
│           └────────┬───────────┘                │
│                    ▼                            │
│           RRF Fusion (automatic)                │
└─────────────────────────────────────────────────┘
                    │
                    ▼
         Top 20 → Reranker → Top 5 → LLM
```

**Replaces ~100 lines of manual BM25 + RRF code!**

</details>

### Evaluation Methods Comparison

| Method | How it works | Pros | Cons |
|--------|--------------|------|------|
| **Keyword** | Word overlap between expected/predicted | Fast, free | Misses synonyms, underestimates |
| **LLM Judge** | LLM scores semantic similarity | Accurate, understands meaning | Extra API calls, slight bias |

---

##  Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INITIAL_K` | 20 | Docs retrieved before reranking |
| `FINAL_K` | 5 | Docs after reranking |
| `chunk_size` | 2000 | Characters per chunk |
| `chunk_overlap` | 400 | Overlap between chunks |
| `HYBRID_ENABLED` | True | Use Qdrant hybrid search |
| `RERANK_ENABLED` | True | Use CrossEncoder reranking |

---

##  Tech Stack

| Component | Technology |
|-----------|------------|
| **Vector DB** | Qdrant (hybrid: dense + sparse) |
| **Dense Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Sparse Embeddings** | `Qdrant/bm25` (FastEmbed) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **LLM** | Llama-4-Scout via Groq |
| **Eval LLM** | `llama-3.1-8b-instant` via Groq |
| **Framework** | LangChain + FastAPI |
| **Tracing** | LangSmith |

---

<details>
<summary><strong>Roadmap</strong></summary>

| Week | Change | Status |
|------|--------|--------|
| 1 | Baseline v1 + eval | ✅ Done |
| 2 | Chunking ablation + LLM eval | ✅ Done |
| 3 | Reranking | ✅ Done |
| 4 | Hybrid retrieval (BM25 + Vector + RRF) | ✅ Done |
| 5 | Vector DB swap (Qdrant) | ✅ Done |
| 6 | Query rewriting / HyDE | 🔜 Next |
| 7 | Multi-document support | 📋 Planned |
| 8 | Final report + ablation table | 📋 Planned |

</details>

---

## Notes
- `.env` is ignored by git (see root `.gitignore`).
- Embeddings preload on server start for faster indexing after the first request.
- Run `python evaluate_local.py` in `backend/` to reproduce evaluation results.
- LLM-as-Judge uses a different model (`llama-3.1-8b`) than RAG to avoid self-bias.
- **Conclusion**: Qdrant hybrid search provides best quality with minimal code.

## 📄 License

MIT License
