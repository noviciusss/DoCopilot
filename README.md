# DoCopilot

A Next.js + FastAPI RAG app: upload PDFs or text, embed them into FAISS, and chat with sourced answers. LangSmith is available for tracing.

## Project Structure
- **backend/** FastAPI API, RAG pipeline, FAISS store
- **frontend/** Next.js UI (app router) with PDF upload + chat

## Prerequisites
- Python 3.10+ with pip or conda
- Node.js 18+
- Google API key for Gemini (if using ChatGoogleGenerativeAI)
- (Optional) LangSmith API key for tracing

## Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Environment (place in `.env`, do not commit):
```
GOOGLE_API_KEY=your_key
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

## RAG Evaluation Results

Evaluated on **40 Q&A pairs** from the AWS Overview whitepaper.

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
| **+ Hybrid** | **BM25 + Vector + RRF + Rerank** | **88.7%** | **90.7%** | **2.2s** |

### Best Config: 2000/400 + Hybrid Retrieval

```json
{
  "chunk_size": 2000,
  "chunk_overlap": 400,
  "retrieval": "hybrid (BM25 + Vector + RRF)",
  "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "initial_k": 20,
  "final_k": 5,
  "rrf_k": 60,
  "eval_method": "LLM-as-Judge (Groq llama-3.1-8b)",
  "llm_correctness": 0.887,
  "llm_relevance": 0.907,
  "keyword_correctness": 0.555,
  "keyword_relevance": 0.589,
  "has_sources_rate": 1.0,
  "avg_latency": 2.16
}
```

### Key Findings

| Finding | Insight |
|---------|---------|
| Hybrid wins overall | +1% correctness, +1.7% relevance vs baseline |
| BM25 + Vector > Vector alone | Keyword matching catches what embeddings miss |
| RRF fusion works | Chunks in both lists get boosted |
| Faster than rerank-only | 2.2s vs 3.0s (BM25 is lightweight) |
| 100% source grounding | All answers cite retrieved context |

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
<summary><strong>How Hybrid Retrieval Works</strong></summary>

```
Query: "What is EC2 pricing?"
         │
         ├──► BM25 Search ───────► [Chunk15, Chunk8, Chunk22, ...]
         │    (keyword match)       (has "EC2", "pricing" words)
         │
         └──► Vector Search ─────► [Chunk3, Chunk15, Chunk7, ...]
              (semantic similarity)  (about "compute costs")
                    │
                    ▼
         ┌─────────────────────────────────────────┐
         │  Reciprocal Rank Fusion (RRF)           │
         │                                         │
         │  Chunk15: in BOTH lists → highest score │
         │  Chunk3:  vector only → medium score    │
         │  Chunk8:  BM25 only → medium score      │
         └─────────────────────────────────────────┘
                    │
                    ▼
              Top 5 → Reranker → LLM
```

</details>

### Evaluation Methods Comparison

| Method | How it works | Pros | Cons |
|--------|--------------|------|------|
| **Keyword** | Word overlap between expected/predicted | Fast, free | Misses synonyms, underestimates |
| **LLM Judge** | LLM scores semantic similarity | Accurate, understands meaning | Extra API calls, slight bias |

### RAG Stack
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
- **Vector Store**: FAISS (in-memory)
- **BM25**: `rank-bm25` (keyword search)
- **Fusion**: Reciprocal Rank Fusion (k=60)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM**: Groq `meta-llama/llama-4-scout-17b-16e-instruct`
- **Eval LLM**: Groq `llama-3.1-8b-instant`
- **Retrieval**: BM25 (Top 20) + Vector (Top 20) → RRF → Top 5 → Rerank

<details>
<summary><strong>Roadmap</strong></summary>

| Week | Change | Status |
|------|--------|--------|
| 1 | Baseline v1 + eval | Done |
| 2 | Chunking ablation + LLM eval | Done |
| 3 | Reranking | Done |
| 4 | Hybrid retrieval (BM25 + Vector + RRF) | Done |
| 5 | Query rewriting / HyDE | Next |
| 6 | Vector DB swap (Qdrant) | Planned |
| 7 | Multi-document support | Planned |
| 8 | Final report + ablation table | Planned |

</details>

## Notes
- `.env` is ignored by git (see root `.gitignore`).
- Embeddings preload on server start for faster indexing after the first request.
- Run `python evaluate_local.py` in `backend/` to reproduce evaluation results.
- LLM-as-Judge uses a different model (`llama-3.1-8b`) than RAG to avoid self-bias.
- **Conclusion**: Hybrid retrieval (BM25 + Vector + RRF) provides best quality for this dataset.
