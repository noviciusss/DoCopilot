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

### Ablation Study - Reranking (LLM Judge, 2000/400 chunks)

| Config | Reranker | Initial K | Final K | Correctness | Relevance | Latency |
|--------|----------|-----------|---------|-------------|-----------|---------|
| Baseline | None | 5 | 5 | 87.7% | 89.0% | 2.1s |
| + Rerank | ms-marco-MiniLM-L-6-v2 | 20 | 5 | 87.7% | 89.0% | 3.0s |

### Best Config: 2000/400

```json
{
  "chunk_size": 2000,
  "chunk_overlap": 400,
  "eval_method": "LLM-as-Judge (Groq llama-3.1-8b)",
  "llm_correctness": 0.877,
  "llm_relevance": 0.890,
  "keyword_correctness": 0.523,
  "keyword_relevance": 0.571,
  "has_sources_rate": 1.0,
  "avg_latency": 2.12
}
```

### Best Config: 2000/400 + Reranking

```json
{
  "chunk_size": 2000,
  "chunk_overlap": 400,
  "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "initial_k": 20,
  "final_k": 5,
  "eval_method": "LLM-as-Judge (Groq llama-3.1-8b)",
  "llm_correctness": 0.877,
  "llm_relevance": 0.890,
  "keyword_correctness": 0.536,
  "keyword_relevance": 0.600,
  "has_sources_rate": 1.0,
  "avg_latency": 3.00
}
```

### Key Findings

| Finding | Insight |
|---------|---------|
| 2000/400 wins overall | Best relevance (89%), fastest latency (2.1s) |
| LLM scores >> Keyword scores | Keyword metrics underestimate quality by ~35% |
| 85-88% correctness across configs | All chunk sizes produce accurate answers |
| 100% source grounding | All answers cite retrieved context |
| Larger chunks = faster | 2000/400 is 3x faster than 500/100 |
| Sweet spot: 2000/400 | Best balance of quality + speed |
| Reranking: minimal impact | +0.88s latency, no quality gain on this dataset |

<details>
<summary><strong>Why Reranking Didn't Help Much</strong></summary>

| Reason | Explanation |
|--------|-------------|
| Large chunks (2000) | Already capture full context, less retrieval noise |
| Simple Q&A dataset | Questions are straightforward, top-5 already accurate |
| Small corpus | 40-page PDF doesn't have many confusable chunks |

> **Note**: Reranking typically helps more with smaller chunks, complex queries, or larger corpora.

</details>

### Evaluation Methods Comparison

| Method | How it works | Pros | Cons |
|--------|--------------|------|------|
| **Keyword** | Word overlap between expected/predicted | Fast, free | Misses synonyms, underestimates |
| **LLM Judge** | LLM scores semantic similarity | Accurate, understands meaning | Extra API calls, slight bias |

### RAG Stack
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
- **Vector Store**: FAISS (in-memory)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (optional)
- **LLM**: Groq `meta-llama/llama-4-scout-17b-16e-instruct`
- **Eval LLM**: Groq `llama-3.1-8b-instant`
- **Retrieval**: Top-20 → Rerank → Top-5

<details>
<summary><strong>Roadmap</strong></summary>

| Week | Change | Status |
|------|--------|--------|
| 1 | Baseline v1 + eval | Done |
| 2 | Chunking ablation + LLM eval | Done |
| 3 | Reranking | Done (minimal impact) |
| 4 | Hybrid retrieval (BM25 + vector) | Next |
| 5 | RRF fusion | Planned |
| 6 | Vector DB swap (Qdrant) | Planned |
| 7 | Query rewriting / HyDE | Planned |
| 8 | Final report + ablation table | Planned |

</details>

## Notes
- `.env` is ignored by git (see root `.gitignore`).
- Embeddings preload on server start for faster indexing after the first request.
- Run `python evaluate_local.py` in `backend/` to reproduce evaluation results.
- LLM-as-Judge uses a different model (`llama-3.1-8b`) than RAG to avoid self-bias.
- **Conclusion**: 2000/400 chunks provide best quality + speed trade-off for this dataset.
