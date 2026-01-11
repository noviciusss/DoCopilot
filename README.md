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

### Ablation Study - Chunk Size Comparison

| Config | Chunk Size | Overlap | Correctness | Relevance | Sources | Latency |
|--------|------------|---------|-------------|-----------|---------|---------|
| **Small** | 500 | 100 | 47.4% | 35.5% | 100% | 8.1s |
| **Medium** | 1000 | 200 | 48.0% | 34.4% | 100% | 15.2s |
| **Large (Baseline)** | 2000 | 400 | 50.3% | 37.2% | 100% | 17.8s |

### Detailed Results

#### Config: 500 chunk / 100 overlap
```json
{
  "chunk_size": 500,
  "chunk_overlap": 100,
  "avg_correctness": 0.474,
  "avg_relevance": 0.355,
  "has_sources_rate": 1.0,
  "avg_latency": 8.09
}
```

#### Config: 1000 chunk / 200 overlap
```json
{
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "avg_correctness": 0.480,
  "avg_relevance": 0.344,
  "has_sources_rate": 1.0,
  "avg_latency": 15.2
}
```

#### Config: 2000 chunk / 400 overlap (Baseline)
```json
{
  "chunk_size": 2000,
  "chunk_overlap": 400,
  "avg_correctness": 0.503,
  "avg_relevance": 0.372,
  "has_sources_rate": 1.0,
  "avg_latency": 17.8
}
```

### Key Findings

| Finding | Insight |
|---------|---------|
| ✅ **Sources 100%** | RAG properly grounds all answers in retrieved context |
| 📊 **Larger chunks = better quality** | 2000/400 gives +2.9% correctness over 500/100 |
| ⏱️ **Smaller chunks = faster** | 500/100 is 2x faster than 2000/400 |
| ⚖️ **Trade-off** | Choose based on latency vs quality requirements |

### RAG Stack
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
- **Vector Store**: FAISS (in-memory)
- **LLM**: Groq `llama-3.3-70b-versatile`
- **Retrieval**: Top-5 similarity search

## Roadmap

| Week | Change | Status |
|------|--------|--------|
| 1 | Baseline v1 + eval | ✅ Done |
| 2 | Chunking ablation | ✅ Done |
| 3 | Reranking | 🔜 Next |
| 4 | Hybrid retrieval (BM25 + vector) | Planned |
| 5 | RRF fusion | Planned |
| 6 | Vector DB swap (Qdrant) | Planned |
| 7 | Query rewriting / HyDE | Planned |
| 8 | Final report + ablation table | Planned |

## Notes
- `.env` is ignored by git (see root `.gitignore`).
- Embeddings preload on server start for faster indexing after the first request.
- Run `python evaluate_local.py` in `backend/` to reproduce evaluation results.
