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

Evaluated on **40 Q&A pairs** from the AWS Overview whitepaper using a basic RAG pipeline:

| Metric | Score | Notes |
|--------|-------|-------|
| **Success Rate** | 100% | All questions answered without errors |
| **Correctness** | 50.3% | Keyword overlap with expected answers |
| **Relevance** | 37.2% | Question-answer alignment score |
| **Source Citation** | 100% | All answers included source references |
| **Avg Latency** | 17.8s | Includes rate-limit delays |

### RAG Stack
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
- **Vector Store**: FAISS (in-memory)
- **Chunking**: 2000 chars, 400 overlap
- **LLM**: Groq `llama-3.3-70b-versatile`
- **Retrieval**: Top-5 similarity search

### Analysis
- ✅ **High source citation rate** – RAG properly grounds answers in retrieved context
- ⚠️ **Moderate correctness** – Answers are semantically correct but use different phrasing than expected
- ⚠️ **Lower relevance** – Simple keyword overlap metric underestimates actual quality
- 📈 **Room for improvement** – Better chunking, reranking, or prompt tuning could boost scores

## Notes
- `.env` is ignored by git (see root `.gitignore`).
- Embeddings preload on server start for faster indexing after the first request.
- Run `python evaluate_local.py` in `backend/` to reproduce evaluation results.
