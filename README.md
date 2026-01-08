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

## Notes
- `.env` is ignored by git (see root `.gitignore`).
- Embeddings preload on server start for faster indexing after the first request.
