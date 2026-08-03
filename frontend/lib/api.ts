const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("docopilot_token");
}

function authHeaders(extra: Record<string, string> = {}): HeadersInit {
  const token = getToken();
  return {
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...extra,
  };
}

// ── Auth ──────────────────────────────────────────────────────────────────────
export async function apiRegister(email: string, password: string) {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail ?? "Registration failed");
  return data as { access_token: string; token_type: string };
}

export async function apiLogin(email: string, password: string) {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail ?? "Login failed");
  return data as { access_token: string; token_type: string };
}

// ── Upload (async, returns job_id) ────────────────────────────────────────────
export async function apiUpload(formData: FormData) {
  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    headers: authHeaders(), // No Content-Type — browser sets multipart boundary
    body: formData,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail ?? "Upload failed");
  return data as { job_id: string; document_id: string; status: string; message: string };
}

// ── Job polling ───────────────────────────────────────────────────────────────
export async function apiGetJobStatus(jobId: string) {
  const res = await fetch(`${API_BASE}/ingestion/jobs/${jobId}`, {
    headers: authHeaders(),
  });
  const data = await res.json();
  if (!res.ok) throw new Error("Job status check failed");
  return data as { status: string; failure_reason?: string };
}

// ── Document Library ──────────────────────────────────────────────────────────
export type DocumentLibraryItem = {
  id: string;
  filename: string;
  file_size_bytes: number;
  mime_type: string;
  created_at: string;
  ingestion_status: string;
  qdrant_collection: string | null;
};

export async function apiGetDocuments(myDocs = true): Promise<DocumentLibraryItem[]> {
  const res = await fetch(`${API_BASE}/documents?my_docs=${myDocs}`, {
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error("Failed to fetch document library");
  return res.json();
}

export async function apiDeleteDocument(docId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/documents/${docId}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error("Failed to delete document");
}

// ── Chat (SSE stream via fetch, supports Bearer header) ───────────────────────
// Note: Native EventSource does NOT support custom headers, so we use fetch + ReadableStream.
export function apiStreamChat(
  question: string,
  documentId: string,
  onToken: (token: string) => void,
  onDone: (sources: string[], fullAnswer: string) => void,
  onError: (err: string) => void
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ question, document_id: documentId }),
        signal: controller.signal,
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        onError(data.detail ?? "Chat request failed");
        return;
      }

      if (!res.body) { onError("No response body"); return; }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let fullAnswer = "";

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
            if (ev.token) { fullAnswer += ev.token; onToken(ev.token); }
            if (ev.done)  { onDone(ev.sources ?? [], ev.answer ?? fullAnswer); }
            if (ev.error) { onError(ev.error); }
          } catch { /* malformed SSE line — ignore */ }
        }
      }
    } catch (err: unknown) {
      const e = err as Error;
      if (e.name !== "AbortError") onError(e.message ?? "Stream error");
    }
  })();

  return () => controller.abort();
}
