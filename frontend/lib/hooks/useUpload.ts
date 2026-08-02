"use client";
import { useState, useRef, useCallback } from "react";
import { apiUpload, apiGetJobStatus } from "../api";

export type IngestionStatus = "idle" | "uploading" | "queued" | "running" | "succeeded" | "failed";

export const UPLOAD_STATUS_LABELS: Record<IngestionStatus, string> = {
  idle:      "Index Source",
  uploading: "Sending file…",
  queued:    "Queued — waiting for worker…",
  running:   "Processing & embedding…",
  succeeded: "Indexed — ready to chat ✓",
  failed:    "Indexing failed",
};

export function useUpload() {
  const [status, setStatus]       = useState<IngestionStatus>("idle");
  const [documentId, setDocumentId] = useState<string | null>(
    typeof window !== "undefined" ? sessionStorage.getItem("document_id") : null
  );
  const [error, setError]         = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  };

  const upload = useCallback(async (
    file: File | null,
    plainText: string | null,
    uploadType: "pdf" | "txt" | "text"
  ) => {
    setStatus("uploading");
    setError(null);
    stopPolling();

    try {
      const form = new FormData();
      if ((uploadType === "pdf" || uploadType === "txt") && file) {
        form.append(uploadType === "pdf" ? "pdf_file" : "txt_file", file);
      } else if (uploadType === "text" && plainText) {
        form.append("plain_text", plainText);
      } else {
        throw new Error("No file or text provided");
      }

      // POST /upload returns HTTP 202 immediately with job_id
      const { job_id, document_id } = await apiUpload(form);

      sessionStorage.setItem("document_id", document_id);
      setDocumentId(document_id);
      setStatus("queued");

      // If backend already returned succeeded (idempotency hit), skip polling
      // Otherwise poll every 1.5s until terminal state
      pollRef.current = setInterval(async () => {
        try {
          const job = await apiGetJobStatus(job_id);
          setStatus(job.status as IngestionStatus);
          if (job.status === "succeeded" || job.status === "failed") {
            stopPolling();
            if (job.status === "failed") {
              setError(job.failure_reason ?? "Ingestion failed — check backend logs.");
            }
          }
        } catch {
          stopPolling();
          setError("Lost connection while checking job status.");
          setStatus("failed");
        }
      }, 1500);

    } catch (e: unknown) {
      setStatus("failed");
      setError((e as Error).message);
    }
  }, []);

  const clearDocument = useCallback(() => {
    stopPolling();
    sessionStorage.removeItem("document_id");
    setDocumentId(null);
    setStatus("idle");
    setError(null);
  }, []);

  return { status, documentId, error, upload, clearDocument };
}
