"use client";

import { CheckCircle2, Loader2, XCircle, Upload, Layers } from "lucide-react";
import { IngestionStatus } from "../../../lib/hooks/useUpload";

interface Props {
  status: IngestionStatus;
  error: string | null;
}

const STEPS: { key: IngestionStatus[]; label: string }[] = [
  { key: ["uploading"],                         label: "Uploading" },
  { key: ["queued"],                            label: "Queued" },
  { key: ["running"],                           label: "Extracting & embedding" },
  { key: ["succeeded", "failed"],               label: "Indexed" },
];

function stepState(
  stepKeys: IngestionStatus[],
  current: IngestionStatus
): "done" | "active" | "error" | "idle" {
  const order: IngestionStatus[] = ["idle", "uploading", "queued", "running", "succeeded", "failed"];
  const currentIdx = order.indexOf(current);

  if (current === "failed" && stepKeys.includes("failed")) return "error";
  if (current === "succeeded" && stepKeys.includes("succeeded")) return "done";

  const stepMaxIdx = Math.max(...stepKeys.map((k) => order.indexOf(k)));
  if (currentIdx > stepMaxIdx) return "done";
  if (stepKeys.includes(current)) return "active";
  return "idle";
}

export default function UploadStatus({ status, error }: Props) {
  if (status === "idle") return null;

  return (
    <div
      className="rounded-lg p-3 space-y-2.5"
      style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
      role="status"
      aria-live="polite"
      aria-label={`Document processing: ${status}`}
    >
      <div className="flex items-center gap-2 mb-1">
        <Layers size={12} style={{ color: "var(--text-2)" }} aria-hidden="true" />
        <span className="text-label">Processing</span>
      </div>

      <div className="space-y-1.5">
        {STEPS.map((step, i) => {
          const state = stepState(step.key, status);
          return (
            <div key={i} className="flex items-center gap-2">
              <div className="w-4 h-4 flex items-center justify-center flex-shrink-0">
                {state === "done" && (
                  <CheckCircle2 size={13} style={{ color: "var(--green)" }} aria-hidden="true" />
                )}
                {state === "active" && (
                  <Loader2 size={13} className="animate-spin" style={{ color: "var(--cobalt)" }} aria-hidden="true" />
                )}
                {state === "error" && (
                  <XCircle size={13} style={{ color: "var(--red)" }} aria-hidden="true" />
                )}
                {state === "idle" && (
                  <div
                    className="w-1.5 h-1.5 rounded-full"
                    style={{ background: "var(--border-light)" }}
                    aria-hidden="true"
                  />
                )}
              </div>
              <span
                className="text-xs"
                style={{
                  color:
                    state === "done"   ? "var(--green)"  :
                    state === "active" ? "var(--text-1)" :
                    state === "error"  ? "var(--red)"    :
                    "var(--text-3)",
                }}
              >
                {step.label}
                {state === "active" && step.key.includes("running") && "…"}
              </span>
            </div>
          );
        })}
      </div>

      {status === "succeeded" && (
        <p className="text-xs pt-1" style={{ color: "var(--green)" }}>
          <span className="flex items-center gap-1.5">
            <Upload size={11} aria-hidden="true" />
            Document indexed — ready for questions
          </span>
        </p>
      )}

      {status === "failed" && error && (
        <p
          className="text-xs pt-1 leading-snug"
          style={{ color: "var(--red)" }}
        >
          {error}
        </p>
      )}
    </div>
  );
}
