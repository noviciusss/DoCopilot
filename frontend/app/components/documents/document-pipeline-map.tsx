"use client";

import { useReducedMotion, motion } from "motion/react";
import { Check, Loader2, AlertCircle } from "lucide-react";
import { IngestionStatus } from "../../../lib/hooks/useUpload";

interface PipelineMapProps {
  status: IngestionStatus;
  isAsking?: boolean;
  hasAnswer?: boolean;
  hasSources?: boolean;
  className?: string;
}

type StepState = "done" | "active" | "ready" | "error" | "idle";

interface StepConfig {
  id: string;
  label: string;
  description: string;
  getState: (
    status: IngestionStatus,
    isAsking?: boolean,
    hasAnswer?: boolean,
    hasSources?: boolean
  ) => StepState;
}

const STEPS: StepConfig[] = [
  {
    id: "upload",
    label: "UPLOAD",
    description: "File received",
    getState: (status) => {
      if (status === "uploading") return "active";
      if (status === "failed") return "error";
      if (["queued", "running", "succeeded"].includes(status)) return "done";
      return "idle";
    },
  },
  {
    id: "extract",
    label: "EXTRACT",
    description: "Text parsed",
    getState: (status) => {
      if (status === "queued") return "active";
      if (["running", "succeeded"].includes(status)) return "done";
      if (status === "failed") return "error";
      return "idle";
    },
  },
  {
    id: "index",
    label: "INDEX",
    description: "Qdrant vectorstore",
    getState: (status) => {
      if (status === "running") return "active";
      if (status === "succeeded") return "done";
      if (status === "failed") return "error";
      return "idle";
    },
  },
  {
    id: "retrieve",
    label: "RETRIEVE",
    description: "Context search",
    getState: (status, isAsking, hasAnswer, hasSources) => {
      if (status !== "succeeded") return "idle";
      if (isAsking) return "active";
      if (hasAnswer || hasSources) return "done";
      return "ready";
    },
  },
  {
    id: "answer",
    label: "ANSWER",
    description: "Grounded output",
    getState: (status, isAsking, hasAnswer) => {
      if (status !== "succeeded") return "idle";
      if (isAsking) return "active";
      if (hasAnswer) return "done";
      return "ready";
    },
  },
];

export default function DocumentPipelineMap({
  status,
  isAsking = false,
  hasAnswer = false,
  hasSources = false,
  className = "",
}: PipelineMapProps) {
  const reduced = useReducedMotion() ?? false;

  return (
    <div
      className={`rounded-xl p-4 space-y-3 ${className}`}
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
      }}
      aria-label="Document pipeline system map"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className="w-1.5 h-1.5 rounded-full"
            style={{
              background:
                status === "succeeded"
                  ? "var(--green)"
                  : status === "failed"
                  ? "var(--red)"
                  : ["uploading", "queued", "running"].includes(status) || isAsking
                  ? "var(--cobalt)"
                  : "var(--text-3)",
            }}
            aria-hidden="true"
          />
          <h3
            className="text-[10px] font-mono tracking-wider uppercase font-semibold"
            style={{ color: "var(--text-2)" }}
          >
            DOCUMENT PIPELINE
          </h3>
        </div>
        <span
          className="text-[10px] font-mono"
          style={{
            color:
              status === "succeeded"
                ? "var(--green)"
                : status === "failed"
                ? "var(--red)"
                : ["uploading", "queued", "running"].includes(status) || isAsking
                ? "var(--cobalt)"
                : "var(--text-3)",
          }}
        >
          {status === "succeeded"
            ? isAsking
              ? "RETRIEVING CONTEXT…"
              : hasAnswer
              ? "GROUNDED ANSWER READY"
              : "INDEXED & READY"
            : status === "running"
            ? "INDEXING VECTORS…"
            : status === "queued"
            ? "EXTRACTING TEXT…"
            : status === "uploading"
            ? "UPLOADING BYTES…"
            : status === "failed"
            ? "INGESTION ERROR"
            : "AWAITING INPUT"}
        </span>
      </div>

      {/* Horizontal pipeline track */}
      <div className="flex items-center justify-between gap-1 sm:gap-2 overflow-x-auto py-1">
        {STEPS.map((step, idx) => {
          const state = step.getState(status, isAsking, hasAnswer, hasSources);

          return (
            <div key={step.id} className="flex items-center flex-1 min-w-[58px]">
              {/* Step Node */}
              <div className="flex flex-col items-center text-center flex-1">
                <div
                  className="w-6 h-6 rounded-full flex items-center justify-center transition-colors mb-1.5 flex-shrink-0"
                  style={{
                    background:
                      state === "done"
                        ? "var(--green-dim)"
                        : state === "active"
                        ? "var(--cobalt-dim)"
                        : state === "error"
                        ? "var(--red-dim)"
                        : "var(--surface-2)",
                    border: `1px solid ${
                      state === "done"
                        ? "var(--green)"
                        : state === "active"
                        ? "var(--cobalt)"
                        : state === "error"
                        ? "var(--red)"
                        : "var(--border)"
                    }`,
                    color:
                      state === "done"
                        ? "var(--green)"
                        : state === "active"
                        ? "var(--cobalt)"
                        : state === "error"
                        ? "var(--red)"
                        : "var(--text-3)",
                  }}
                  title={`${step.label}: ${step.description}`}
                >
                  {state === "done" ? (
                    <Check size={11} strokeWidth={2.5} aria-hidden="true" />
                  ) : state === "active" ? (
                    <Loader2 size={11} className="animate-spin" aria-hidden="true" />
                  ) : state === "error" ? (
                    <AlertCircle size={11} aria-hidden="true" />
                  ) : (
                    <span
                      className="w-1.5 h-1.5 rounded-full"
                      style={{
                        background:
                          state === "ready"
                            ? "var(--text-2)"
                            : "var(--border-light)",
                      }}
                    />
                  )}
                </div>

                <span
                  className="text-[9px] sm:text-[10px] font-mono tracking-wider font-semibold truncate"
                  style={{
                    color:
                      state === "done"
                        ? "var(--green)"
                        : state === "active"
                        ? "var(--cobalt)"
                        : state === "error"
                        ? "var(--red)"
                        : state === "ready"
                        ? "var(--text-1)"
                        : "var(--text-3)",
                  }}
                >
                  {step.label}
                </span>

                <span
                  className="text-[8px] sm:text-[9px] truncate max-w-[65px] hidden sm:block mt-0.5"
                  style={{ color: "var(--text-3)" }}
                >
                  {step.description}
                </span>
              </div>

              {/* Connector line between steps */}
              {idx < STEPS.length - 1 && (
                <div className="flex-1 px-1 flex items-center -mt-3 sm:-mt-4">
                  <svg
                    viewBox="0 0 40 4"
                    className="w-full h-1"
                    preserveAspectRatio="none"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <motion.line
                      x1="0"
                      y1="2"
                      x2="40"
                      y2="2"
                      stroke={
                        state === "done"
                          ? "var(--green)"
                          : state === "active"
                          ? "var(--cobalt)"
                          : "var(--border)"
                      }
                      strokeWidth="1"
                      strokeDasharray={state === "done" || state === "active" ? "none" : "2 2"}
                      strokeOpacity={state === "done" || state === "active" ? 0.7 : 0.4}
                      initial={reduced ? false : { pathLength: 0 }}
                      animate={{ pathLength: 1 }}
                      transition={
                        reduced
                          ? { duration: 0 }
                          : { duration: 0.4, delay: idx * 0.08, ease: [0.16, 1, 0.3, 1] }
                      }
                    />
                  </svg>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
