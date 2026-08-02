import json
import time
import asyncio
import logging
from typing import Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.session import AsyncSessionLocal
from backend.db.models import EvaluationRun
from backend.rag import query_document
from backend.eval.metrics import calculate_recall_at_k, calculate_mrr, calculate_citation_precision, evaluate_groundedness_heuristic

logger = logging.getLogger(__name__)

async def run_evaluation_suite(dataset_path: str = "backend/eval/dataset_v2.jsonl", tenant_id: str = "default_eval_tenant"):
    """
    Executes the full evaluation suite:
    1. Loads dataset cases.
    2. Runs RAG query for each case.
    3. Computes 7 metrics per category and overall.
    4. Persists results in Postgres `evaluation_runs`.
    """
    cases = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    total_cases = len(cases)
    logger.info("Starting Evaluation Run on %d test cases...", total_cases)

    recalls = []
    mrrs = []
    citation_precisions = []
    groundedness_scores = []
    latencies_ms = []
    failures = 0

    category_stats: Dict[str, List[float]] = {}

    for case in cases:
        start_t = time.perf_counter()
        cat = case.get("category", "general")
        if cat not in category_stats:
            category_stats[cat] = []

        try:
            res = await query_document(question=case["question"], tenant_id=tenant_id)
            elapsed_ms = (time.perf_counter() - start_t) * 1000
            latencies_ms.append(elapsed_ms)

            answer = res.get("answer", "")
            sources = res.get("sources", [])
            
            r_k = calculate_recall_at_k(sources, case.get("ground_truth_chunks", []))
            mrr = calculate_mrr(sources, case.get("ground_truth_chunks", []))
            cit_prec = calculate_citation_precision(answer, sources)
            grnd = evaluate_groundedness_heuristic(answer, " ".join(sources))

            recalls.append(r_k)
            mrrs.append(mrr)
            citation_precisions.append(cit_prec)
            groundedness_scores.append(grnd)
            category_stats[cat].append(r_k)

        except Exception as exc:
            failures += 1
            logger.error("Eval case %s failed: %s", case.get("id"), exc)

    # Compute Summary Statistics
    latencies_ms.sort()
    p50_latency = latencies_ms[int(len(latencies_ms) * 0.50)] if latencies_ms else 0
    p95_latency = latencies_ms[int(len(latencies_ms) * 0.95)] if latencies_ms else 0

    metrics_summary = {
        "mean_recall_at_5": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        "mean_mrr": round(sum(mrrs) / len(mrrs), 4) if mrrs else 0.0,
        "citation_precision": round(sum(citation_precisions) / len(citation_precisions), 4) if citation_precisions else 0.0,
        "groundedness_score": round(sum(groundedness_scores) / len(groundedness_scores), 4) if groundedness_scores else 0.0,
        "p50_latency_ms": round(p50_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "failure_rate": round(failures / total_cases, 4) if total_cases > 0 else 0.0,
        "category_breakdown": {k: round(sum(v)/len(v), 2) for k, v in category_stats.items() if v}
    }

    # Persist metrics to PostgreSQL `evaluation_runs` table
    async with AsyncSessionLocal() as db:
        eval_run = EvaluationRun(
            run_name=f"Eval Run {time.strftime('%Y-%m-%d %H:%M:%S')}",
            dataset_version="v2.0",
            total_cases=total_cases,
            metrics=metrics_summary
        )
        db.add(eval_run)
        await db.commit()
        logger.info("Evaluation metrics successfully recorded to Postgres (ID: %s)", eval_run.id)

    print("\n" + "="*50)
    print("        DOCOPILOT EVALUATION SUMMARY REPORT      ")
    print("="*50)
    print(json.dumps(metrics_summary, indent=2))
    print("="*50 + "\n")

    return metrics_summary

if __name__ == "__main__":
    asyncio.run(run_evaluation_suite())
