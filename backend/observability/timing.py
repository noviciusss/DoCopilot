import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

logger = logging.getLogger(__name__)

@asynccontextmanager
async def timed_stage(stage_name: str, request_id: Optional[str] = None, tenant_id: Optional[str] = None):
    """
    Async context manager that times a named pipeline stage and logs the result.
    
    Usage:
        async with timed_stage("hybrid_search", request_id=rid, tenant_id=tid):
            results = await hybrid_search(query)
    
    Produces a structured log like:
        {"stage": "hybrid_search", "latency_ms": 342.1, "request_id": "...", "tenant_id": "..."}
    
    Why context manager: Clean, non-invasive. No need to manually call start/stop
    timers throughout the code. Works with exceptions too (finally block).
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "Stage completed: %s in %sms",
            stage_name,
            elapsed_ms,
            extra={
                "stage": stage_name,
                "latency_ms": elapsed_ms,
                "request_id": request_id or "unknown",
                "tenant_id": tenant_id or "unknown",
            }
        )


def estimate_token_cost(input_tokens: int, output_tokens: int, model: str = "qwen/qwen3-32b") -> float:
    """
    Rough token cost estimation for logging purposes.
    Groq pricing as of 2025 (approximate — update as needed).
    
    Why log costs: Lets you calculate per-tenant and per-query cost, enabling
    cost allocation reporting for enterprise clients or internal budgeting.
    """
    rates = {
        "qwen/qwen3-32b": {"input": 0.29, "output": 0.39},        # per 1M tokens
        "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
        "gemma2-9b-it": {"input": 0.20, "output": 0.20},
    }
    rate = rates.get(model, {"input": 0.30, "output": 0.40})
    cost = (input_tokens * rate["input"] + output_tokens * rate["output"]) / 1_000_000
    return round(cost, 8)
