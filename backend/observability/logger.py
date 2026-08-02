import json
import logging
import sys
from datetime import datetime, timezone

class JSONFormatter(logging.Formatter):
    """
    Emits every log record as a single JSON line.
    
    Why: Machine-readable structured logs can be queried, filtered, and
    aggregated by log management systems (Azure Monitor, Datadog, Loki).
    Plain text logs require fragile regex parsing to extract fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Base structured log payload — always present on every line
        log_dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Pull extra context fields attached via logger.info(..., extra={...})
        for key in ("request_id", "tenant_id", "user_id", "endpoint", "latency_ms", "token_count", "estimated_cost_usd", "stage"):
            if hasattr(record, key):
                log_dict[key] = getattr(record, key)

        # Attach exception info if present (for ERROR/CRITICAL)
        if record.exc_info:
            log_dict["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(log_dict, default=str)


def setup_structured_logging(level: str = "INFO") -> None:
    """
    Call once at app startup (in main.py's lifespan or module-level).
    Replaces all existing handlers with a JSON-emitting stdout handler.
    
    Why stdout: Azure Container Apps / Docker automatically collect stdout
    and forward to Azure Monitor Log Analytics — no log file management needed.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove any existing handlers (uvicorn installs its own)
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root_logger.addHandler(handler)
