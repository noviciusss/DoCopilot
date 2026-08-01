import os
from celery import Celery


REDIS_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
REDIS_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
celery_app = Celery(
    "docopilot_ingestion",
    broker=REDIS_BROKER_URL,
    backend=REDIS_RESULT_BACKEND,
    include=["backend.ingestion.tasks"]
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,       # Hard limit: 10 mins per document
    task_soft_time_limit=540,  # Soft limit: 9 mins
)