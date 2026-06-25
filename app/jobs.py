"""In-memory job store + thread-safe progress events for SSE streaming."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from queue import Queue
from typing import Any

_lock = threading.Lock()
_jobs: dict[str, "Job"] = {}
_ctx = threading.local()


@dataclass
class Job:
    job_id: str
    filename: str = ""
    status: str = "queued"  # queued | running | complete | failed
    events: Queue = field(default_factory=Queue)
    result: dict | None = None
    error: str | None = None


def create_job(filename: str = "") -> str:
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = Job(job_id=job_id, filename=filename)
    return job_id


def get_job(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def set_status(job_id: str, status: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job.status = status


def bind_job(job_id: str | None) -> None:
    _ctx.job_id = job_id


def current_job_id() -> str | None:
    return getattr(_ctx, "job_id", None)


def emit(event: str, **data: Any) -> None:
    job_id = current_job_id()
    if not job_id:
        return
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return
    payload = {"event": event, "data": data, "ts": time.time()}
    job.events.put(payload)


def complete_job(job_id: str, result: dict) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job.status = "complete"
            job.result = result
    bind_job(job_id)
    emit("pipeline_complete", **result)


def fail_job(job_id: str, error: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job.status = "failed"
            job.error = error
    bind_job(job_id)
    emit("pipeline_error", message=error)
