"""Data models and persistence helpers for asynchronous jobs."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, Field

JobStatus = Literal["queued", "running", "done", "error", "cancelled"]


class Job(BaseModel):
    """Representation of a scheduled job."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    cmd: list[str]
    status: JobStatus = "queued"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    ended_at: datetime | None = None
    exit_code: int | None = None
    log_file: str | None = None
    outdir: str | None = None


class JobsDB:
    """Simple JSON-backed persistence for :class:`Job` records."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        if not self.path.is_absolute():
            self.path = Path.cwd() / self.path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.path.exists():
            self._write([])

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        raw = self.path.read_text(encoding="utf-8")
        if not raw.strip():
            return []
        return json.loads(raw)

    def _write(self, jobs: Iterable[dict]) -> None:
        payload = json.dumps(list(jobs), indent=2, ensure_ascii=False)
        self.path.write_text(payload, encoding="utf-8")

    def add(self, job: Job) -> Job:
        """Persist a new job."""

        with self._lock:
            records = self._read()
            records.append(job.model_dump(mode="json"))
            self._write(records)
        return job

    def update_status(self, job_id: str, status: JobStatus, **updates: object) -> Job:
        """Update the status and optional metadata of a job."""

        with self._lock:
            records = self._read()
            for index, record in enumerate(records):
                if record.get("id") != job_id:
                    continue
                job = Job.model_validate(record)
                payload = job.model_copy(update={"status": status, **updates})
                records[index] = payload.model_dump(mode="json")
                self._write(records)
                return payload
        raise KeyError(f"Job '{job_id}' not found")

    def list(self) -> list[Job]:
        """Return all persisted jobs sorted by creation timestamp."""

        records = self._read()
        jobs = [Job.model_validate(item) for item in records]
        return sorted(jobs, key=lambda job: job.created_at)

    def get(self, job_id: str) -> Job:
        """Retrieve a single job by identifier."""

        for record in self._read():
            if record.get("id") == job_id:
                return Job.model_validate(record)
        raise KeyError(f"Job '{job_id}' not found")

    def job_directory(self, job_id: str) -> Path:
        """Return the workspace directory for ``job_id``."""

        return self.path.parent / job_id


DEFAULT_JOBS_PATH = Path("workspace") / "jobs" / "jobs.json"


def get_default_db() -> JobsDB:
    """Return the default jobs database instance."""

    return JobsDB(DEFAULT_JOBS_PATH)
