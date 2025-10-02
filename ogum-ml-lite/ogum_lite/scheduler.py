"""Background job execution helpers."""

from __future__ import annotations

import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from .jobs import Job, JobsDB

_LOG_FILE_NAME = "job.log"

_RUNNING_PROCESSES: Dict[str, subprocess.Popen] = {}
_SCHEDULED_TIMERS: Dict[str, threading.Timer] = {}
_ACTIVE_THREADS: Dict[str, threading.Thread] = {}
_CANCELLED_JOBS: set[str] = set()
_STATE_LOCK = threading.Lock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_paths(job: Job, db: JobsDB) -> tuple[Path, Path]:
    job_dir = db.job_directory(job.id)
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(job.log_file) if job.log_file else job_dir / _LOG_FILE_NAME
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return job_dir, log_path


def run_job(job: Job, db: JobsDB) -> None:
    """Execute ``job`` synchronously, updating its state in ``db``."""

    job_dir, log_path = _ensure_paths(job, db)
    started_at = _now()
    db.update_status(
        job.id,
        "running",
        started_at=started_at,
        log_file=str(log_path),
        outdir=str(job_dir),
    )

    try:
        with log_path.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(
                job.cmd,
                stdout=handle,
                stderr=handle,
                text=True,
            )
            with _STATE_LOCK:
                _RUNNING_PROCESSES[job.id] = process
            exit_code = process.wait()
    except Exception:
        with _STATE_LOCK:
            _RUNNING_PROCESSES.pop(job.id, None)
        ended_at = _now()
        db.update_status(job.id, "error", ended_at=ended_at, exit_code=None)
        raise

    with _STATE_LOCK:
        _RUNNING_PROCESSES.pop(job.id, None)
    ended_at = _now()
    status = "done" if exit_code == 0 else "error"
    if job.id in _CANCELLED_JOBS:
        status = "cancelled"
        _CANCELLED_JOBS.discard(job.id)
    db.update_status(job.id, status, ended_at=ended_at, exit_code=exit_code)


def _thread_target(job: Job, db: JobsDB) -> None:
    try:
        run_job(job, db)
    finally:
        with _STATE_LOCK:
            _ACTIVE_THREADS.pop(job.id, None)


def start_job(job: Job, db: JobsDB) -> threading.Thread:
    """Start ``job`` in a dedicated thread and return it."""

    thread = threading.Thread(
        target=_thread_target,
        args=(job, db),
        name=f"ogum-job-{job.id}",
        daemon=True,
    )
    with _STATE_LOCK:
        _ACTIVE_THREADS[job.id] = thread
    thread.start()
    return thread


def schedule_job(job: Job, db: JobsDB, at: datetime) -> threading.Timer:
    """Schedule ``job`` to run in the future using :class:`threading.Timer`."""

    now = _now()
    delay = max(0.0, (at - now).total_seconds())

    def _launch() -> None:
        with _STATE_LOCK:
            _SCHEDULED_TIMERS.pop(job.id, None)
        start_job(job, db)

    timer = threading.Timer(delay, _launch)
    timer.daemon = True
    with _STATE_LOCK:
        _SCHEDULED_TIMERS[job.id] = timer
    timer.start()
    return timer


def cancel_job(job_id: str, db: JobsDB) -> bool:
    """Cancel a scheduled or running job."""

    with _STATE_LOCK:
        timer = _SCHEDULED_TIMERS.pop(job_id, None)
        process = _RUNNING_PROCESSES.get(job_id)

    if timer is not None:
        timer.cancel()
        _CANCELLED_JOBS.add(job_id)
        db.update_status(job_id, "cancelled", ended_at=_now(), exit_code=None)
        return True

    if process is not None:
        if process.poll() is None:
            _CANCELLED_JOBS.add(job_id)
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        with _STATE_LOCK:
            _RUNNING_PROCESSES.pop(job_id, None)
        db.update_status(
            job_id,
            "cancelled",
            ended_at=_now(),
            exit_code=process.returncode,
        )
        return True

    # If the job is neither scheduled nor running we fall back to persisted state.
    try:
        job = db.get(job_id)
    except KeyError:
        return False
    if job.status in {"done", "error", "cancelled"}:
        return False
    db.update_status(job_id, "cancelled", ended_at=_now(), exit_code=None)
    return True
