from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ogum_lite import scheduler
from ogum_lite.jobs import Job, JobsDB


def _create_job(db: JobsDB, cmd: list[str]) -> Job:
    job = Job(cmd=cmd)
    job_dir = db.job_directory(job.id)
    job = job.model_copy(
        update={"outdir": str(job_dir), "log_file": str(job_dir / "job.log")}
    )
    db.add(job)
    return job


def test_run_job_executes_command(tmp_path: Path) -> None:
    db = JobsDB(tmp_path / "jobs.json")
    job = _create_job(db, [sys.executable, "-c", "print('hello from job')"])

    scheduler.run_job(job, db)
    stored = db.get(job.id)
    assert stored.status == "done"
    log_path = Path(stored.log_file or "")
    assert log_path.exists()
    assert "hello from job" in log_path.read_text(encoding="utf-8")


def test_schedule_job_runs_later(tmp_path: Path) -> None:
    db = JobsDB(tmp_path / "jobs.json")
    job = _create_job(db, [sys.executable, "-c", "print('delayed job')"])
    run_at = datetime.now(timezone.utc) + timedelta(milliseconds=150)

    scheduler.schedule_job(job, db, run_at)
    time.sleep(0.6)

    assert db.get(job.id).status == "done"


def test_cancel_running_job(tmp_path: Path) -> None:
    db = JobsDB(tmp_path / "jobs.json")
    job = _create_job(db, [sys.executable, "-c", "import time; time.sleep(2)"])

    thread = scheduler.start_job(job, db)
    time.sleep(0.2)
    assert scheduler.cancel_job(job.id, db)
    thread.join(timeout=3.0)

    assert db.get(job.id).status == "cancelled"
