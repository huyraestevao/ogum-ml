from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.pages import page_jobs
from ogum_lite.jobs import Job, JobsDB
from streamlit.testing.v1 import AppTest


def test_jobs_db_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "jobs.json"
    db = JobsDB(db_path)
    job = Job(cmd=["python", "-V"])
    job_dir = db.job_directory(job.id)
    job = job.model_copy(
        update={"outdir": str(job_dir), "log_file": str(job_dir / "job.log")}
    )

    db.add(job)
    stored = db.get(job.id)
    assert stored.cmd == job.cmd
    assert stored.status == "queued"

    started = datetime.now(timezone.utc)
    updated = db.update_status(job.id, "running", started_at=started)
    assert updated.status == "running"
    assert updated.started_at is not None

    listed = db.list()
    assert len(listed) == 1
    assert listed[0].id == job.id


def test_jobs_page_renders_without_jobs(tmp_path: Path, monkeypatch) -> None:
    jobs_path = tmp_path / "jobs.json"
    monkeypatch.setattr(page_jobs, "DEFAULT_JOBS_PATH", jobs_path)

    def app() -> None:
        from app.i18n.translate import I18N as PageI18N
        from app.pages import page_jobs as jobs_page

        jobs_page.render(PageI18N())

    test = AppTest.from_function(app)
    result = test.run()
    assert not result.exception
