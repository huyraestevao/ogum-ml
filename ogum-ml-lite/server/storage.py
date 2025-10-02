"""Utility helpers for the collaborative storage layer."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import IO, Iterable

from .models import JobInfo, RunInfo, SimulationBundle

DEFAULT_SUBDIRS = ("runs", "jobs", "sims", "users")


def get_storage_base() -> Path:
    """Return the root directory for collaborative storage."""

    env_value = os.environ.get("OGUM_STORAGE_PATH")
    if env_value:
        base = Path(env_value).expanduser()
    else:
        base = Path(__file__).resolve().parent.parent / "server_storage"
    base.mkdir(parents=True, exist_ok=True)
    for sub in DEFAULT_SUBDIRS:
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


def get_storage_path(name: str) -> Path:
    """Return a subdirectory inside the storage base and ensure it exists."""

    base = get_storage_base()
    path = base / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path, default: dict | list | None = None):
    if not path.exists():
        return default if default is not None else {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data) -> None:  # type: ignore[no-untyped-def]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=_json_default)


def _json_default(value):  # type: ignore[no-untyped-def]
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def list_runs() -> list[RunInfo]:
    """Return metadata about the available runs."""

    runs_dir = get_storage_path("runs")
    runs: list[RunInfo] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "metadata.json"
        metadata = load_json(meta_path, default={}) or {}
        created = metadata.get("created_at")
        if created:
            created_at = datetime.fromisoformat(created)
        else:
            created_at = datetime.fromtimestamp(run_dir.stat().st_mtime)
        runs.append(
            RunInfo(
                name=metadata.get("name", run_dir.name),
                description=metadata.get("description"),
                created_at=created_at,
            )
        )
    return runs


def _job_from_dict(job_id: str, payload: dict) -> JobInfo:
    created_at = payload.get("created_at")
    updated_at = payload.get("updated_at")
    created = datetime.fromisoformat(created_at) if created_at else datetime.utcnow()
    updated = datetime.fromisoformat(updated_at) if updated_at else created
    return JobInfo(
        id=job_id,
        status=payload.get("status", "queued"),
        message=payload.get("message"),
        created_at=created,
        updated_at=updated,
    )


def list_jobs() -> list[JobInfo]:
    jobs_dir = get_storage_path("jobs")
    jobs: list[JobInfo] = []
    for job_file in sorted(jobs_dir.glob("*.json")):
        payload = load_json(job_file, default={}) or {}
        jobs.append(_job_from_dict(job_file.stem, payload))
    return jobs


def upsert_job(job: JobInfo) -> None:
    jobs_dir = get_storage_path("jobs")
    payload = {
        "status": job.status,
        "message": job.message,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
    }
    save_json(jobs_dir / f"{job.id}.json", payload)


def register_simulation(bundle: SimulationBundle) -> Path:
    sims_dir = get_storage_path("sims")
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    artifact_path = sims_dir / f"{bundle.name}-{timestamp}.json"
    save_json(
        artifact_path,
        {
            "name": bundle.name,
            "description": bundle.description,
            "dataset": bundle.dataset,
            "metadata": bundle.metadata or {},
            "created_at": datetime.utcnow().isoformat(),
        },
    )
    return artifact_path


def ensure_demo_run(name: str = "demo-run") -> None:
    """Create a simple run entry used by smoke tests if missing."""

    runs_dir = get_storage_path("runs")
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        save_json(
            meta_path,
            {
                "name": name,
                "description": "Example run generated for smoke tests.",
                "created_at": datetime.utcnow().isoformat(),
            },
        )


def save_run_artifact(
    filename: str, data_stream: IO[bytes], description: str | None = None
) -> RunInfo:
    """Persist an uploaded run artifact to the storage area."""

    runs_dir = get_storage_path("runs")
    sanitized = Path(filename).stem or "uploaded-run"
    run_dir = runs_dir / sanitized
    run_dir.mkdir(parents=True, exist_ok=True)
    target_file = run_dir / Path(filename).name
    if hasattr(data_stream, "seek"):
        data_stream.seek(0)
    with target_file.open("wb") as handle:
        handle.write(data_stream.read())
    metadata = {
        "name": sanitized,
        "description": description,
        "created_at": datetime.utcnow().isoformat(),
        "artifact": target_file.name,
    }
    save_json(run_dir / "metadata.json", metadata)
    return RunInfo(
        name=sanitized,
        description=description,
        created_at=datetime.fromisoformat(metadata["created_at"]),
    )


def iter_users() -> Iterable[dict]:
    users_file = get_storage_path("users") / "users.json"
    users = load_json(users_file, default=[])
    if not isinstance(users, list):
        return []
    return users


def save_users(users: list[dict]) -> None:
    users_file = get_storage_path("users") / "users.json"
    save_json(users_file, users)
