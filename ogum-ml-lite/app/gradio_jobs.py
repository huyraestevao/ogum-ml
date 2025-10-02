"""Gradio tab helpers for job monitoring."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import gradio as gr
import pandas as pd
from ogum_lite.jobs import DEFAULT_JOBS_PATH, JobsDB
from ogum_lite.scheduler import cancel_job


def _db() -> JobsDB:
    return JobsDB(DEFAULT_JOBS_PATH)


def _format_ts(value: datetime | None) -> str:
    if value is None:
        return "—"
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _jobs_table() -> pd.DataFrame:
    db = _db()
    rows = []
    for job in db.list():
        rows.append(
            {
                "id": job.id,
                "status": job.status,
                "created_at": _format_ts(job.created_at),
                "started_at": _format_ts(job.started_at),
                "ended_at": _format_ts(job.ended_at),
                "command": " ".join(job.cmd),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "status",
                "created_at",
                "started_at",
                "ended_at",
                "command",
            ]
        )
    return pd.DataFrame(rows)


def _read_log(job_id: str) -> str:
    if not job_id:
        return "Informe um Job ID para visualizar o log."
    db = _db()
    try:
        job = db.get(job_id)
    except KeyError:
        return f"Job {job_id} não encontrado."
    log_path = Path(job.log_file or db.job_directory(job.id) / "job.log")
    if not log_path.exists():
        return "Log ainda não disponível."
    return log_path.read_text(encoding="utf-8")


def _cancel(job_id: str) -> Tuple[str, pd.DataFrame]:
    if not job_id:
        return "Informe um Job ID para cancelar.", _jobs_table()
    db = _db()
    if cancel_job(job_id, db):
        return f"Job {job_id} cancelado.", _jobs_table()
    return f"Job {job_id} não está em execução.", _jobs_table()


def render_jobs_tab() -> None:
    """Render the Jobs tab inside an existing ``gr.Blocks`` context."""

    with gr.Tab("Jobs"):
        refresh_btn = gr.Button("Listar Jobs")
        jobs_df = gr.Dataframe(value=_jobs_table(), label="Jobs", interactive=False)
        job_id = gr.Textbox(label="Job ID", placeholder="Cole o identificador do job")
        log_btn = gr.Button("Ver Log")
        cancel_btn = gr.Button("Cancelar Job")
        log_output = gr.Textbox(label="Log", lines=12)

        refresh_btn.click(_jobs_table, outputs=jobs_df)
        log_btn.click(_read_log, inputs=job_id, outputs=log_output)
        cancel_btn.click(_cancel, inputs=job_id, outputs=[log_output, jobs_df])
