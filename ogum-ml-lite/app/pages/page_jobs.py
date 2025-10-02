"""Streamlit monitor for background jobs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from ogum_lite.jobs import DEFAULT_JOBS_PATH, JobsDB
from ogum_lite.scheduler import cancel_job

from ..i18n.translate import I18N

try:  # pragma: no cover - optional dependency
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # pragma: no cover - fallback for tests

    def st_autorefresh(*_: object, **__: object) -> None:
        return None


def _format_ts(value: datetime | None) -> str:
    if value is None:
        return "—"
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _log_tail(path: Path, *, limit: int = 4000) -> str:
    if not path.exists():
        return "Log ainda não disponível."
    text = path.read_text(encoding="utf-8")
    if len(text) <= limit:
        return text
    return text[-limit:]


def render(_: I18N) -> None:
    """Render the Jobs Monitor page."""

    st.subheader("Jobs Monitor")
    st.caption("Agende, acompanhe e cancele execuções em segundo plano.")
    st_autorefresh(interval=5000, key="jobs-monitor-autorefresh")

    db = JobsDB(DEFAULT_JOBS_PATH)
    jobs = list(reversed(db.list()))
    if not jobs:
        st.info("Nenhum job encontrado. Submeta um job pela CLI ou UI.")
        return

    for job in jobs:
        job_dir = db.job_directory(job.id)
        log_path = Path(job.log_file or job_dir / "job.log")
        status = job.status.capitalize()
        with st.container(border=True):
            st.markdown(f"**{job.id}** — {status}")
            st.code(" ".join(job.cmd), language="bash")
            cols = st.columns([1, 1, 1])
            cols[0].markdown(f"**Criado:** {_format_ts(job.created_at)}")
            cols[1].markdown(f"**Iniciado:** {_format_ts(job.started_at)}")
            cols[2].markdown(f"**Finalizado:** {_format_ts(job.ended_at)}")

            with st.expander("Ver log"):
                st.text(_log_tail(log_path))

            if job.status == "running":
                if st.button("Cancelar job", key=f"cancel-{job.id}"):
                    cancel_job(job.id, db)
                    st.experimental_rerun()
