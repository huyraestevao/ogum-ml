"""Export workflow page."""

from __future__ import annotations

import streamlit as st

from ..i18n.translate import I18N
from ..services import run_cli, state


def render(translator: I18N) -> None:
    """Render export utilities."""

    st.subheader(translator.t("menu.export"))
    workspace = state.get_workspace()
    preset = state.get_preset()

    if st.button(translator.t("actions.export"), key="export_run"):
        with st.spinner("Gerando artefatos..."):
            result = run_cli.export_report(workspace.path, preset, workspace)
        for key, path in result.outputs.items():
            state.register_artifact(f"export_{key}", path, description="export")
        state.register_artifact("session_zip", result.outputs["zip"], description="zip")
        st.toast(translator.t("messages.ready"))

    report = state.get_artifact("export_report")
    if report and report.exists():
        st.download_button(
            label=f"XLSX · {report.name}",
            data=report.read_bytes(),
            file_name=report.name,
            key="report_dl",
        )
    zip_path = state.get_artifact("session_zip")
    if zip_path and zip_path.exists():
        st.download_button(
            label=f"ZIP · {zip_path.name}",
            data=zip_path.read_bytes(),
            file_name=zip_path.name,
            key="zip_dl",
        )
