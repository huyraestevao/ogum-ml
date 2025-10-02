"""Prep and validation Streamlit page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ..i18n.translate import I18N
from ..services import run_cli, state, validators


def _list_csv(workspace_path: Path) -> list[Path]:
    uploads = workspace_path / "uploads"
    if not uploads.exists():
        return []
    return sorted(uploads.glob("*.csv"))


def render(translator: I18N) -> None:
    """Render the preparation workflow."""

    st.subheader(translator.t("menu.prep"))
    workspace = state.get_workspace()
    preset = state.get_preset()

    uploaded = st.file_uploader("CSV", type=["csv"], key="prep_upload")
    if uploaded is not None:
        target = state.persist_upload(uploaded)
        st.toast(f"[ok] {uploaded.name} → {target.name}")

    csv_files = _list_csv(workspace.path)
    if not csv_files:
        st.info(translator.t("messages.no_artifacts"))
        return

    option = st.selectbox(
        "Dataset",
        options=[str(path) for path in csv_files],
        format_func=lambda value: Path(value).name,
        key="prep_dataset",
    )
    if not option:
        return

    selected = Path(option)
    cols = st.columns(2)
    if cols[0].button(translator.t("actions.validate"), key="prep_validate"):
        summary = validators.validate_long(selected)
        if summary.ok:
            st.toast(translator.t("messages.validation_ok"))
        for issue in summary.issues:
            st.warning(issue)

    if cols[1].button(translator.t("actions.run"), key="prep_run"):
        with st.spinner("Rodando preprocess..."):
            result = run_cli.run_prep(selected, preset, workspace)
        prep_csv = result.outputs["prep_csv"]
        state.register_artifact("prep_csv", prep_csv, description="prep")
        st.toast(translator.t("messages.ready"))

    preview_target = state.get_artifact("prep_csv") or selected
    if preview_target.exists():
        st.caption(f"Preview · {preview_target.name}")
        st.dataframe(pd.read_csv(preview_target).head(30))
