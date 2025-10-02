"""Prep and validation Streamlit page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ..design.a11y import aria_label, focus_hint
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
    st.caption(str(focus_hint(translator.t("wizard.focus_hint"))))
    workspace = state.get_workspace()
    preset = state.get_preset()

    uploaded = st.file_uploader(
        translator.t("microcopy.upload_csv"),
        type=["csv"],
        key="prep_upload",
        **aria_label(translator.t("microcopy.upload_csv")),
    )
    if uploaded is not None:
        target = state.persist_upload(uploaded)
        st.toast(f"[ok] {uploaded.name} → {target.name}")

    csv_files = _list_csv(workspace.path)
    if not csv_files:
        st.info(translator.t("microcopy.missing_artifact"))
        return

    option = st.selectbox(
        translator.t("microcopy.select_csv"),
        options=[str(path) for path in csv_files],
        format_func=lambda value: Path(value).name,
        key="prep_dataset",
    )
    if not option:
        return

    selected = Path(option)
    cols = st.columns(2)
    if cols[0].button(
        translator.t("microcopy.validate_data"),
        key="prep_validate",
        **aria_label(translator.t("microcopy.validate_data")),
    ):
        summary = validators.validate_long(selected)
        if summary.ok:
            st.toast(translator.t("microcopy.validation_ok"))
        else:
            st.toast(translator.t("microcopy.validation_warn"))
        for issue in summary.issues:
            st.warning(issue)

    if cols[1].button(
        translator.t("microcopy.run_prep"),
        key="prep_run",
        **aria_label(translator.t("microcopy.run_prep")),
    ):
        with st.spinner(translator.t("microcopy.spinner_prep")):
            result = run_cli.run_prep(selected, preset, workspace)
        prep_csv = result.outputs["prep_csv"]
        state.register_artifact("prep_csv", prep_csv, description="prep")
        st.toast(translator.t("microcopy.prep_ready"))

    preview_target = state.get_artifact("prep_csv") or selected
    if preview_target.exists():
        st.caption(f"Preview · {preview_target.name}")
        st.dataframe(pd.read_csv(preview_target).head(30))
