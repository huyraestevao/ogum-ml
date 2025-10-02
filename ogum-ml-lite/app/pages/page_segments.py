"""Segmentation workflow page."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ..design.a11y import aria_label, describe_chart, focus_hint
from ..i18n.translate import I18N
from ..services import run_cli, state


def render(translator: I18N) -> None:
    """Render the segmentation pipeline."""

    st.subheader(translator.t("menu.segments"))
    st.caption(str(focus_hint(translator.t("wizard.focus_hint"))))
    workspace = state.get_workspace()
    preset = state.get_preset()
    prep_csv = state.get_artifact("prep_csv")
    if prep_csv is None:
        st.info(translator.t("wizard.blockers.need_prep"))
        return

    if st.button(
        translator.t("actions.run"),
        key="segmentation_run",
        **aria_label(translator.t("microcopy.run_segments")),
    ):
        with st.spinner(translator.t("microcopy.spinner_segments")):
            result = run_cli.run_segmentation(prep_csv, preset, workspace)
        segments_path = result.outputs["segments"]
        state.register_artifact("segments", segments_path, description="segments")
        st.toast(translator.t("microcopy.segments_ready"))

    segments_path = state.get_artifact("segments")
    if segments_path and segments_path.exists():
        try:
            data = json.loads(segments_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            st.code(segments_path.read_text(encoding="utf-8"), language="json")
        else:
            st.dataframe(pd.DataFrame(data))
            st.caption(
                describe_chart(translator.t("microcopy.describe_segments"), prefix="🧮")
            )
        st.download_button(
            label=f"JSON · {segments_path.name}",
            data=segments_path.read_bytes(),
            file_name=segments_path.name,
            key="segments_dl",
        )
