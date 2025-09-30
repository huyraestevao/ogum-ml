"""θ and MSC visualisation page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ..i18n.translate import I18N
from ..services import run_cli, state


def render(translator: I18N) -> None:
    """Render θ/MSC orchestration."""

    st.subheader(translator.t("menu.msc"))
    workspace = state.get_workspace()
    preset = state.get_preset()

    prep_csv = state.get_artifact("prep_csv")
    if prep_csv is None:
        st.info(translator.t("messages.no_artifacts"))
        return

    if st.button(translator.t("actions.run"), key="msc_run"):
        with st.spinner("Executando θ/MSC..."):
            result = run_cli.run_theta_msc(prep_csv, preset, workspace)
        for key, path in result.outputs.items():
            state.register_artifact(key, path, description="msc")
        st.toast(translator.t("messages.ready"))

    curve_path = state.get_artifact("msc_curve")
    plot_path = state.get_artifact("msc_plot")
    if curve_path and curve_path.exists():
        df = pd.read_csv(curve_path)
        if {"Ea_kJ", "metric"}.issubset(df.columns):
            figure = px.line(df, x="Ea_kJ", y="metric", markers=True)
            st.plotly_chart(figure, use_container_width=True)
        st.download_button(
            label=f"CSV · {curve_path.name}",
            data=curve_path.read_bytes(),
            file_name=curve_path.name,
            key="msc_csv_dl",
        )
    if plot_path and plot_path.exists():
        st.download_button(
            label=f"PNG · {plot_path.name}",
            data=plot_path.read_bytes(),
            file_name=plot_path.name,
            key="msc_png_dl",
        )
