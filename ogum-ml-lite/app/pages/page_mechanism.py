"""Mechanism analysis page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..i18n.translate import I18N
from ..services import run_cli, state


def render(translator: I18N) -> None:
    """Render mechanism detection workflow."""

    st.subheader(translator.t("menu.mechanism"))
    workspace = state.get_workspace()
    preset = state.get_preset()

    theta_table = state.get_artifact("theta_table")
    if theta_table is None:
        st.info(translator.t("messages.no_artifacts"))
        return

    if st.button(translator.t("actions.run"), key="mechanism_run"):
        with st.spinner("Calculando mecanismo..."):
            result = run_cli.run_mechanism(theta_table, preset, workspace)
        mech_path = result.outputs["mechanism"]
        state.register_artifact("mechanism", mech_path, description="mechanism")
        st.toast(translator.t("messages.ready"))

    mech_path = state.get_artifact("mechanism")
    if mech_path and mech_path.exists():
        df = pd.read_csv(mech_path)
        st.dataframe(df.head(50))
        st.download_button(
            label=f"CSV · {mech_path.name}",
            data=mech_path.read_bytes(),
            file_name=mech_path.name,
            key="mechanism_dl",
        )
