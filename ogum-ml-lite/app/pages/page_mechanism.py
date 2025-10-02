"""Mechanism detection Streamlit page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..design.a11y import aria_label, describe_chart, focus_hint
from ..i18n.translate import I18N
from ..services import run_cli, state


def render(translator: I18N) -> None:
    """Render mechanism detection workflow."""

    st.subheader(translator.t("menu.mechanism"))
    st.caption(str(focus_hint(translator.t("wizard.focus_hint"))))
    workspace = state.get_workspace()
    preset = state.get_preset()

    theta_table = state.get_artifact("theta_table")
    if theta_table is None:
        st.info(translator.t("wizard.blockers.need_theta"))
        return

    if st.button(
        translator.t("actions.run"),
        key="mechanism_run",
        **aria_label(translator.t("microcopy.run_mechanism")),
    ):
        with st.spinner(translator.t("microcopy.spinner_mechanism")):
            result = run_cli.run_mechanism(theta_table, preset, workspace)
        mech_path = result.outputs["mechanism"]
        state.register_artifact("mechanism", mech_path, description="mechanism")
        st.toast(translator.t("microcopy.mechanism_ready"))

    mech_path = state.get_artifact("mechanism")
    if mech_path and mech_path.exists():
        df = pd.read_csv(mech_path)
        st.dataframe(df.head(50))
        st.caption(
            describe_chart(translator.t("microcopy.describe_mechanism"), prefix="🧪")
        )
        st.download_button(
            label=f"CSV · {mech_path.name}",
            data=mech_path.read_bytes(),
            file_name=mech_path.name,
            key="mechanism_dl",
        )
