"""Feature engineering Streamlit page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..design.a11y import aria_label, focus_hint
from ..i18n.translate import I18N
from ..services import run_cli, state, validators


def render(translator: I18N) -> None:
    """Render the feature engineering workflow."""

    st.subheader(translator.t("menu.features"))
    st.caption(str(focus_hint(translator.t("wizard.focus_hint"))))
    workspace = state.get_workspace()
    preset = state.get_preset()

    prep_csv = state.get_artifact("prep_csv")
    if prep_csv is None:
        st.info(translator.t("wizard.blockers.need_prep"))
        return

    run_cols = st.columns(2)
    if run_cols[0].button(
        translator.t("actions.run"),
        key="features_run",
        **aria_label(translator.t("microcopy.run_features")),
    ):
        with st.spinner(translator.t("microcopy.spinner_features")):
            result = run_cli.run_features(prep_csv, preset, workspace)
        features_csv = result.outputs["features_csv"]
        state.register_artifact("features_csv", features_csv, description="features")
        st.toast(translator.t("microcopy.features_ready"))

    features_csv = state.get_artifact("features_csv")
    if features_csv and features_csv.exists():
        if run_cols[1].button(
            translator.t("actions.validate"),
            key="features_validate",
            **aria_label(translator.t("microcopy.validate_data")),
        ):
            summary = validators.validate_features(features_csv)
            if summary.ok:
                st.toast(translator.t("microcopy.validation_ok"))
            else:
                st.toast(translator.t("microcopy.validation_warn"))
            for issue in summary.issues:
                st.warning(issue)

        st.caption(f"Preview · {features_csv.name}")
        st.dataframe(pd.read_csv(features_csv).head(25))
    else:
        st.info(translator.t("microcopy.missing_artifact"))
