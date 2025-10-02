"""Machine learning orchestration page."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ..design.a11y import aria_label, describe_chart, focus_hint
from ..i18n.translate import I18N
from ..services import run_cli, state


def _load_card(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:  # pragma: no cover - defensive
            return {}
    return {}


def _list_models(workspace_root: Path) -> list[Path]:
    return sorted(workspace_root.rglob("*.joblib"))


def render(translator: I18N) -> None:
    """Render ML training and inference workflow."""

    st.subheader(translator.t("menu.ml"))
    st.caption(str(focus_hint(translator.t("wizard.focus_hint"))))
    workspace = state.get_workspace()
    preset = state.get_preset()
    features_csv = state.get_artifact("features_csv")
    if features_csv is None:
        st.info(translator.t("wizard.blockers.need_features"))
        return

    tab_train_cls, tab_train_reg, tab_predict = st.tabs(
        [
            translator.t("microcopy.run_ml_cls"),
            translator.t("microcopy.run_ml_reg"),
            translator.t("actions.predict"),
        ]
    )

    with tab_train_cls:
        if st.button(
            translator.t("actions.train"),
            key="ml_train_cls",
            **aria_label(translator.t("microcopy.run_ml_cls")),
        ):
            with st.spinner(translator.t("microcopy.spinner_ml")):
                result = run_cli.run_ml_train_cls(features_csv, preset, workspace)
            outdir = result.outputs["outdir"]
            state.register_artifact("ml_cls", outdir, description="ml_cls")
            state.register_artifact(
                "ml_cls_card", result.outputs["model_card"], description="ml_cls"
            )
            st.toast(translator.t("microcopy.ml_ready"))
            card = _load_card(result.outputs["model_card"])
            if card:
                st.json(card)

    with tab_train_reg:
        if st.button(
            translator.t("actions.train"),
            key="ml_train_reg",
            **aria_label(translator.t("microcopy.run_ml_reg")),
        ):
            with st.spinner(translator.t("microcopy.spinner_ml")):
                result = run_cli.run_ml_train_reg(features_csv, preset, workspace)
            outdir = result.outputs["outdir"]
            state.register_artifact("ml_reg", outdir, description="ml_reg")
            state.register_artifact(
                "ml_reg_card", result.outputs["model_card"], description="ml_reg"
            )
            st.toast(translator.t("microcopy.ml_ready"))
            card = _load_card(result.outputs["model_card"])
            if card:
                st.json(card)

    with tab_predict:
        models = _list_models(workspace.path)
        option = st.selectbox(
            translator.t("microcopy.run_ml_predict"),
            options=[str(path) for path in models],
            format_func=lambda value: Path(value).name,
            key="ml_model_pick",
        )
        if option and st.button(
            translator.t("actions.predict"),
            key="ml_predict",
            **aria_label(translator.t("microcopy.run_ml_predict")),
        ):
            with st.spinner(translator.t("microcopy.spinner_predict")):
                result = run_cli.run_ml_predict(
                    features_csv, Path(option), preset, workspace
                )
            predictions = result.outputs["predictions"]
            state.register_artifact(
                "predictions", predictions, description="predictions"
            )
            st.toast(translator.t("microcopy.predict_ready"))
        predictions = state.get_artifact("predictions")
        if predictions and predictions.exists():
            st.dataframe(pd.read_csv(predictions).head(25))
            st.caption(
                describe_chart(translator.t("microcopy.describe_ml"), prefix="🤖")
            )
            st.download_button(
                label=f"CSV · {predictions.name}",
                data=predictions.read_bytes(),
                file_name=predictions.name,
                key="predictions_dl",
            )
