"""Machine learning orchestration page."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

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
    workspace = state.get_workspace()
    preset = state.get_preset()
    features_csv = state.get_artifact("features_csv")
    if features_csv is None:
        st.info(translator.t("messages.no_artifacts"))
        return

    tab_train_cls, tab_train_reg, tab_predict = st.tabs(
        ["Train CLS", "Train REG", "Predict"]
    )

    with tab_train_cls:
        if st.button(translator.t("actions.train"), key="ml_train_cls"):
            with st.spinner("Treinando classificador..."):
                result = run_cli.run_ml_train_cls(features_csv, preset, workspace)
            outdir = result.outputs["outdir"]
            state.register_artifact("ml_cls", outdir, description="ml_cls")
            state.register_artifact(
                "ml_cls_card", result.outputs["model_card"], description="ml_cls"
            )
            st.toast(translator.t("messages.ready"))
            card = _load_card(result.outputs["model_card"])
            if card:
                st.json(card)

    with tab_train_reg:
        if st.button(translator.t("actions.train"), key="ml_train_reg"):
            with st.spinner("Treinando regressor..."):
                result = run_cli.run_ml_train_reg(features_csv, preset, workspace)
            outdir = result.outputs["outdir"]
            state.register_artifact("ml_reg", outdir, description="ml_reg")
            state.register_artifact(
                "ml_reg_card", result.outputs["model_card"], description="ml_reg"
            )
            st.toast(translator.t("messages.ready"))
            card = _load_card(result.outputs["model_card"])
            if card:
                st.json(card)

    with tab_predict:
        models = _list_models(workspace.path)
        option = st.selectbox(
            "Modelo",
            options=[str(path) for path in models],
            format_func=lambda value: Path(value).name,
            key="ml_model_pick",
        )
        if option and st.button(translator.t("actions.predict"), key="ml_predict"):
            with st.spinner("Gerando predições..."):
                result = run_cli.run_ml_predict(
                    features_csv, Path(option), preset, workspace
                )
            predictions = result.outputs["predictions"]
            state.register_artifact(
                "predictions", predictions, description="predictions"
            )
            st.toast(translator.t("messages.ready"))
        predictions = state.get_artifact("predictions")
        if predictions and predictions.exists():
            st.dataframe(pd.read_csv(predictions).head(25))
            st.download_button(
                label=f"CSV · {predictions.name}",
                data=predictions.read_bytes(),
                file_name=predictions.name,
                key="predictions_dl",
            )
