"""Streamlit page for the educational mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from ogum_lite.ui.workspace import Workspace

from app.edu import components, exercises, exporter, simulators
from app.i18n.translate import I18N
from app.services import run_cli, state, validators


def _workspace_csvs(workspace: Workspace) -> list[Path]:
    uploads = workspace.resolve("uploads")
    if not uploads.exists():
        return []
    return sorted(uploads.glob("*.csv"))


def _build_sample_dataset(workspace: state.Workspace) -> Path:
    curves = exercises.get_reference_curves()
    rows: list[dict[str, Any]] = []
    for curve in curves:
        for t, temp, y in zip(curve["time_s"], curve["temp_C"], curve["y"]):
            rows.append(
                {
                    "sample_id": curve["label"],
                    "time_s": float(t),
                    "temp_C": float(temp),
                    "rho_rel": float(y),
                }
            )
    df = pd.DataFrame(rows)
    target = workspace.resolve("edu_sample.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return target


def _load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _extract_curves(df: pd.DataFrame) -> list[dict]:
    expected = {"sample_id", "time_s", "temp_C", "rho_rel"}
    if not expected.issubset(df.columns):
        missing = ", ".join(sorted(expected - set(df.columns)))
        raise ValueError(f"Missing required columns: {missing}")
    curves = []
    for sample_id, group in df.groupby("sample_id"):
        ordered = group.sort_values("time_s")
        curves.append(
            {
                "label": str(sample_id),
                "time_s": ordered["time_s"].to_numpy(dtype=float),
                "temp_C": ordered["temp_C"].to_numpy(dtype=float),
                "y": ordered["rho_rel"].to_numpy(dtype=float),
            }
        )
    return curves


def _store_fig(session_key: str, fig) -> None:
    st.session_state[session_key] = fig
    st.session_state[f"{session_key}_bytes"] = exporter.capture_fig(fig)


def _get_fig_bytes(session_key: str) -> bytes | None:
    return st.session_state.get(f"{session_key}_bytes")


def _render_dataset_step(translator: I18N) -> pd.DataFrame | None:
    st.header(translator.t("edu.steps.load.title"))
    st.caption(translator.t("edu.steps.load.hint"))

    workspace = state.get_workspace()
    sample_path = _build_sample_dataset(workspace)

    uploaded = st.file_uploader(
        translator.t("edu.labels.upload"),
        type=["csv"],
        key="edu_upload",
    )
    if uploaded is not None:
        state.persist_upload(uploaded, subdir="uploads")
        st.toast(translator.t("edu.messages.upload_success", name=uploaded.name))
    csv_files = [sample_path] + _workspace_csvs(workspace)
    options = {str(path): path for path in csv_files}
    default_key = str(sample_path)
    selection = st.selectbox(
        translator.t("edu.labels.dataset_select"),
        options=list(options.keys()),
        format_func=lambda key: options[key].name,
        index=list(options.keys()).index(default_key) if default_key in options else 0,
        key="edu_dataset_option",
    )
    dataset_path = options.get(selection)
    dataset_df: pd.DataFrame | None = None

    if dataset_path and dataset_path.exists():
        try:
            dataset_df = _load_dataset(dataset_path)
        except Exception as exc:  # pragma: no cover - defensive
            st.error(translator.t("edu.messages.load_failed", error=str(exc)))
    else:
        st.warning(translator.t("edu.messages.dataset_missing"))

    cols = st.columns(2)
    if dataset_path and cols[0].button(translator.t("edu.actions.validate")):
        summary = validators.validate_long(dataset_path)
        if summary.ok:
            st.toast(translator.t("edu.messages.validation_ok"))
        else:
            st.toast(translator.t("edu.messages.validation_warn"))
        for issue in summary.issues:
            st.warning(issue)

    if dataset_path and cols[1].button(translator.t("edu.actions.prepare")):
        preset = state.get_preset()
        with st.spinner(translator.t("edu.messages.prep_running")):
            result = run_cli.run_prep(dataset_path, preset, workspace)
        prep_csv = result.outputs.get("prep_csv")
        if prep_csv:
            state.register_artifact("edu_prep_csv", prep_csv, description="edu")
            st.toast(translator.t("edu.messages.prep_ok", name=prep_csv.name))
        else:
            st.error(translator.t("edu.messages.prep_failed"))

    if dataset_df is not None:
        st.dataframe(dataset_df.head(20))

    components.card_conceito(
        translator.t("edu.cards.theta.title"),
        translator.t("edu.cards.theta.body"),
        translator.t("edu.cards.theta.formula"),
    )
    return dataset_df


def _render_simulation_step(
    translator: I18N,
    dataset_df: pd.DataFrame | None,
) -> tuple[simulators.CollapseResult | None, simulators.BlaineResult | None, float]:
    st.header(translator.t("edu.steps.simulate.title"))
    st.caption(translator.t("edu.steps.simulate.hint"))

    if dataset_df is None:
        components.callout("warn", translator.t("edu.messages.need_dataset"))
        return None, None, exercises.get_references()["ea"]

    try:
        curves = _extract_curves(dataset_df)
    except ValueError as exc:
        message = translator.t("edu.messages.extract_failed", error=str(exc))
        components.callout("warn", message)
        return None, None, exercises.get_references()["ea"]

    references = exercises.get_references()
    ea_default = st.session_state.get("edu_ea", references["ea"])
    ea_value = st.slider(
        translator.t("edu.labels.ea_slider"),
        min_value=200.0,
        max_value=400.0,
        value=float(ea_default),
        step=5.0,
        key="edu_ea",
    )

    theta_container = st.container()
    if st.button(translator.t("edu.actions.compute_theta")):
        first_curve = curves[0]
        theta = simulators.simulate_theta(
            first_curve["temp_C"], first_curve["time_s"], ea_value
        )
        fig = simulators.make_fig_theta(
            first_curve["time_s"], first_curve["temp_C"], theta
        )
        _store_fig("edu_theta_fig", fig)
        with theta_container:
            components.figure_plotly(fig, translator.t("edu.captions.theta"))
    elif "edu_theta_fig" in st.session_state:
        fig = st.session_state["edu_theta_fig"]
        with theta_container:
            components.figure_plotly(fig, translator.t("edu.captions.theta"))

    cols = st.columns(3)
    seg_inputs = {}
    for idx, level in enumerate((55, 70, 90)):
        seg_inputs[level] = cols[idx].number_input(
            translator.t(f"edu.labels.seg_{level}"),
            min_value=0.0,
            max_value=1.0,
            value=float(references[f"seg_{level}"]),
            step=0.01,
            key=f"edu_seg_{level}",
        )

    collapse_result = None
    collapse_placeholder = st.container()
    if st.button(translator.t("edu.actions.collapse_msc")):
        collapse_result = simulators.simulate_msc_collapse(curves, ea_value)
        st.session_state["edu_collapse"] = collapse_result
        _store_fig("edu_collapse_fig", collapse_result.figure)
        with collapse_placeholder:
            components.figure_plotly(
                collapse_result.figure, translator.t("edu.captions.collapse")
            )
        st.metric("MSE", f"{collapse_result.mse:.4f}")
    elif "edu_collapse" in st.session_state:
        collapse_result = st.session_state["edu_collapse"]
        with collapse_placeholder:
            components.figure_plotly(
                collapse_result.figure, translator.t("edu.captions.collapse")
            )
        st.metric("MSE", f"{collapse_result.mse:.4f}")

    blaine_result = None
    blaine_placeholder = st.container()
    if st.button(translator.t("edu.actions.linearize_blaine")):
        target = collapse_result or st.session_state.get("edu_collapse")
        if target is None:
            components.callout("info", translator.t("edu.messages.need_collapse"))
        else:
            blaine_result = simulators.simulate_blaine_linearization(
                target.grid_theta + 1e-6, target.mean_curve
            )
            st.session_state["edu_blaine"] = blaine_result
            _store_fig("edu_blaine_fig", blaine_result.figure)
            with blaine_placeholder:
                components.figure_plotly(
                    blaine_result.figure, translator.t("edu.captions.blaine")
                )
            st.metric("n", f"{blaine_result.n_est:.3f}")
            st.metric("R²", f"{blaine_result.r2:.3f}")
            st.metric("MSE", f"{blaine_result.mse:.4f}")
    elif "edu_blaine" in st.session_state:
        blaine_result = st.session_state["edu_blaine"]
        with blaine_placeholder:
            components.figure_plotly(
                blaine_result.figure, translator.t("edu.captions.blaine")
            )
        st.metric("n", f"{blaine_result.n_est:.3f}")
        st.metric("R²", f"{blaine_result.r2:.3f}")
        st.metric("MSE", f"{blaine_result.mse:.4f}")

    components.card_conceito(
        translator.t("edu.cards.blaine.title"),
        translator.t("edu.cards.blaine.body"),
        translator.t("edu.cards.blaine.formula"),
    )

    st.session_state["edu_segments"] = seg_inputs
    return collapse_result, blaine_result, ea_value


def _render_explore_step(
    translator: I18N,
    collapse_result: simulators.CollapseResult | None,
    blaine_result: simulators.BlaineResult | None,
    ea_value: float,
) -> None:
    st.header(translator.t("edu.steps.explore.title"))
    st.caption(translator.t("edu.steps.explore.hint"))

    answers = st.session_state.setdefault("edu_answers", {})
    results = []
    for exercise in exercises.EXERCISES:
        st.markdown(exercise.statement_md)
        inputs_payload = {}
        cols = st.columns(len(exercise.inputs_spec))
        for idx, (key, spec) in enumerate(exercise.inputs_spec.items()):
            default_value = answers.get(exercise.key, {}).get(key, 0.0)
            inputs_payload[key] = cols[idx].number_input(
                spec.get("label", key),
                value=float(default_value),
            )
        if st.button(translator.t("edu.actions.check"), key=f"check_{exercise.key}"):
            outcome = exercise.evaluate(inputs_payload)
            answers.setdefault(exercise.key, {})
            answers[exercise.key].update(inputs_payload)
            answers[exercise.key]["score"] = outcome["score"]
            answers[exercise.key]["feedback"] = outcome["feedback"]
            st.session_state["edu_answers"] = answers
        stored = answers.get(exercise.key, {})
        if stored.get("feedback"):
            st.info(f"Score {stored['score']:.2f} — {stored['feedback']}")
        results.append(
            {
                "title": exercise.statement_md.split("\n", 1)[0],
                "statement": exercise.statement_md,
                "answer": stored,
            }
        )
        st.divider()

    components.callout("tip", translator.t("edu.messages.export_hint"))

    figures = {}
    for key in ("edu_theta_fig", "edu_collapse_fig", "edu_blaine_fig"):
        fig_bytes = _get_fig_bytes(key)
        if fig_bytes:
            figures[key] = fig_bytes

    context = {
        "title": translator.t("edu.title"),
        "concepts": [
            {
                "title": translator.t("edu.cards.theta.title"),
                "body": translator.t("edu.cards.theta.body"),
                "formula": translator.t("edu.cards.theta.formula"),
            }
        ],
        "simulations": [
            {
                "title": translator.t("edu.captions.theta"),
                "body": translator.t("edu.messages.sim_theta", ea=ea_value),
                "figure_key": "edu_theta_fig",
            },
            {
                "title": translator.t("edu.captions.collapse"),
                "body": translator.t("edu.messages.sim_collapse"),
                "figure_key": "edu_collapse_fig",
            },
            {
                "title": translator.t("edu.captions.blaine"),
                "body": translator.t("edu.messages.sim_blaine"),
                "figure_key": "edu_blaine_fig",
            },
        ],
        "exercises": [
            {
                "title": translator.t("edu.messages.exercise_title"),
                "statement": item["statement"],
                "answer": item["answer"],
            }
            for item in results
        ],
    }

    export_dir = state.get_workspace().resolve("edu_exports")
    html_path = export_dir / "modo_educacional.html"
    if st.button(translator.t("edu.actions.export_html")):
        with st.spinner(translator.t("edu.messages.export_running")):
            exporter.export_html(context, figures, html_path)
        st.toast(translator.t("edu.messages.export_ok", name=html_path.name))

    if st.button(translator.t("edu.actions.export_pdf")):
        with st.spinner(translator.t("edu.messages.export_running")):
            pdf_path = export_dir / "modo_educacional.pdf"
            result = exporter.export_pdf(context, figures, pdf_path)
        if result is None:
            components.callout("warn", translator.t("edu.messages.pdf_missing"))
        else:
            st.toast(translator.t("edu.messages.export_ok", name=pdf_path.name))

    if st.button(translator.t("edu.actions.open_advanced")):
        st.session_state["main_menu"] = "wizard"
        st.experimental_rerun()


def render(translator: I18N) -> None:
    """Render the Educational Mode page."""

    st.subheader(translator.t("edu.title"))
    st.write(translator.t("edu.intro"))

    dataset_df = _render_dataset_step(translator)
    collapse_result, blaine_result, ea_value = _render_simulation_step(
        translator, dataset_df
    )
    _render_explore_step(translator, collapse_result, blaine_result, ea_value)
