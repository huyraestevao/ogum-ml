"""Guided wizard page orchestrating the Ogum pipeline end-to-end."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import plotly.express as px
import streamlit as st

from ..design.a11y import aria_label, describe_chart, focus_hint
from ..i18n.translate import I18N
from ..services import run_cli, state, telemetry, validators

STATE_KEY = "wizard_step"
FLAGS_KEY = "wizard_flags"
CONTEXT_KEY = "wizard_context"


def _variant(name: str, default: str) -> str:
    variants = st.session_state.get("ab_variants", {})
    return variants.get(name, default)


def _buttons_on_top() -> bool:
    return _variant("run_buttons_position", "top") == "top"


def _msc_layout() -> str:
    return _variant("msc_controls_layout", "compact")


@dataclass(frozen=True)
class WizardStep:
    """Definition for a wizard step."""

    key: str
    renderer: Callable[[I18N], None]
    blocker_key: str | None = None


def _context() -> dict[str, Any]:
    return st.session_state.setdefault(CONTEXT_KEY, {})


def _flags() -> dict[str, bool]:
    flags: dict[str, bool] = st.session_state.setdefault(FLAGS_KEY, {})
    return flags


def _mark_complete(step_key: str) -> None:
    flags = _flags()
    flags[step_key] = True
    st.session_state[FLAGS_KEY] = flags


def _is_complete(step_key: str) -> bool:
    return _flags().get(step_key, False)


def _artifact_exists(key: str) -> bool:
    path = state.get_artifact(key)
    return bool(path and path.exists())


def _sync_flags() -> None:
    auto_rules: dict[str, Callable[[], bool]] = {
        "data": lambda: _artifact_exists("prep_csv"),
        "features": lambda: _artifact_exists("features_csv"),
        "msc": lambda: _artifact_exists("theta_table"),
        "segments": lambda: _artifact_exists("segments"),
        "mechanism": lambda: _artifact_exists("mechanism"),
        "ml": lambda: _artifact_exists("ml_cls")
        or _artifact_exists("ml_reg")
        or _artifact_exists("predictions"),
        "export": lambda: _artifact_exists("export_report")
        and _artifact_exists("session_zip"),
    }
    flags = _flags()
    for key, rule in auto_rules.items():
        if not flags.get(key) and rule():
            flags[key] = True
    st.session_state[FLAGS_KEY] = flags


def _init_state() -> None:
    st.session_state.setdefault(STATE_KEY, 0)
    _context()
    _sync_flags()


def _render_step_progress(translator: I18N, current_index: int) -> None:
    cols = st.columns(len(STEPS))
    for idx, step in enumerate(STEPS):
        label = translator.t(f"wizard.steps.{step.key}.title")
        description = translator.t(f"wizard.steps.{step.key}.description")
        status = (
            "✅" if _is_complete(step.key) else ("🟡" if idx == current_index else "⚪")
        )
        cols[idx].markdown(f"{status} **{label}**")
        if idx == current_index:
            cols[idx].caption(description)


def _render_navigation(translator: I18N, current_index: int, step: WizardStep) -> None:
    cols = st.columns([1, 1, 3])
    if cols[0].button(
        translator.t("wizard.actions.previous"),
        key=f"wizard_prev_{current_index}",
        disabled=current_index == 0,
        **aria_label(translator.t("wizard.actions.previous")),
    ):
        st.session_state[STATE_KEY] = max(0, current_index - 1)

    can_advance = current_index < len(STEPS) - 1 and _is_complete(step.key)
    if not can_advance and step.blocker_key:
        cols[2].caption(translator.t(step.blocker_key))

    if cols[1].button(
        translator.t("wizard.actions.next"),
        key=f"wizard_next_{current_index}",
        disabled=not can_advance,
        **aria_label(translator.t("wizard.actions.next")),
    ):
        st.session_state[STATE_KEY] = min(len(STEPS) - 1, current_index + 1)


def _list_csv(workspace_path: Path) -> list[Path]:
    uploads = workspace_path / "uploads"
    if not uploads.exists():
        return []
    return sorted(uploads.glob("*.csv"))


def _render_data_step(translator: I18N) -> None:
    workspace = state.get_workspace()
    preset = state.get_preset()
    ctx = _context()

    st.caption(f"ⓘ {translator.t('wizard.tooltips.data')}")
    uploaded = st.file_uploader(
        translator.t("microcopy.upload_csv"),
        type=["csv"],
        key="wizard_upload",
        **aria_label(translator.t("microcopy.upload_csv")),
    )
    if uploaded is not None:
        target = state.persist_upload(uploaded)
        st.toast(f"[ok] {uploaded.name} → {target.name}")

    csv_files = _list_csv(workspace.path)
    if not csv_files:
        st.info(translator.t("microcopy.missing_artifact"))
        return

    default = ctx.get("selected_csv")
    option = st.selectbox(
        translator.t("microcopy.select_csv"),
        options=[str(path) for path in csv_files],
        format_func=lambda value: Path(value).name,
        key="wizard_dataset",
        index=(
            [str(path) for path in csv_files].index(default)
            if default in [str(path) for path in csv_files]
            else 0
        ),
    )
    selected = Path(option)
    ctx["selected_csv"] = str(selected)

    def _render_actions() -> None:
        cols = st.columns(2)
        if cols[0].button(
            translator.t("microcopy.validate_data"),
            key="wizard_validate",
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
            key="wizard_prep",
            **aria_label(translator.t("microcopy.run_prep")),
        ):
            telemetry.log_event(
                "ui.action",
                {
                    "action": "run_prep",
                    "experiment": "run_buttons_position",
                    "variant": _variant("run_buttons_position", "top"),
                },
                workspace=workspace,
            )
            with st.spinner(translator.t("microcopy.spinner_prep")):
                result = run_cli.run_prep(selected, preset, workspace)
            prep_csv = result.outputs["prep_csv"]
            state.register_artifact("prep_csv", prep_csv, description="prep")
            st.toast(translator.t("microcopy.prep_ready"))
            _mark_complete("data")

    if _buttons_on_top():
        _render_actions()

    preview_target = state.get_artifact("prep_csv") or selected
    if preview_target.exists():
        st.caption(translator.t("microcopy.focus_primary"))
        st.dataframe(pd.read_csv(preview_target).head(30))

    if not _buttons_on_top():
        st.divider()
        _render_actions()


def _render_features_step(translator: I18N) -> None:
    workspace = state.get_workspace()
    preset = state.get_preset()
    st.caption(f"ⓘ {translator.t('wizard.tooltips.features')}")

    prep_csv = state.get_artifact("prep_csv")
    if prep_csv is None or not prep_csv.exists():
        st.info(translator.t("wizard.blockers.need_prep"))
        return

    ctx = _context()
    use_prep = st.checkbox(
        translator.t("wizard.features.use_prep"),
        value=ctx.get("features_use_prep", True),
        key="wizard_features_use_prep",
    )
    ctx["features_use_prep"] = use_prep

    def _run_features() -> None:
        telemetry.log_event(
            "ui.action",
            {
                "action": "run_features",
                "experiment": "run_buttons_position",
                "variant": _variant("run_buttons_position", "top"),
            },
            workspace=workspace,
        )
        with st.spinner(translator.t("microcopy.spinner_features")):
            result = run_cli.run_features(
                prep_csv,
                preset,
                workspace,
                use_prep=use_prep,
            )
        features_csv = result.outputs["features_csv"]
        state.register_artifact("features_csv", features_csv, description="features")
        st.toast(translator.t("microcopy.features_ready"))
        _mark_complete("features")

    if _buttons_on_top() and st.button(
        translator.t("microcopy.run_features"),
        key="wizard_features_run",
        **aria_label(translator.t("microcopy.run_features")),
    ):
        _run_features()

    features_csv = state.get_artifact("features_csv")
    if features_csv and features_csv.exists():
        st.dataframe(pd.read_csv(features_csv).head(25))
    else:
        st.info(translator.t("microcopy.missing_artifact"))

    if not _buttons_on_top():
        st.divider()
        if st.button(
            translator.t("microcopy.run_features"),
            key="wizard_features_run_bottom",
            **aria_label(translator.t("microcopy.run_features")),
        ):
            _run_features()


def _render_msc_step(translator: I18N) -> None:
    workspace = state.get_workspace()
    preset = state.get_preset()
    prep_csv = state.get_artifact("prep_csv")
    if prep_csv is None or not prep_csv.exists():
        st.info(translator.t("wizard.blockers.need_prep"))
        return

    layout_variant = _msc_layout()

    def _run_msc(key_suffix: str) -> None:
        telemetry.log_event(
            "ui.action",
            {
                "action": "run_msc",
                "experiment": "msc_controls_layout",
                "variant": layout_variant,
            },
            workspace=workspace,
        )
        with st.spinner(translator.t("microcopy.spinner_msc")):
            result = run_cli.run_theta_msc(prep_csv, preset, workspace)
        for key, path in result.outputs.items():
            state.register_artifact(key, path, description="msc")
        st.toast(translator.t("microcopy.msc_ready"))
        _mark_complete("msc")

    def _render_controls(key_suffix: str) -> None:
        label = translator.t("microcopy.run_msc")
        if layout_variant == "expanded":
            cols = st.columns([3, 2])
            with cols[0]:
                if st.button(
                    label,
                    key=f"wizard_msc_run_{key_suffix}",
                    **aria_label(label),
                ):
                    _run_msc(key_suffix)
            with cols[1]:
                msc_cfg = preset.get("msc", {})
                st.markdown(
                    "\n".join(
                        [
                            f"**Ea (kJ/mol)**: {msc_cfg.get('ea_kj', [])}",
                            f"**Métrica**: {msc_cfg.get('metric', 'segmented')}",
                        ]
                    )
                )
        else:
            if st.button(
                label,
                key=f"wizard_msc_run_{key_suffix}",
                **aria_label(label),
            ):
                _run_msc(key_suffix)

    st.caption(f"ⓘ {translator.t('wizard.tooltips.msc')}")
    if _buttons_on_top():
        _render_controls("top")

    curve_path = state.get_artifact("msc_curve")
    if curve_path and curve_path.exists():
        df = pd.read_csv(curve_path)
        if {"Ea_kJ", "metric"}.issubset(df.columns):
            figure = px.line(df, x="Ea_kJ", y="metric", markers=True)
            st.plotly_chart(figure, use_container_width=True)
            st.caption(describe_chart(translator.t("microcopy.describe_msc")))
        st.download_button(
            label=f"CSV · {curve_path.name}",
            data=curve_path.read_bytes(),
            file_name=curve_path.name,
            key="wizard_msc_csv",
        )

    if not _buttons_on_top():
        st.divider()
        _render_controls("bottom")


def _render_segments_step(translator: I18N) -> None:
    workspace = state.get_workspace()
    preset = state.get_preset()
    prep_csv = state.get_artifact("prep_csv")
    if prep_csv is None or not prep_csv.exists():
        st.info(translator.t("wizard.blockers.need_prep"))
        return

    st.caption(f"ⓘ {translator.t('wizard.tooltips.segments')}")

    def _run_segments(key_suffix: str) -> None:
        telemetry.log_event(
            "ui.action",
            {
                "action": "run_segments",
                "experiment": "run_buttons_position",
                "variant": _variant("run_buttons_position", "top"),
            },
            workspace=workspace,
        )
        with st.spinner(translator.t("microcopy.spinner_segments")):
            result = run_cli.run_segmentation(prep_csv, preset, workspace)
        segments_path = result.outputs["segments"]
        state.register_artifact("segments", segments_path, description="segments")
        st.toast(translator.t("microcopy.segments_ready"))
        _mark_complete("segments")

    if _buttons_on_top() and st.button(
        translator.t("microcopy.run_segments"),
        key="wizard_segments_run",
        **aria_label(translator.t("microcopy.run_segments")),
    ):
        _run_segments("top")

    segments_path = state.get_artifact("segments")
    if segments_path and segments_path.exists():
        try:
            payload = json.loads(segments_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            st.code(segments_path.read_text(encoding="utf-8"), language="json")
        else:
            st.dataframe(pd.DataFrame(payload))
            st.caption(
                describe_chart(translator.t("microcopy.describe_segments"), prefix="🧮")
            )
        st.download_button(
            label=f"JSON · {segments_path.name}",
            data=segments_path.read_bytes(),
            file_name=segments_path.name,
            key="wizard_segments_json",
        )

    if not _buttons_on_top():
        st.divider()
        if st.button(
            translator.t("microcopy.run_segments"),
            key="wizard_segments_run_bottom",
            **aria_label(translator.t("microcopy.run_segments")),
        ):
            _run_segments("bottom")


def _render_mechanism_step(translator: I18N) -> None:
    workspace = state.get_workspace()
    preset = state.get_preset()
    theta_table = state.get_artifact("theta_table")
    if theta_table is None or not theta_table.exists():
        st.info(translator.t("wizard.blockers.need_theta"))
        return

    st.caption(f"ⓘ {translator.t('wizard.tooltips.mechanism')}")

    def _run_mechanism(key_suffix: str) -> None:
        telemetry.log_event(
            "ui.action",
            {
                "action": "run_mechanism",
                "experiment": "run_buttons_position",
                "variant": _variant("run_buttons_position", "top"),
            },
            workspace=workspace,
        )
        with st.spinner(translator.t("microcopy.spinner_mechanism")):
            result = run_cli.run_mechanism(theta_table, preset, workspace)
        mech_path = result.outputs["mechanism"]
        state.register_artifact("mechanism", mech_path, description="mechanism")
        st.toast(translator.t("microcopy.mechanism_ready"))
        _mark_complete("mechanism")

    if _buttons_on_top() and st.button(
        translator.t("microcopy.run_mechanism"),
        key="wizard_mechanism_run",
        **aria_label(translator.t("microcopy.run_mechanism")),
    ):
        _run_mechanism("top")

    mech_path = state.get_artifact("mechanism")
    if mech_path and mech_path.exists():
        st.dataframe(pd.read_csv(mech_path).head(50))
        st.caption(
            describe_chart(translator.t("microcopy.describe_mechanism"), prefix="🧪")
        )
        st.download_button(
            label=f"CSV · {mech_path.name}",
            data=mech_path.read_bytes(),
            file_name=mech_path.name,
            key="wizard_mechanism_csv",
        )

    if not _buttons_on_top():
        st.divider()
        if st.button(
            translator.t("microcopy.run_mechanism"),
            key="wizard_mechanism_run_bottom",
            **aria_label(translator.t("microcopy.run_mechanism")),
        ):
            _run_mechanism("bottom")


def _render_ml_step(translator: I18N) -> None:
    workspace = state.get_workspace()
    preset = state.get_preset()
    features_csv = state.get_artifact("features_csv")
    if features_csv is None or not features_csv.exists():
        st.info(translator.t("wizard.blockers.need_features"))
        return

    mechanism = state.get_artifact("mechanism")
    if mechanism is None or not mechanism.exists():
        st.warning(translator.t("wizard.blockers.need_mechanism"))

    st.caption(f"ⓘ {translator.t('wizard.tooltips.ml')}")

    def _run_ml_train(action: str, runner: Callable[[], Any], key_suffix: str) -> None:
        telemetry.log_event(
            "ui.action",
            {
                "action": action,
                "experiment": "run_buttons_position",
                "variant": _variant("run_buttons_position", "top"),
            },
            workspace=workspace,
        )
        with st.spinner(translator.t("microcopy.spinner_ml")):
            result = runner()
        outdir = result.outputs["outdir"]
        artifact_key = "ml_cls" if action == "run_ml_cls" else "ml_reg"
        state.register_artifact(artifact_key, outdir, description=artifact_key)
        state.register_artifact(
            f"{artifact_key}_card",
            result.outputs["model_card"],
            description=artifact_key,
        )
        st.toast(translator.t("microcopy.ml_ready"))
        _mark_complete("ml")
        if result.stdout:
            st.code(result.stdout)

    def _render_train_buttons(position: str) -> None:
        cols = st.columns(2)
        if cols[0].button(
            translator.t("microcopy.run_ml_cls"),
            key=f"wizard_ml_cls_{position}",
            **aria_label(translator.t("microcopy.run_ml_cls")),
        ):
            _run_ml_train(
                "run_ml_cls",
                lambda: run_cli.run_ml_train_cls(features_csv, preset, workspace),
                position,
            )
        if cols[1].button(
            translator.t("microcopy.run_ml_reg"),
            key=f"wizard_ml_reg_{position}",
            **aria_label(translator.t("microcopy.run_ml_reg")),
        ):
            _run_ml_train(
                "run_ml_reg",
                lambda: run_cli.run_ml_train_reg(features_csv, preset, workspace),
                position,
            )

    if _buttons_on_top():
        _render_train_buttons("top")

    models = sorted(workspace.path.rglob("*.joblib"))
    option = st.selectbox(
        translator.t("microcopy.run_ml_predict"),
        options=[str(path) for path in models],
        format_func=lambda value: Path(value).name,
        key="wizard_ml_model",
    )
    if option and st.button(
        translator.t("microcopy.run_ml_predict"),
        key="wizard_ml_predict",
        **aria_label(translator.t("microcopy.run_ml_predict")),
    ):
        telemetry.log_event(
            "ui.action",
            {
                "action": "run_ml_predict",
                "experiment": "run_buttons_position",
                "variant": _variant("run_buttons_position", "top"),
            },
            workspace=workspace,
        )
        with st.spinner(translator.t("microcopy.spinner_predict")):
            result = run_cli.run_ml_predict(
                features_csv, Path(option), preset, workspace
            )
        predictions = result.outputs["predictions"]
        state.register_artifact("predictions", predictions, description="predictions")
        st.toast(translator.t("microcopy.predict_ready"))
        _mark_complete("ml")

    if not _buttons_on_top():
        st.divider()
        _render_train_buttons("bottom")

    predictions = state.get_artifact("predictions")
    if predictions and predictions.exists():
        st.dataframe(pd.read_csv(predictions).head(25))
        st.caption(describe_chart(translator.t("microcopy.describe_ml"), prefix="🤖"))
        st.download_button(
            label=f"CSV · {predictions.name}",
            data=predictions.read_bytes(),
            file_name=predictions.name,
            key="wizard_predictions_csv",
        )


def _render_export_step(translator: I18N) -> None:
    workspace = state.get_workspace()
    preset = state.get_preset()
    st.caption(f"ⓘ {translator.t('wizard.tooltips.export')}")

    def _run_export(key_suffix: str) -> None:
        telemetry.log_event(
            "ui.action",
            {
                "action": "run_export",
                "experiment": "run_buttons_position",
                "variant": _variant("run_buttons_position", "top"),
            },
            workspace=workspace,
        )
        with st.spinner(translator.t("microcopy.spinner_export")):
            result = run_cli.export_report(workspace.path, preset, workspace)
        for key, path in result.outputs.items():
            state.register_artifact(f"export_{key}", path, description="export")
        state.register_artifact("session_zip", result.outputs["zip"], description="zip")
        st.toast(translator.t("microcopy.export_ready"))
        _mark_complete("export")

    if _buttons_on_top() and st.button(
        translator.t("microcopy.run_export"),
        key="wizard_export_run",
        **aria_label(translator.t("microcopy.run_export")),
    ):
        _run_export("top")

    report = state.get_artifact("export_report")
    if report and report.exists():
        st.download_button(
            label=f"XLSX · {report.name}",
            data=report.read_bytes(),
            file_name=report.name,
            key="wizard_export_xlsx",
        )
    zip_path = state.get_artifact("session_zip")
    if zip_path and zip_path.exists():
        st.download_button(
            label=f"ZIP · {zip_path.name}",
            data=zip_path.read_bytes(),
            file_name=zip_path.name,
            key="wizard_export_zip",
        )

    if not _buttons_on_top():
        st.divider()
        if st.button(
            translator.t("microcopy.run_export"),
            key="wizard_export_run_bottom",
            **aria_label(translator.t("microcopy.run_export")),
        ):
            _run_export("bottom")


STEPS: tuple[WizardStep, ...] = (
    WizardStep("data", _render_data_step, blocker_key="wizard.blockers.need_prep"),
    WizardStep(
        "features", _render_features_step, blocker_key="wizard.blockers.need_features"
    ),
    WizardStep("msc", _render_msc_step, blocker_key="wizard.blockers.need_theta"),
    WizardStep(
        "segments", _render_segments_step, blocker_key="wizard.blockers.need_segments"
    ),
    WizardStep(
        "mechanism",
        _render_mechanism_step,
        blocker_key="wizard.blockers.need_mechanism",
    ),
    WizardStep("ml", _render_ml_step, blocker_key="wizard.blockers.need_ml"),
    WizardStep("export", _render_export_step),
)


def _render_wizard_layout(translator: I18N) -> None:
    current_index = st.session_state.get(STATE_KEY, 0)
    current_index = max(0, min(current_index, len(STEPS) - 1))
    st.session_state[STATE_KEY] = current_index

    _render_step_progress(translator, current_index)
    st.divider()

    step = STEPS[current_index]
    st.markdown(f"### {translator.t(f'wizard.steps.{step.key}.title')}")
    st.caption(translator.t(f"wizard.steps.{step.key}.description"))

    step.renderer(translator)

    _render_navigation(translator, current_index, step)


def _render_tabs_layout(translator: I18N) -> None:
    status_line = " ".join(("✅" if _is_complete(step.key) else "⚪" for step in STEPS))
    st.caption(translator.t("wizard.tabs_status", status=status_line))
    tab_labels = [translator.t(f"wizard.steps.{step.key}.title") for step in STEPS]
    tabs = st.tabs(tab_labels)
    for step, tab in zip(STEPS, tabs):
        with tab:
            st.markdown(f"### {translator.t(f'wizard.steps.{step.key}.title')}")
            st.caption(translator.t(f"wizard.steps.{step.key}.description"))
            step.renderer(translator)


def render(translator: I18N) -> None:
    """Render the guided wizard UI."""

    st.subheader(translator.t("wizard.title"))
    st.caption(translator.t("wizard.intro"))
    st.caption(str(focus_hint(translator.t("wizard.focus_hint"))))

    _init_state()
    layout_variant = _variant("wizard_vs_tabs", "wizard")
    if layout_variant == "tabs":
        _render_tabs_layout(translator)
    else:
        _render_wizard_layout(translator)
