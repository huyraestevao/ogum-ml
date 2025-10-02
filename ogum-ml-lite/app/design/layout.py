"""Layout primitives for rendering the Streamlit shell."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import streamlit as st

from ..i18n.translate import I18N
from ..services import state, telemetry
from ..services.themes_loader import load_theme_yaml
from .theme import get_theme

THEMES_ROOT = Path(__file__).resolve().parents[1] / "config" / "themes"


def _theme_override() -> dict | None:
    return st.session_state.get("theme_override")


def _apply_theme(dark: bool) -> None:
    theme = get_theme(dark, _theme_override())
    st.markdown(
        f"""
        <style>
        body {{
            background: {theme['colors']['background']};
            color: {theme['colors']['text']};
            font-family: {theme['typography']['family']};
        }}
        .stApp header {{ background: transparent; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(translator: I18N) -> None:
    workspace = state.get_workspace()
    st.sidebar.header(translator.t("workspace.title"))
    current_path = str(workspace.path)
    new_path = st.sidebar.text_input(
        translator.t("workspace.path_label"), value=current_path
    )
    if new_path and new_path != current_path:
        state.set_workspace(Path(new_path))
        st.sidebar.success(translator.t("messages.workspace_changed"))

    uploaded = st.sidebar.file_uploader(
        translator.t("workspace.load_file"), type=["yaml", "yml"], key="preset_upload"
    )
    if uploaded is not None:
        text = uploaded.read().decode("utf-8")
        state.set_preset_yaml(text)
        state.get_preset()
        st.sidebar.info(translator.t("messages.preset_applied"))

    preset_yaml = st.sidebar.text_area(
        translator.t("workspace.preset_label"),
        value=state.get_preset_yaml(),
        height=220,
    )
    if st.sidebar.button(translator.t("workspace.apply_preset")):
        state.set_preset_yaml(preset_yaml)
        state.get_preset()
        st.sidebar.success(translator.t("messages.preset_applied"))
    if st.sidebar.button(translator.t("workspace.restore")):
        state.reset_preset()
        st.sidebar.success(translator.t("messages.preset_applied"))

    st.sidebar.subheader(translator.t("workspace.theme_section"))
    theme_files = [
        path.stem
        for path in THEMES_ROOT.glob("*.yaml")
        if path.stem not in {"base", "dark"}
    ]
    options = ["base", "dark", *theme_files, "upload"]
    current_choice = st.session_state.get("theme_choice", "base")
    choice = st.sidebar.selectbox(
        translator.t("workspace.theme_label"),
        options=options,
        index=options.index(current_choice) if current_choice in options else 0,
        key="theme_choice",
    )
    override: dict | None = None
    dark_mode = st.session_state.get("dark_mode", False)
    if choice == "dark":
        dark_mode = True
    elif choice == "base":
        dark_mode = False
    if choice not in {"base", "dark", "upload"}:
        theme_path = THEMES_ROOT / f"{choice}.yaml"
        if theme_path.exists():
            override = load_theme_yaml(theme_path)
    elif choice == "upload":
        custom_upload = st.sidebar.file_uploader(
            translator.t("workspace.theme_upload"),
            type=["yaml", "yml"],
            key="theme_upload_file",
        )
        if custom_upload is not None:
            content = custom_upload.read().decode("utf-8")
            tmp = state.get_workspace().resolve("tmp_theme.yaml")
            tmp.write_text(content, encoding="utf-8")
            override = load_theme_yaml(tmp)
        else:
            override = st.session_state.get("theme_override")
    st.session_state["theme_override"] = override
    st.session_state["dark_mode"] = dark_mode

    st.sidebar.subheader(translator.t("workspace.profile_section"))
    profile_options = list(state.available_profiles().keys())
    if profile_options:
        active_profile = state.get_profile_name()
        selected_profile = st.sidebar.selectbox(
            translator.t("workspace.profile_label"),
            options=profile_options,
            index=(
                profile_options.index(active_profile)
                if active_profile in profile_options
                else 0
            ),
            key="profile_select",
        )
        if selected_profile != active_profile:
            state.set_profile(selected_profile)
        preview = state.profile_preview(selected_profile)
        if preview:
            st.sidebar.caption(translator.t("workspace.profile_preview"))
            st.sidebar.json(preview)
    else:  # pragma: no cover - defensive fallback
        st.sidebar.info("No profiles available")

    st.sidebar.subheader(translator.t("workspace.telemetry_section"))
    opt_in_default = st.session_state.get("telemetry_enabled", telemetry.is_enabled())
    enabled = st.sidebar.checkbox(
        translator.t("workspace.telemetry_label"),
        value=bool(opt_in_default),
        key="telemetry_toggle",
    )
    telemetry.set_opt_in(enabled)

    tail = state.workspace_log_tail()
    if tail:
        st.sidebar.caption(translator.t("workspace.log_tail"))
        st.sidebar.code("\n".join(tail), language="json")

    artifacts = state.list_artifacts()
    if artifacts:
        st.sidebar.caption("Artifacts")
        for item in artifacts:
            if item.path.exists() and item.path.is_file():
                st.sidebar.download_button(
                    label=f"{item.key}",
                    data=item.path.read_bytes(),
                    file_name=item.path.name,
                    key=f"dl_{item.key}",
                )
    else:
        st.sidebar.warning(translator.t("messages.no_artifacts"))


def render_shell(page_fn: Callable[[], None], *, title: str, dark: bool) -> bool:
    """Render the shared layout and execute ``page_fn`` within it.

    Returns
    -------
    bool
        ``True`` when the user requested a theme toggle.
    """

    state.ensure_session()
    st.session_state.setdefault("i18n", I18N())
    translator: I18N = st.session_state["i18n"]
    st.session_state["dark_mode"] = dark
    _apply_theme(dark)

    _render_sidebar(translator)

    st.title(title)
    st.caption(translator.t("app.subtitle"))
    cols = st.columns([6, 2, 2])
    with cols[1]:
        zip_artifact = state.get_artifact("session_zip")
        if zip_artifact and zip_artifact.exists():
            st.download_button(
                label=translator.t("actions.export"),
                data=zip_artifact.read_bytes(),
                file_name=zip_artifact.name,
                key="export_zip",
            )
    toggle = False
    with cols[2]:
        label = (
            translator.t("actions.toggle_light")
            if dark
            else translator.t("actions.toggle_dark")
        )
        if st.button(label, key="toggle_theme"):
            toggle = True
            st.session_state["theme_choice"] = "dark" if not dark else "base"
    page_fn()
    return toggle
