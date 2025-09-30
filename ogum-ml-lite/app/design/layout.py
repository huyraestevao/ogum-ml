"""Layout primitives for rendering the Streamlit shell."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import streamlit as st

from ..i18n.translate import I18N
from ..services import state
from .theme import get_theme


def _apply_theme(dark: bool) -> None:
    theme = get_theme(dark)
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
        translator.t("workspace.load_file"), type=["yaml", "yml"]
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
    page_fn()
    return toggle
