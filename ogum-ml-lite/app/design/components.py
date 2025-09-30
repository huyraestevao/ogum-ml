"""Reusable Streamlit UI components used across the dashboard."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from .theme import get_theme


def card(title: str, body: Any, help: str | None = None) -> DeltaGenerator:
    """Render a card container with consistent styling.

    Parameters
    ----------
    title:
        Card title rendered in bold text.
    body:
        Body content to render. Objects are passed directly to ``st.write``.
    help:
        Optional tooltip text appended to the title.

    Returns
    -------
    streamlit.delta_generator.DeltaGenerator
        Container reference so callers may append extra elements if needed.
    """

    theme = get_theme(st.session_state.get("dark_mode", False))
    container = st.container()
    with container:
        st.markdown(
            '<div style="'
            f"background:{theme['colors']['surface']};"
            f" padding:{theme['space']['md']}px;"
            f" border-radius:{theme['radii']['md']}px;"
            f" border:1px solid {theme['colors']['border']};"
            '">',
            unsafe_allow_html=True,
        )
        title_style = theme["typography"]["subtitle_size"]
        title_html = (
            "<div style='font-weight:600;font-size:" f"{title_style}'>" f"{title}</div>"
        )
        st.markdown(title_html, unsafe_allow_html=True)
        if help:
            st.caption(help)
        st.write(body)
        st.markdown("</div>", unsafe_allow_html=True)
    return container


def alert(kind: Literal["info", "success", "warn", "error"], text: str) -> None:
    """Render a semantic alert with high-contrast colours.

    Parameters
    ----------
    kind:
        Alert type. Maps to Streamlit's status elements.
    text:
        Message to display.
    """

    mapping = {
        "info": st.info,
        "success": st.success,
        "warn": st.warning,
        "error": st.error,
    }
    mapping.get(kind, st.info)(text)


def toolbar(actions: Sequence[tuple[str, str]]) -> str | None:
    """Render a compact toolbar of buttons.

    Parameters
    ----------
    actions:
        Iterable of ``(label, callback_key)`` tuples. The callback key is stored
        in ``st.session_state`` when the button is activated.

    Returns
    -------
    str or None
        The key corresponding to the pressed button during the current rerun.
    """

    if not actions:
        return None

    cols = st.columns(len(actions))
    for idx, (label, key) in enumerate(actions):
        if cols[idx].button(label, key=f"toolbar_{key}"):
            st.session_state["toolbar_last_action"] = key
            return key
    return None
