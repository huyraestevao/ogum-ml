"""Main entry point for the Streamlit dashboard."""

from __future__ import annotations

from typing import Callable

import streamlit as st

from app.design.layout import render_shell
from app.i18n.translate import I18N
from app.pages import (
    page_export,
    page_features,
    page_mechanism,
    page_ml,
    page_msc,
    page_prep,
    page_segments,
)
from app.services import state

PAGES: dict[str, tuple[str, Callable[[I18N], None]]] = {
    "prep": ("menu.prep", page_prep.render),
    "features": ("menu.features", page_features.render),
    "msc": ("menu.msc", page_msc.render),
    "segments": ("menu.segments", page_segments.render),
    "mechanism": ("menu.mechanism", page_mechanism.render),
    "ml": ("menu.ml", page_ml.render),
    "export": ("menu.export", page_export.render),
}


def _select_page(translator: I18N) -> str:
    options = list(PAGES.keys())
    selected = st.sidebar.radio(
        "Menu",
        options=options,
        format_func=lambda key: translator.t(PAGES[key][0]),
        key="main_menu",
    )
    return selected


def _select_locale() -> str:
    return st.sidebar.selectbox("Idioma", options=["pt", "en"], key="locale_select")


def main() -> None:
    st.set_page_config(page_title="Ogum-ML", layout="wide")
    state.ensure_session()
    translator: I18N = st.session_state.setdefault("i18n", I18N())

    locale = _select_locale()
    st.session_state["locale"] = locale
    selected_page = _select_page(translator)

    dark_mode = st.session_state.get("dark_mode", False)

    def _page_runner() -> None:
        translator = st.session_state["i18n"]
        _, page_fn = PAGES[selected_page]
        page_fn(translator)

    toggle = render_shell(_page_runner, title=translator.t("app.title"), dark=dark_mode)
    if toggle:
        st.session_state["dark_mode"] = not dark_mode
        st.experimental_rerun()


if __name__ == "__main__":  # pragma: no cover - manual run
    main()
