"""Tests for the i18n helper."""

from __future__ import annotations

import streamlit as st
from app.i18n.translate import I18N


def test_translation_lookup_and_fallback() -> None:
    st.session_state.clear()
    st.session_state["locale"] = "en"
    helper = I18N()
    assert helper.t("actions.run") == "Run"
    st.session_state["locale"] = "xx"
    assert helper.t("actions.run") == "Executar"


def test_interpolation() -> None:
    st.session_state.clear()
    helper = I18N("pt")
    st.session_state["locale"] = "en"
    assert helper.t("messages.ready") == "[ok] ready"
