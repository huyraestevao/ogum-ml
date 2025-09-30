"""Tests for frontend wiring."""

from __future__ import annotations

from app import streamlit_app


def test_pages_registry_contains_expected_entries() -> None:
    expected = {"prep", "features", "msc", "segments", "mechanism", "ml", "export"}
    assert expected.issubset(streamlit_app.PAGES.keys())
