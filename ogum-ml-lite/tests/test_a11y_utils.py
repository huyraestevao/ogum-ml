"""Tests for accessibility helper functions."""

from __future__ import annotations

from app.design import a11y


def test_focus_hint_default() -> None:
    hint = a11y.focus_hint()
    assert str(hint) == "Use Tab/Shift+Tab to navigate."


def test_focus_hint_custom_message() -> None:
    hint = a11y.focus_hint("Press Enter to confirm.")
    assert str(hint) == "Press Enter to confirm."


def test_aria_label_injects_help() -> None:
    payload = a11y.aria_label("Run action")
    assert payload == {"help": "Run action"}


def test_describe_chart_handles_empty_summary() -> None:
    assert a11y.describe_chart(" ") == "📈 Chart summary unavailable"


def test_describe_chart_custom_prefix() -> None:
    assert a11y.describe_chart("theta curve", prefix="θ") == "θ theta curve"
