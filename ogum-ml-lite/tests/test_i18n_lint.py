"""Tests for translation catalogue linting."""

from __future__ import annotations

from app.services import i18n_lint


def test_locales_have_same_keys() -> None:
    report = i18n_lint.lint()
    assert all(
        not payload["missing"] and not payload["extra"] for payload in report.values()
    )
