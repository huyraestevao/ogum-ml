"""Lightweight i18n helper for the dashboard."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import streamlit as st

LOCALES_PATH = Path(__file__).with_suffix("").parent / "locales"
DEFAULT_LOCALE = "pt"


def _load_locale(locale: str) -> Mapping[str, Any]:
    path = LOCALES_PATH / f"{locale}.json"
    if not path.exists():
        raise FileNotFoundError(f"Locale file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def _catalogue(locale: str) -> Mapping[str, Any]:
    return _load_locale(locale)


def _lookup(catalog: Mapping[str, Any], key: str) -> str | None:
    cursor: Any = catalog
    for piece in key.split("."):
        if not isinstance(cursor, Mapping):
            return None
        cursor = cursor.get(piece)
        if cursor is None:
            return None
    if isinstance(cursor, str):
        return cursor
    return None


class I18N:
    """Translation helper with session-aware locale selection."""

    def __init__(self, locale: str = DEFAULT_LOCALE) -> None:
        self.default_locale = locale

    @property
    def locale(self) -> str:
        return st.session_state.get("locale", self.default_locale)

    def t(self, key: str, **kwargs: Any) -> str:
        """Return the translated text for ``key`` with optional formatting."""

        for candidate in (self.locale, self.default_locale):
            try:
                catalog = _catalogue(candidate)
            except FileNotFoundError:
                continue
            message = _lookup(catalog, key)
            if message:
                return message.format(**kwargs)
        return key
