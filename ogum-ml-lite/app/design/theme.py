"""Streamlit design system definitions for Ogum ML Lite."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

THEME: Dict[str, Any] = {
    "colors": {
        "background": "#f7f9fc",
        "surface": "#ffffff",
        "text": "#1d1f24",
        "muted": "#5f6b7a",
        "primary": "#2453A6",
        "success": "#2d8a34",
        "warning": "#b36b00",
        "danger": "#ba1a1a",
        "border": "#d5dbe5",
    },
    "typography": {
        "family": "'Inter', 'Segoe UI', sans-serif",
        "title_size": "1.75rem",
        "subtitle_size": "1.1rem",
        "body_size": "0.95rem",
    },
    "radii": {"sm": 4, "md": 8, "lg": 16},
    "space": {"xs": 4, "sm": 6, "md": 12, "lg": 20, "xl": 32},
    "dark": {
        "colors": {
            "background": "#12151c",
            "surface": "#1b1f29",
            "text": "#f4f6fb",
            "muted": "#9aa4b5",
            "primary": "#88a8f5",
            "success": "#7dd48a",
            "warning": "#ffd084",
            "danger": "#ffb4ab",
            "border": "#2c3240",
        }
    },
}


def get_theme(dark: bool = False) -> Dict[str, Any]:
    """Return the merged theme for the given mode.

    Parameters
    ----------
    dark:
        Whether the dark palette should be used.

    Returns
    -------
    dict
        Deep copy of the theme dictionary, with the appropriate palette applied.
    """

    base = deepcopy({key: value for key, value in THEME.items() if key != "dark"})
    if dark:
        dark_overrides = THEME.get("dark", {})
        for key, value in dark_overrides.items():
            if isinstance(value, dict):
                base.setdefault(key, {})
                base[key].update(value)
            else:  # pragma: no cover - future proofing
                base[key] = deepcopy(value)
    return base
