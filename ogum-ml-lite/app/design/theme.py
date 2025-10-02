"""Theme loader merging YAML customisations with defaults."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from app.services.themes_loader import load_theme_yaml, merge_theme

THEMES_ROOT = Path(__file__).resolve().parents[1] / "config" / "themes"


@lru_cache(maxsize=4)
def _base_theme() -> dict[str, Any]:
    return load_theme_yaml(THEMES_ROOT / "base.yaml")


@lru_cache(maxsize=4)
def _dark_overrides() -> dict[str, Any]:
    return load_theme_yaml(THEMES_ROOT / "dark.yaml")


def get_theme(
    dark: bool = False,
    override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the merged theme for the given mode."""

    theme = dict(_base_theme())
    if dark:
        theme = merge_theme(theme, _dark_overrides())
    if override:
        theme = merge_theme(theme, override)
    return theme


__all__ = ["get_theme"]
