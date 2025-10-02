"""Helpers for loading and merging theme definitions."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

THEME_KEYS = {"colors", "space", "radii", "typography"}


def _validate_theme(theme: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in theme.items():
        if key not in THEME_KEYS:
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"Theme key '{key}' must be a mapping")
        payload[key] = dict(value)
    return payload


def load_theme_yaml(path: Path) -> dict[str, Any]:
    """Load and validate a theme YAML file."""

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError("Theme YAML must contain a mapping")
    return _validate_theme(data)


def merge_theme(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Deep merge ``override`` into ``base`` returning a new mapping."""

    merged = deepcopy(dict(base))
    for key, value in override.items():
        if key not in THEME_KEYS:
            continue
        if isinstance(value, Mapping):
            merged.setdefault(key, {})
            child = dict(merged[key])
            child.update(value)
            merged[key] = child
        else:
            merged[key] = value
    return merged


__all__ = ["load_theme_yaml", "merge_theme"]
