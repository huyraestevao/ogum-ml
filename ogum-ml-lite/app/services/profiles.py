"""Helpers to load and apply execution profiles."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

PROFILE_KEYS = {
    "columns",
    "msc",
    "features",
    "ml",
    "segmentation",
    "electric_features",
}


def _profile_root(root: Path | None = None) -> Path:
    base = Path(__file__).resolve().parents[1] / "config" / "profiles"
    if root is not None:
        base = Path(root)
    return base


def load_profile(name: str, *, root: Path | None = None) -> dict[str, Any]:
    """Load a profile from the configured directory."""

    directory = _profile_root(root)
    path = directory / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Profile '{name}' not found in {directory}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError("Profile YAML must contain a mapping")
    return _validate_profile(data)


def list_profiles(*, root: Path | None = None) -> dict[str, Path]:
    """Return available profile names and their paths."""

    directory = _profile_root(root)
    return {
        path.stem: path for path in sorted(directory.glob("*.yaml")) if path.is_file()
    }


def _validate_profile(data: Mapping[str, Any]) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    for key, value in data.items():
        if key not in PROFILE_KEYS:
            continue
        if isinstance(value, Mapping):
            profile[key] = dict(value)
        else:
            profile[key] = value
    return profile


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        elif value is not None:
            merged[key] = value
    return merged


def apply_profile(
    preset: Mapping[str, Any], profile: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge ``profile`` overrides into ``preset`` returning a copy."""

    allowed = _validate_profile(profile)
    base = deepcopy(dict(preset))
    return _deep_merge(base, allowed)


__all__ = ["apply_profile", "list_profiles", "load_profile"]
