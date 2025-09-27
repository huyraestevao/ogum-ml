"""Preset management helpers for the Ogum Lite UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

PresetDict = Dict[str, Any]


def load_presets(path: Path) -> PresetDict:
    """Load presets from a YAML file.

    Parameters
    ----------
    path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed preset dictionary. Returns an empty dictionary when the file
        does not exist.
    """

    path = Path(path)
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):  # pragma: no cover - defensive
        raise TypeError("Preset file must contain a mapping at the top level")
    return data


def save_presets(preset: PresetDict, path: Path) -> None:
    """Persist a preset dictionary to disk as YAML."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(preset, sort_keys=False, allow_unicode=True))


def merge_presets(base: PresetDict, override: PresetDict) -> PresetDict:
    """Deep-merge two preset dictionaries.

    ``override`` values take precedence over ``base``. Nested dictionaries are
    merged recursively while other objects are replaced.
    """

    def _merge(a: Any, b: Any) -> Any:
        if isinstance(a, dict) and isinstance(b, dict):
            result: dict[str, Any] = {}
            keys = set(a) | set(b)
            for key in keys:
                if key in a and key in b:
                    result[key] = _merge(a[key], b[key])
                elif key in a:
                    result[key] = a[key]
                else:
                    result[key] = b[key]
            return result
        return b

    merged = _merge(dict(base), dict(override)) if override else dict(base)
    return merged
