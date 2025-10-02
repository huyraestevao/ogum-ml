"""Discovery and parsing helpers for Ogum-ML run artifacts."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


def _extract_zip(source: Path) -> Path:
    """Extract *source* zip file into a temporary directory and return the root."""

    temp_dir = Path(tempfile.mkdtemp(prefix="ogum_compare_"))
    with zipfile.ZipFile(source) as zf:
        zf.extractall(temp_dir)
    # Attempt to identify a single root folder inside the archive; otherwise
    # use temp_dir.
    candidates = [p for p in temp_dir.iterdir() if p.is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    return temp_dir


def find_run_root(path: Path) -> Path:
    """Return the directory that contains a run manifest.

    Parameters
    ----------
    path:
        Directory or zip file pointing to an Ogum-ML run.
    """

    path = Path(path)
    if path.is_dir():
        return path
    if path.is_file() and path.suffix.lower() == ".zip":
        return _extract_zip(path)
    raise FileNotFoundError(f"Unsupported run path: {path}")


def _first_existing(root: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    for name in names:
        matches = list(root.rglob(name))
        if matches:
            return matches[0]
    return None


def scan_run(root: Path) -> dict[str, Any]:
    """Return a manifest for the run located at *root*.

    The manifest contains best-effort pointers to relevant artifacts.
    Missing entries are reported as ``None`` so callers can handle
    incomplete runs gracefully.
    """

    root = Path(root)
    manifest: dict[str, Any] = {"root": root}

    manifest["presets"] = _first_existing(root, ["preset.yaml", "presets.yaml"])

    manifest["msc_csv"] = _first_existing(root, ["msc.csv"])
    manifest["msc_plot"] = _first_existing(root, ["msc.png", "msc_plot.png"])
    manifest["msc_theta_zip"] = _first_existing(root, ["theta.zip", "theta_csv.zip"])

    manifest["segments_json"] = _first_existing(root, ["segments.json"])
    manifest["segments_table"] = _first_existing(
        root, ["n_segments.csv", "segments.csv"]
    )

    manifest["mechanism_csv"] = _first_existing(
        root, ["mechanism.csv", "mech_report.csv"]
    )

    manifest["ml_model_card"] = _first_existing(root, ["model_card.json"])
    manifest["ml_cv_metrics"] = _first_existing(root, ["cv_metrics.json"])
    manifest["ml_features"] = _first_existing(root, ["feature_cols.json"])
    manifest["ml_classifier"] = _first_existing(root, ["classifier.joblib"])
    manifest["ml_regressor"] = _first_existing(root, ["regressor.joblib"])
    manifest["ml_confusion"] = _first_existing(root, ["confusion.png"])
    manifest["ml_scatter"] = _first_existing(root, ["scatter.png"])

    manifest["summary_html"] = _first_existing(root, ["report.html", "summary.html"])
    manifest["summary_xlsx"] = _first_existing(root, ["report.xlsx", "summary.xlsx"])

    manifest["run_log"] = _first_existing(root, ["run_log.jsonl"])
    manifest["telemetry"] = _first_existing(root, ["telemetry.jsonl"])

    return manifest


def load_json(path: Path | None) -> dict[str, Any] | None:
    """Read a JSON file if available."""

    if path is None or not Path(path).exists():
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path | None) -> dict[str, Any] | None:
    """Read a YAML file if available."""

    if path is None or not Path(path).exists():
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Expected YAML document to be a mapping")
    return data


def load_csv(path: Path | None) -> pd.DataFrame | None:
    """Read a CSV file into a :class:`pandas.DataFrame` if the file exists."""

    if path is None or not Path(path).exists():
        return None
    return pd.read_csv(path)


__all__ = [
    "find_run_root",
    "scan_run",
    "load_json",
    "load_yaml",
    "load_csv",
]
