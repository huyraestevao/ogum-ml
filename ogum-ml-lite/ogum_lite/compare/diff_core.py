"""Semantic diff helpers for Ogum-ML artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .loaders import load_csv, load_json, load_yaml


@dataclass(slots=True)
class MetricDelta:
    """Container describing the delta between metrics."""

    metric: str
    a: float | None
    b: float | None
    delta: float | None
    delta_pct: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "a": self.a,
            "b": self.b,
            "delta": self.delta,
            "delta_pct": self.delta_pct,
        }


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def diff_presets(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any]:
    """Compute differences between two preset dictionaries."""

    result: dict[str, Any] = {
        "missing": {"a": a is None, "b": b is None},
        "added": {},
        "removed": {},
        "changed": {},
        "lists": {},
    }
    if a is None or b is None:
        return result

    keys_a = set(a.keys())
    keys_b = set(b.keys())
    for key in sorted(keys_b - keys_a):
        result["added"][key] = b[key]
    for key in sorted(keys_a - keys_b):
        result["removed"][key] = a[key]

    for key in sorted(keys_a & keys_b):
        left = a[key]
        right = b[key]
        if isinstance(left, list) and isinstance(right, list):
            left_set = list(dict.fromkeys(left))
            right_set = list(dict.fromkeys(right))
            result["lists"][key] = {
                "only_a": [item for item in left_set if item not in right_set],
                "only_b": [item for item in right_set if item not in left_set],
                "shared": [item for item in left_set if item in right_set],
            }
            if result["lists"][key]["only_a"] or result["lists"][key]["only_b"]:
                result["changed"][key] = {"a": left, "b": right}
            continue
        if left != right:
            result["changed"][key] = {"a": left, "b": right}
    return result


def _extract_metric(row: pd.Series, candidates: Iterable[str]) -> float | None:
    for name in candidates:
        if name in row:
            value = _safe_float(row[name])
            if value is not None:
                return value
    return None


def diff_msc(a_csv: Path | None, b_csv: Path | None) -> dict[str, Any]:
    """Compare MSC summaries produced by two runs."""

    df_a = load_csv(a_csv)
    df_b = load_csv(b_csv)
    result: dict[str, Any] = {
        "missing": {"a": df_a is None, "b": df_b is None},
        "metrics": {},
        "curve": None,
    }
    if df_a is None or df_b is None:
        return result

    row_a = df_a.iloc[0]
    row_b = df_b.iloc[0]

    for metric, aliases in {
        "mse_global": ["mse_global", "mse"],
        "mse_segmented": ["mse_segmented", "mse_seg"],
    }.items():
        value_a = _extract_metric(row_a, aliases)
        value_b = _extract_metric(row_b, aliases)
        delta = None
        delta_pct = None
        if value_a is not None and value_b is not None:
            delta = value_b - value_a
            if value_a != 0:
                delta_pct = delta / value_a
        result["metrics"][metric] = MetricDelta(
            metric=metric,
            a=value_a,
            b=value_b,
            delta=delta,
            delta_pct=delta_pct,
        ).to_dict()

    if {
        "theta_norm",
        "prediction",
        "target",
    }.issubset(df_a.columns) and {
        "theta_norm",
        "prediction",
        "target",
    }.issubset(df_b.columns):
        merged = df_a.merge(df_b, on="theta_norm", suffixes=("_a", "_b"))
        if not merged.empty:
            merged["delta"] = (merged["prediction_b"] - merged["prediction_a"]).abs()
            result["curve"] = {
                "mean_abs_delta": float(merged["delta"].mean()),
                "n_points": int(len(merged)),
            }
    return result


def diff_segments(a_table: Path | None, b_table: Path | None) -> dict[str, Any]:
    """Compare per-segment metrics."""

    df_a = load_csv(a_table)
    df_b = load_csv(b_table)
    result: dict[str, Any] = {
        "missing": {"a": df_a is None, "b": df_b is None},
        "changed": {},
    }
    if df_a is None or df_b is None:
        return result

    key_cols = [
        col
        for col in ("sample_id", "segment_id")
        if col in df_a.columns and col in df_b.columns
    ]
    if not key_cols:
        return result
    merged = df_a.merge(df_b, on=key_cols, suffixes=("_a", "_b"))
    metrics = ["n_est", "mse", "r2"]
    for _, row in merged.iterrows():
        key = tuple(row[col] for col in key_cols)
        entry: dict[str, Any] = {}
        for metric in metrics:
            col_a = f"{metric}_a"
            col_b = f"{metric}_b"
            if col_a in row and col_b in row:
                value_a = _safe_float(row[col_a])
                value_b = _safe_float(row[col_b])
                if value_a != value_b:
                    entry[metric] = {
                        "a": value_a,
                        "b": value_b,
                        "delta": (
                            None
                            if value_a is None or value_b is None
                            else value_b - value_a
                        ),
                    }
        if entry:
            result["changed"][key] = entry
    return result


def diff_mechanism(a_mech: Path | None, b_mech: Path | None) -> dict[str, Any]:
    """Compare mechanism change detection outputs."""

    df_a = load_csv(a_mech)
    df_b = load_csv(b_mech)
    result: dict[str, Any] = {
        "missing": {"a": df_a is None, "b": df_b is None},
        "changed": {},
    }
    if df_a is None or df_b is None:
        return result

    key_cols = [
        col for col in ("sample_id",) if col in df_a.columns and col in df_b.columns
    ]
    if not key_cols:
        return result
    merged = df_a.merge(df_b, on=key_cols, suffixes=("_a", "_b"))
    metrics = ["has_change", "tau", "bic", "aic"]
    for _, row in merged.iterrows():
        key = tuple(row[col] for col in key_cols)
        entry: dict[str, Any] = {}
        for metric in metrics:
            col_a = f"{metric}_a"
            col_b = f"{metric}_b"
            if col_a in row and col_b in row:
                value_a = row[col_a]
                value_b = row[col_b]
                if value_a != value_b:
                    entry[metric] = {
                        "a": value_a,
                        "b": value_b,
                    }
        if entry:
            result["changed"][key] = entry
    return result


def diff_ml(
    a_model_card: Path | None,
    b_model_card: Path | None,
    a_cv: Path | None,
    b_cv: Path | None,
) -> dict[str, Any]:
    """Compare ML model metadata and metrics."""

    card_a = load_json(a_model_card) or load_yaml(a_model_card)
    card_b = load_json(b_model_card) or load_yaml(b_model_card)
    cv_a = load_json(a_cv)
    cv_b = load_json(b_cv)

    result: dict[str, Any] = {
        "missing": {
            "card_a": card_a is None,
            "card_b": card_b is None,
            "cv_a": cv_a is None,
            "cv_b": cv_b is None,
        },
        "algorithm": None,
        "hyperparameters": {},
        "metrics": {},
    }

    if card_a and card_b:
        algo_a = card_a.get("algorithm") or card_a.get("estimator")
        algo_b = card_b.get("algorithm") or card_b.get("estimator")
        if algo_a != algo_b:
            result["algorithm"] = {"a": algo_a, "b": algo_b}
        params_a = card_a.get("hyperparameters", {}) or {}
        params_b = card_b.get("hyperparameters", {}) or {}
        keys = set(params_a) | set(params_b)
        for key in sorted(keys):
            if params_a.get(key) != params_b.get(key):
                result["hyperparameters"][key] = {
                    "a": params_a.get(key),
                    "b": params_b.get(key),
                }

    metric_aliases = {
        "accuracy": ["accuracy", "acc"],
        "f1": ["f1", "f1_score"],
        "mae": ["mae", "mean_absolute_error"],
        "rmse": ["rmse", "root_mean_squared_error", "mse"],
    }
    if isinstance(cv_a, dict) and isinstance(cv_b, dict):
        for metric, aliases in metric_aliases.items():
            value_a = None
            value_b = None
            for alias in aliases:
                if alias in cv_a:
                    value_a = _safe_float(cv_a[alias])
                if alias in cv_b:
                    value_b = _safe_float(cv_b[alias])
            if value_a is not None or value_b is not None:
                delta = None
                delta_pct = None
                if value_a is not None and value_b is not None:
                    delta = value_b - value_a
                    if value_a != 0:
                        delta_pct = delta / value_a
                result["metrics"][metric] = MetricDelta(
                    metric=metric,
                    a=value_a,
                    b=value_b,
                    delta=delta,
                    delta_pct=delta_pct,
                ).to_dict()
    return result


def compose_diff_summary(
    *,
    presets: dict[str, Any] | None = None,
    msc: dict[str, Any] | None = None,
    segments: dict[str, Any] | None = None,
    mechanism: dict[str, Any] | None = None,
    ml: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a consolidated summary of the diffs."""

    summary: dict[str, Any] = {"kpis": []}

    for container in (msc or {}).get("metrics", {}).values():
        summary["kpis"].append(container)

    for container in (ml or {}).get("metrics", {}).values():
        summary["kpis"].append(container)

    summary["kpis"].sort(key=lambda item: item.get("metric", ""))

    summary["alerts"] = []
    if presets:
        if presets.get("added") or presets.get("removed"):
            summary["alerts"].append("Preset keys diverged")
    if segments and segments.get("changed"):
        summary["alerts"].append("Segments changed")
    if mechanism and mechanism.get("changed"):
        summary["alerts"].append("Mechanism detection differs")

    return summary


__all__ = [
    "diff_presets",
    "diff_msc",
    "diff_segments",
    "diff_mechanism",
    "diff_ml",
    "compose_diff_summary",
]
