"""Mechanism change detection using piecewise linear models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .segmentation import PiecewiseLinearModel, fit_piecewise_linear


@dataclass
class MechanismModel:
    n_segments: int
    breakpoints: list[int]
    total_sse: float
    slopes: list[float]
    intercepts: list[float]
    n_points: int
    criterion: float
    baseline_criterion: float
    change_detected: bool
    breakpoint_theta: float | None
    breakpoint_densification: float | None

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "n_segments": self.n_segments,
            "breakpoints": self.breakpoints,
            "total_sse": self.total_sse,
            "slopes": self.slopes,
            "intercepts": self.intercepts,
            "n_points": self.n_points,
            "criterion": self.criterion,
            "baseline_criterion": self.baseline_criterion,
            "change_detected": self.change_detected,
            "breakpoint_theta": self.breakpoint_theta,
            "breakpoint_densification": self.breakpoint_densification,
        }


def _model_from_piecewise(
    model: PiecewiseLinearModel,
    *,
    baseline: float,
    theta: np.ndarray,
    densification: np.ndarray,
    threshold: float,
    criterion: str,
    slope_delta_threshold: float,
) -> MechanismModel:
    slopes = [segment.slope for segment in model.segments]
    intercepts = [segment.intercept for segment in model.segments]
    crit = model.information_criterion(criterion)
    slope_delta = max(np.abs(np.diff(slopes))) if len(slopes) > 1 else 0.0
    change_detected = (
        model.n_segments > 1
        and (baseline - crit) > threshold
        and slope_delta >= slope_delta_threshold
    )

    breakpoint_theta = None
    breakpoint_densification = None
    if change_detected:
        first_break = model.breakpoints[0]
        breakpoint_theta = float(theta[first_break - 1])
        breakpoint_densification = float(densification[first_break - 1])

    return MechanismModel(
        n_segments=model.n_segments,
        breakpoints=model.breakpoints,
        total_sse=model.total_sse,
        slopes=slopes,
        intercepts=intercepts,
        n_points=model.n_points,
        criterion=crit,
        baseline_criterion=baseline,
        change_detected=change_detected,
        breakpoint_theta=breakpoint_theta,
        breakpoint_densification=breakpoint_densification,
    )


def detect_mechanism_change(
    df: pd.DataFrame,
    *,
    group_col: str = "sample_id",
    theta_col: str = "theta",
    y_col: str = "densification",
    max_segments: int = 2,
    min_size: int = 5,
    criterion: str = "bic",
    threshold: float = 2.0,
    slope_delta: float = 0.02,
) -> pd.DataFrame:
    """Detect mechanism changes by comparing piecewise linear fits."""

    if theta_col not in df.columns:
        raise KeyError(f"Column '{theta_col}' not present in dataframe")
    if y_col not in df.columns:
        raise KeyError(f"Column '{y_col}' not present in dataframe")

    groups: Iterable[tuple[str | None, pd.DataFrame]]
    if group_col in df.columns:
        groups = df.groupby(group_col)
    else:
        groups = [(None, df)]

    records: list[dict[str, float | int | bool | None | str]] = []

    for sample_id, group in groups:
        ordered = group.sort_values(theta_col)
        theta_values = ordered[theta_col].to_numpy(dtype=float)
        densification = ordered[y_col].to_numpy(dtype=float)

        if theta_values.size < min_size:
            records.append(
                {
                    group_col: sample_id,
                    "change_detected": False,
                    "n_segments": 1,
                    "criterion": float("nan"),
                    "baseline_criterion": float("nan"),
                    "breakpoint_theta": float("nan"),
                    "breakpoint_densification": float("nan"),
                }
            )
            continue

        base_model = fit_piecewise_linear(theta_values, densification, 1, min_size=min_size)
        if base_model is None:
            records.append(
                {
                    group_col: sample_id,
                    "change_detected": False,
                    "n_segments": 1,
                    "criterion": float("nan"),
                    "baseline_criterion": float("nan"),
                    "breakpoint_theta": float("nan"),
                    "breakpoint_densification": float("nan"),
                }
            )
            continue

        baseline_score = base_model.information_criterion(criterion)
        best_model = _model_from_piecewise(
            base_model,
            baseline=baseline_score,
            theta=theta_values,
            densification=densification,
            threshold=threshold,
            criterion=criterion,
            slope_delta_threshold=slope_delta,
        )

        for n_segments in range(2, max_segments + 1):
            candidate = fit_piecewise_linear(
                theta_values, densification, n_segments, min_size=min_size
            )
            if candidate is None:
                continue
            candidate_model = _model_from_piecewise(
                candidate,
                baseline=baseline_score,
                theta=theta_values,
                densification=densification,
                threshold=threshold,
                criterion=criterion,
                slope_delta_threshold=slope_delta,
            )
            if candidate_model.criterion < best_model.criterion:
                best_model = candidate_model

        if not best_model.change_detected and best_model.n_segments > 1:
            best_model = _model_from_piecewise(
                base_model,
                baseline=baseline_score,
                theta=theta_values,
                densification=densification,
                threshold=threshold,
                criterion=criterion,
                slope_delta_threshold=slope_delta,
            )

        records.append(
            {
                group_col: sample_id,
                "change_detected": best_model.change_detected,
                "n_segments": best_model.n_segments,
                "criterion": best_model.criterion,
                "baseline_criterion": best_model.baseline_criterion,
                "breakpoint_theta": best_model.breakpoint_theta
                if best_model.breakpoint_theta is not None
                else float("nan"),
                "breakpoint_densification": best_model.breakpoint_densification
                if best_model.breakpoint_densification is not None
                else float("nan"),
            }
        )

    return pd.DataFrame.from_records(records)


__all__ = ["MechanismModel", "detect_mechanism_change"]
