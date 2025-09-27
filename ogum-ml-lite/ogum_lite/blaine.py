"""Blaine linearisation utilities for sintering segments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .segmentation import Segment


@dataclass
class BlaineResult:
    sample_id: str | int | float | None
    segment_index: int
    method: str
    n_points: int
    n: float
    slope: float
    intercept: float
    r2: float
    mse: float
    start_time_s: float
    end_time_s: float
    start_y: float
    end_y: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "sample_id": self.sample_id,
            "segment_index": self.segment_index,
            "method": self.method,
            "n_points": self.n_points,
            "n": self.n,
            "slope": self.slope,
            "intercept": self.intercept,
            "r2": self.r2,
            "mse": self.mse,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "start_y": self.start_y,
            "end_y": self.end_y,
        }


def _linearise_blaine(time_s: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    mask = (time_s > 0) & (y > 0) & (y < 1)
    if mask.sum() < 2:
        return (float("nan"),) * 4

    log_t = np.log(time_s[mask])
    log_blaine = np.log(1.0 / (1.0 - y[mask]))
    A = np.vstack([log_t, np.ones_like(log_t)]).T
    coeffs, *_ = np.linalg.lstsq(A, log_blaine, rcond=None)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])
    predictions = slope * log_t + intercept
    residuals = log_blaine - predictions
    sse = float(np.sum(residuals**2))
    mse = sse / log_t.size
    sst = float(np.sum((log_blaine - np.mean(log_blaine)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    return slope, intercept, r2, mse


def fit_blaine_segment(time_s: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    time_arr = np.asarray(time_s, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    slope, intercept, r2, mse = _linearise_blaine(time_arr, y_arr)
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "mse": mse,
        "n": slope,
    }


def fit_blaine_by_segments(
    df: pd.DataFrame,
    segments: Iterable[Segment],
    *,
    t_col: str = "time_s",
    y_col: str = "rho_rel",
) -> list[BlaineResult]:
    results: list[BlaineResult] = []
    for segment in segments:
        subset = df.loc[segment.indices].sort_values(t_col)
        if subset.empty:
            slope = intercept = r2 = mse = float("nan")
            n_value = float("nan")
        else:
            stats = fit_blaine_segment(subset[t_col], subset[y_col])
            slope = stats["slope"]
            intercept = stats["intercept"]
            r2 = stats["r2"]
            mse = stats["mse"]
            n_value = stats["n"]

        results.append(
            BlaineResult(
                sample_id=segment.sample_id,
                segment_index=segment.segment_index,
                method=segment.method,
                n_points=segment.n_points,
                n=n_value,
                slope=slope,
                intercept=intercept,
                r2=r2,
                mse=mse,
                start_time_s=segment.start_time_s,
                end_time_s=segment.end_time_s,
                start_y=segment.start_y,
                end_y=segment.end_y,
            )
        )
    return results


__all__ = ["BlaineResult", "fit_blaine_segment", "fit_blaine_by_segments"]
