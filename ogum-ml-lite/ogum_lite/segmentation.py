"""Automatic segmentation utilities for densification curves."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class Segment:
    """Summary of a segmentation interval for a single sample."""

    sample_id: str | int | float | None
    segment_index: int
    method: str
    lower: float
    upper: float
    start_time_s: float
    end_time_s: float
    start_y: float
    end_y: float
    n_points: int
    indices: np.ndarray = field(repr=False)

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "sample_id": self.sample_id,
            "segment_index": self.segment_index,
            "method": self.method,
            "lower": self.lower,
            "upper": self.upper,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "start_y": self.start_y,
            "end_y": self.end_y,
            "n_points": self.n_points,
        }


@dataclass
class LinearSegmentStats:
    start: int
    end: int
    slope: float
    intercept: float
    sse: float
    r2: float
    mse: float


@dataclass
class PiecewiseLinearModel:
    segments: list[LinearSegmentStats]
    breakpoints: list[int]
    total_sse: float
    n_points: int

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def n_parameters(self) -> int:
        # slope + intercept per segment, plus (segments - 1) breakpoints
        return 2 * self.n_segments + max(0, self.n_segments - 1)

    def information_criterion(self, criterion: str = "bic") -> float:
        return compute_information_criterion(
            self.total_sse, self.n_points, self.n_parameters, criterion=criterion
        )


def compute_information_criterion(
    sse: float, n_points: int, n_parameters: int, *, criterion: str = "bic"
) -> float:
    """Compute AIC/BIC-like criteria from sum of squared errors."""

    if n_points <= 0:
        raise ValueError("n_points must be positive")
    if n_parameters <= 0:
        raise ValueError("n_parameters must be positive")
    if criterion not in {"aic", "bic"}:
        raise ValueError("criterion must be 'aic' or 'bic'")

    sse = float(sse)
    if sse <= 0.0:
        # Guard against log(0) when the fit is perfect
        sse = 1e-12
    mse = sse / float(n_points)
    penalty = (
        2 * n_parameters if criterion == "aic" else n_parameters * np.log(n_points)
    )
    return n_points * float(np.log(mse)) + penalty


def _linear_stats(
    x: np.ndarray, y: np.ndarray, start: int, end: int
) -> LinearSegmentStats:
    segment_x = x[start:end]
    segment_y = y[start:end]
    if segment_x.size < 2:
        return LinearSegmentStats(
            start=start,
            end=end,
            slope=float("nan"),
            intercept=float("nan"),
            sse=float("nan"),
            r2=float("nan"),
            mse=float("nan"),
        )

    A = np.vstack([segment_x, np.ones_like(segment_x)]).T
    coeffs, *_ = np.linalg.lstsq(A, segment_y, rcond=None)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])
    predictions = slope * segment_x + intercept
    residuals = segment_y - predictions
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((segment_y - np.mean(segment_y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    mse = sse / segment_x.size
    return LinearSegmentStats(
        start=start,
        end=end,
        slope=slope,
        intercept=intercept,
        sse=sse,
        r2=r2,
        mse=mse,
    )


def _iter_endpoints(
    n_points: int, n_segments: int, min_size: int
) -> Iterable[list[int]]:
    if n_segments <= 0:
        raise ValueError("n_segments must be positive")
    if min_size <= 0:
        raise ValueError("min_size must be positive")

    def backtrack(start: int, remaining: int, acc: List[int]) -> Iterable[list[int]]:
        if remaining == 1:
            if n_points - start >= min_size:
                yield [*acc, n_points]
            return
        max_end = n_points - min_size * (remaining - 1)
        for end in range(start + min_size, max_end + 1):
            yield from backtrack(end, remaining - 1, [*acc, end])

    return backtrack(0, n_segments, [])


def fit_piecewise_linear(
    x: np.ndarray,
    y: np.ndarray,
    n_segments: int,
    *,
    min_size: int = 5,
) -> PiecewiseLinearModel | None:
    """Fit a piecewise linear model with a fixed number of segments."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if x.size < n_segments * min_size:
        return None

    best_model: PiecewiseLinearModel | None = None
    best_sse = float("inf")

    for endpoints in _iter_endpoints(x.size, n_segments, min_size):
        start = 0
        segments: list[LinearSegmentStats] = []
        total_sse = 0.0
        for end in endpoints:
            stats = _linear_stats(x, y, start, end)
            if not np.isfinite(stats.sse):
                total_sse = float("inf")
                break
            segments.append(stats)
            total_sse += stats.sse
            start = end
        if total_sse < best_sse:
            best_sse = total_sse
            best_model = PiecewiseLinearModel(
                segments=segments,
                breakpoints=endpoints,
                total_sse=total_sse,
                n_points=x.size,
            )

    return best_model


def segment_fixed(
    time_s: np.ndarray,
    y: np.ndarray,
    *,
    thresholds: Sequence[float] = (0.55, 0.70, 0.90),
) -> list[tuple[int, int, float, float]]:
    """Segment using predefined densification thresholds."""

    bounds = [0.0, *thresholds, 1.0]
    bounds = [max(0.0, float(b)) for b in bounds]
    segments: list[tuple[int, int, float, float]] = []
    for idx in range(len(bounds) - 1):
        lower = bounds[idx]
        upper = bounds[idx + 1]
        if idx == len(bounds) - 2:
            mask = (y >= lower) & (y <= upper)
        else:
            mask = (y >= lower) & (y < upper)
        indices = np.where(mask)[0]
        if indices.size == 0:
            continue
        segments.append((indices[0], indices[-1] + 1, lower, upper))
    return segments


def _smooth_max_rate(signal: np.ndarray, window: int = 7) -> np.ndarray:
    if signal.size < 3:
        return signal
    max_window = int(signal.size) if signal.size % 2 == 1 else int(signal.size) - 1
    window = max(3, min(int(window), max_window))
    if window % 2 == 0:
        window += 1
    if window >= signal.size:
        return signal
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")


def segment_data_driven(
    time_s: np.ndarray,
    y: np.ndarray,
    *,
    n_segments: int = 3,
    min_size: int = 5,
) -> list[tuple[int, int, float, float]]:
    """Segment using a simple piecewise-linear search over the curve."""

    model = fit_piecewise_linear(time_s, y, n_segments, min_size=min_size)
    if model is None:
        return []

    segments: list[tuple[int, int, float, float]] = []
    start = 0
    for stats in model.segments:
        end = stats.end
        segment_y = y[start:end]
        lower = float(segment_y.min())
        upper = float(segment_y.max())
        segments.append((start, end, lower, upper))
        start = end
    return segments


def segment_max_rate(
    time_s: np.ndarray,
    y: np.ndarray,
    *,
    n_segments: int = 2,
    min_size: int = 5,
) -> list[tuple[int, int, float, float]]:
    """Segment curve around the peak densification rate."""

    if time_s.size < 2 or y.size < 2:
        return []

    dy_dt = np.gradient(y, time_s)
    dy_dt_smooth = _smooth_max_rate(dy_dt, window=min(11, time_s.size))
    finite = np.isfinite(dy_dt_smooth)
    if not finite.any():
        return []
    adjusted = np.where(finite, dy_dt_smooth, -np.inf)
    peak_idx = int(np.nanargmax(adjusted))

    segments: list[tuple[int, int, float, float]] = []

    def append_segment(start: int, end: int) -> None:
        if end - start < max(1, min_size):
            return
        seg_y = y[start:end]
        segments.append((start, end, float(seg_y.min()), float(seg_y.max())))

    start_idx = 0
    include_initial = n_segments >= 3
    if include_initial:
        threshold = 0.2 * float(dy_dt_smooth[peak_idx])
        candidate = np.where(dy_dt_smooth >= threshold)[0]
        if candidate.size:
            first_active = int(candidate[0])
            initial_end = max(min(peak_idx, first_active), min_size)
        else:
            initial_end = min(peak_idx, min_size)
        if initial_end > start_idx:
            append_segment(start_idx, initial_end)
            start_idx = initial_end

    before_end = max(peak_idx + 1, start_idx + min_size)
    before_end = min(before_end, time_s.size)
    if before_end > start_idx:
        append_segment(start_idx, before_end)
        start_idx = before_end

    after_start = max(peak_idx, start_idx - 1 if start_idx > 0 else peak_idx)
    after_start = max(after_start, 0)
    if time_s.size - after_start >= min_size:
        append_segment(after_start, time_s.size)

    if len(segments) > n_segments:
        segments = segments[:n_segments]
    return segments


def segment_group(
    group: pd.DataFrame,
    *,
    t_col: str,
    y_col: str,
    method: str = "fixed",
    thresholds: Sequence[float] = (0.55, 0.70, 0.90),
    n_segments: int = 3,
    min_size: int = 5,
    sample_value: str | int | float | None = None,
    group_col: str = "sample_id",
) -> list[Segment]:
    """Segment a single group of densification measurements."""

    ordered = group.sort_values(t_col)
    time = ordered[t_col].to_numpy(dtype=float)
    y = ordered[y_col].to_numpy(dtype=float)
    index = ordered.index.to_numpy()

    if method == "fixed":
        raw_segments = segment_fixed(time, y, thresholds=thresholds)
    elif method == "data":
        raw_segments = segment_data_driven(
            time, y, n_segments=n_segments, min_size=min_size
        )
    elif method == "max-rate":
        raw_segments = segment_max_rate(
            time, y, n_segments=n_segments, min_size=min_size
        )
    else:
        raise ValueError("method must be 'fixed', 'data' or 'max-rate'")

    results: list[Segment] = []
    if sample_value is None and group_col in ordered.columns:
        series = ordered[group_col]
        if series.nunique() == 1:
            sample_value = series.iloc[0]

    for idx, (start_pos, end_pos, lower, upper) in enumerate(raw_segments, start=1):
        local_indices = index[start_pos:end_pos]
        segment_time = time[start_pos:end_pos]
        segment_y = y[start_pos:end_pos]
        results.append(
            Segment(
                sample_id=sample_value,
                segment_index=idx,
                method=method,
                lower=lower,
                upper=upper,
                start_time_s=float(segment_time[0]),
                end_time_s=float(segment_time[-1]),
                start_y=float(segment_y[0]),
                end_y=float(segment_y[-1]),
                n_points=int(local_indices.size),
                indices=local_indices,
            )
        )
    return results


def segment_dataframe(
    df: pd.DataFrame,
    *,
    group_col: str = "sample_id",
    t_col: str = "time_s",
    y_col: str = "rho_rel",
    method: str = "fixed",
    thresholds: Sequence[float] = (0.55, 0.70, 0.90),
    n_segments: int = 3,
    min_size: int = 5,
) -> list[Segment]:
    """Segment an entire dataframe across all samples."""

    if y_col not in df.columns:
        raise KeyError(f"Column '{y_col}' not present in dataframe")
    if t_col not in df.columns:
        raise KeyError(f"Column '{t_col}' not present in dataframe")

    if group_col in df.columns:
        grouped = df.groupby(group_col)
        segments = []
        for sample_value, group in grouped:
            segments.extend(
                segment_group(
                    group,
                    t_col=t_col,
                    y_col=y_col,
                    method=method,
                    thresholds=thresholds,
                    n_segments=n_segments,
                    min_size=min_size,
                    sample_value=sample_value,
                    group_col=group_col,
                )
            )
        return segments

    return segment_group(
        df,
        t_col=t_col,
        y_col=y_col,
        method=method,
        thresholds=thresholds,
        n_segments=n_segments,
        min_size=min_size,
        group_col=group_col,
    )


def aggregate_max_rate_bounds(
    df: pd.DataFrame,
    *,
    group_col: str = "sample_id",
    t_col: str = "time_s",
    y_col: str = "rho_rel",
    n_segments: int = 3,
    min_size: int = 5,
) -> list[tuple[float, float]]:
    """Compute representative densification bounds from max-rate segmentation."""

    segments = segment_dataframe(
        df,
        group_col=group_col,
        t_col=t_col,
        y_col=y_col,
        method="max-rate",
        n_segments=n_segments,
        min_size=min_size,
    )
    if not segments:
        return []

    grouped: dict[str | int | float | None, list[Segment]] = {}
    for segment in segments:
        grouped.setdefault(segment.sample_id, []).append(segment)

    boundary_lists: list[list[float]] = []
    for sample_segments in grouped.values():
        sample_segments.sort(key=lambda item: item.segment_index)
        for idx in range(len(sample_segments) - 1):
            boundary = float(sample_segments[idx].upper)
            while len(boundary_lists) <= idx:
                boundary_lists.append([])
            boundary_lists[idx].append(boundary)

    lower = min(float(seg.lower) for seg in segments)
    upper = max(float(seg.upper) for seg in segments)
    if lower >= upper:
        return []

    thresholds: list[float] = []
    for values in boundary_lists:
        if not values:
            continue
        thresholds.append(float(np.median(values)))

    bounds = [lower, *sorted(set(thresholds)), upper]
    bounds = [value for value in bounds if np.isfinite(value)]
    unique_bounds = sorted(set(bounds))
    if len(unique_bounds) < 2:
        return []
    segments_bounds: list[tuple[float, float]] = []
    for idx in range(len(unique_bounds) - 1):
        segments_bounds.append((unique_bounds[idx], unique_bounds[idx + 1]))
    return segments_bounds


__all__ = [
    "Segment",
    "LinearSegmentStats",
    "PiecewiseLinearModel",
    "compute_information_criterion",
    "fit_piecewise_linear",
    "segment_dataframe",
    "segment_data_driven",
    "segment_fixed",
    "segment_max_rate",
    "segment_group",
    "aggregate_max_rate_bounds",
]
