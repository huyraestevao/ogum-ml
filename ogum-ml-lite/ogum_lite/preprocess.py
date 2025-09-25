"""Pre-processing utilities: smoothing and numerical derivatives."""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

SmoothMethod = Literal["savgol", "moving", "none"]


def smooth_series(
    values: Iterable[float],
    *,
    method: SmoothMethod = "savgol",
    window: int = 11,
    poly: int = 3,
    moving_k: int = 5,
) -> np.ndarray:
    """Smooth 1D series using Savitzky–Golay or moving average filters."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array

    if method == "none":
        return array.copy()

    if method == "savgol":
        win = max(3, int(window))
        if win % 2 == 0:
            win += 1
        win = min(win, array.size if array.size % 2 == 1 else array.size - 1)
        if win < 3:
            return array.copy()
        if poly >= win:
            poly = win - 1
        return savgol_filter(
            array, window_length=win, polyorder=max(poly, 1), mode="interp"
        )

    if method == "moving":
        k = max(1, int(moving_k))
        window_arr = np.ones(k) / k
        return np.convolve(array, window_arr, mode="same")

    raise ValueError("Unsupported smoothing method")


def finite_diff(y: Iterable[float], x: Iterable[float]) -> np.ndarray:
    """Compute first-order derivatives using centred finite differences."""

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if y.shape != x.shape:
        raise ValueError("`y` and `x` must share the same shape")
    if y.size < 2:
        raise ValueError("At least two samples are required to estimate the derivative")
    if np.any(np.diff(x) <= 0):
        raise ValueError("Array `x` must be strictly increasing")

    derivative = np.empty_like(y, dtype=float)
    derivative[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    derivative[0] = (y[1] - y[0]) / (x[1] - x[0])
    derivative[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return derivative


def derive_all(
    df: pd.DataFrame,
    *,
    t_col: str = "time_s",
    T_col: str = "temp_C",
    y_col: str = "y",
    smooth: SmoothMethod = "savgol",
    window: int = 11,
    poly: int = 3,
    moving_k: int = 5,
) -> pd.DataFrame:
    """Compute time and temperature derivatives used in Arrhenius analyses."""

    required = {t_col, T_col, y_col}
    missing = required - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns: {missing_cols}")

    result = df.copy()
    group_col = "sample_id" if "sample_id" in result.columns else None

    groups: Iterable[tuple[str | int | None, pd.DataFrame]]
    if group_col:
        groups = ((name, group) for name, group in result.groupby(group_col))
    else:
        groups = [(None, result)]

    for _, group in groups:
        subset = group[[t_col, T_col, y_col]].dropna()
        if subset.empty:
            continue
        subset = subset.sort_values(t_col)
        subset = subset.loc[~subset[t_col].duplicated(keep="first")]

        index = subset.index.to_numpy()
        time = subset[t_col].to_numpy(dtype=float)
        temp = subset[T_col].to_numpy(dtype=float)
        y_values = subset[y_col].to_numpy(dtype=float)

        if time.size < 2:
            result.loc[index, "dy_dt"] = np.nan
            result.loc[index, "dT_dt"] = np.nan
            result.loc[index, "dp_dT"] = np.nan
            result.loc[index, "T_times_dy_dt"] = np.nan
            result.loc[index, "T_times_dp_dT_times_dT_dt"] = np.nan
            continue

        y_smooth = smooth_series(
            y_values, method=smooth, window=window, poly=poly, moving_k=moving_k
        )

        try:
            dy_dt = finite_diff(y_smooth, time)
            dT_dt = finite_diff(temp, time)
        except ValueError:
            dy_dt = np.full_like(time, np.nan, dtype=float)
            dT_dt = np.full_like(time, np.nan, dtype=float)

        dp_dT = dy_dt / (dT_dt + 1e-12)
        T_K = temp + 273.15
        T_times_dy_dt = T_K * dy_dt
        T_times_dp_dT_times_dT_dt = T_K * dp_dT * dT_dt

        result.loc[index, "y_smooth"] = y_smooth
        result.loc[index, "dy_dt"] = dy_dt
        result.loc[index, "dT_dt"] = dT_dt
        result.loc[index, "dp_dT"] = dp_dT
        result.loc[index, "T_times_dy_dt"] = T_times_dy_dt
        result.loc[index, "T_times_dp_dT_times_dT_dt"] = T_times_dp_dT_times_dT_dt

    return result


__all__ = ["SmoothMethod", "smooth_series", "finite_diff", "derive_all"]
