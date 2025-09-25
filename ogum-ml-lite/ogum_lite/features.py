"""Feature engineering helpers for Ogum Lite datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .theta_msc import R_GAS_CONSTANT


@dataclass
class _PreparedTimeseries:
    time_s: np.ndarray
    temp_C: np.ndarray
    y_rel: np.ndarray


def finite_diff(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute first-order derivatives using centred finite differences.

    Parameters
    ----------
    y:
        Array with observations evaluated at the positions ``x``.
    x:
        Monotonically increasing positions used to estimate the derivative.

    Returns
    -------
    numpy.ndarray
        Array with the same shape as ``y`` containing the derivative ``dy/dx``.

    Notes
    -----
    The derivative at the interior points is computed using centred differences,
    while the boundaries rely on forward/backward first-order schemes.
    """

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


def _prepare_group(
    group: pd.DataFrame,
    *,
    t_col: str,
    temp_col: str,
    y_col: str,
) -> _PreparedTimeseries:
    subset = group[[t_col, temp_col, y_col]].dropna()
    subset = subset.sort_values(t_col)
    subset = subset.loc[~subset[t_col].duplicated(keep="first")]

    return _PreparedTimeseries(
        time_s=subset[t_col].to_numpy(dtype=float),
        temp_C=subset[temp_col].to_numpy(dtype=float),
        y_rel=subset[y_col].to_numpy(dtype=float),
    )


def aggregate_timeseries(
    df: pd.DataFrame,
    *,
    group_col: str = "sample_id",
    t_col: str = "time_s",
    temp_col: str = "temp_C",
    y_col: str = "rho_rel",
) -> pd.DataFrame:
    """Aggregate per-sample statistics for long-format sintering datasets.

    Parameters
    ----------
    df:
        Long-format dataframe containing the sintering runs.
    group_col:
        Column identifying the sample/experiment id.
    t_col:
        Column with timestamps in seconds.
    temp_col:
        Column with temperatures in Celsius.
    y_col:
        Column with relative density or shrinkage fraction (0–1).

    Returns
    -------
    pandas.DataFrame
        Dataframe indexed by ``group_col`` with engineered features per sample.
    """

    required_columns = {group_col, t_col, temp_col, y_col}
    missing = required_columns - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns: {missing_cols}")

    records: list[dict[str, float | str]] = []

    for sample_id, group in df.groupby(group_col):
        prepared = _prepare_group(group, t_col=t_col, temp_col=temp_col, y_col=y_col)

        time = prepared.time_s
        temp = prepared.temp_C
        y_rel = prepared.y_rel

        metrics: dict[str, float | str] = {group_col: sample_id}

        if time.size >= 2:
            dtemp_dt = finite_diff(temp, time)
            metrics["heating_rate_med_C_per_s"] = float(np.nanmedian(dtemp_dt))

            dy_dt = finite_diff(y_rel, time)
            idx_max = int(np.argmax(dy_dt))
            metrics["dy_dt_max"] = float(dy_dt[idx_max])
            metrics["T_at_dy_dt_max_C"] = float(temp[idx_max])
        else:
            metrics["heating_rate_med_C_per_s"] = float("nan")
            metrics["dy_dt_max"] = float("nan")
            metrics["T_at_dy_dt_max_C"] = float("nan")

        metrics["T_max_C"] = float(group[temp_col].max())
        metrics["y_final"] = float(y_rel[-1]) if y_rel.size else float("nan")

        mask_90 = y_rel >= 0.90
        if mask_90.any():
            metrics["t_to_90pct_s"] = float(time[mask_90][0])
        else:
            metrics["t_to_90pct_s"] = float("nan")

        records.append(metrics)

    return pd.DataFrame.from_records(records)


def _format_ea_label(value: float) -> str:
    formatted = ("%g" % value).replace(".", "p")
    return f"theta_Ea_{formatted}kJ"


def theta_features(
    df_long: pd.DataFrame,
    ea_kj_list: Iterable[float],
    *,
    group_col: str = "sample_id",
    t_col: str = "time_s",
    temp_col: str = "temp_C",
    y_col: str = "rho_rel",
) -> pd.DataFrame:
    """Compute θ(Ea) integrals per sample for a list of activation energies.

    Parameters
    ----------
    df_long:
        Long-format dataframe with the sintering runs.
    ea_kj_list:
        Iterable containing activation energies in kJ/mol.
    group_col, t_col, temp_col, y_col:
        Column names describing the long-format structure.

    Returns
    -------
    pandas.DataFrame
        Dataframe with one row per sample and θ(Ea) totals as additional columns.
    """

    required_columns = {group_col, t_col, temp_col, y_col}
    missing = required_columns - set(df_long.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns: {missing_cols}")

    ea_values = list(ea_kj_list)
    if not ea_values:
        raise ValueError("`ea_kj_list` must contain at least one activation energy")

    records: list[dict[str, float | str]] = []

    for sample_id, group in df_long.groupby(group_col):
        prepared = _prepare_group(group, t_col=t_col, temp_col=temp_col, y_col=y_col)
        time = prepared.time_s
        temp = prepared.temp_C + 273.15

        metrics: dict[str, float | str] = {group_col: sample_id}

        if time.size >= 2:
            for ea in ea_values:
                integrand = np.exp(-(ea * 1000.0) / (R_GAS_CONSTANT * temp))
                theta_total = float(np.trapezoid(integrand, time))
                metrics[_format_ea_label(ea)] = theta_total
        else:
            for ea in ea_values:
                metrics[_format_ea_label(ea)] = float("nan")

        records.append(metrics)

    return pd.DataFrame.from_records(records)


def build_feature_table(
    df_long: pd.DataFrame,
    ea_kj_list: Iterable[float],
    *,
    group_col: str = "sample_id",
    t_col: str = "time_s",
    temp_col: str = "temp_C",
    y_col: str = "rho_rel",
) -> pd.DataFrame:
    """Combine aggregated statistics and θ(Ea) totals for each sample."""

    agg = aggregate_timeseries(
        df_long,
        group_col=group_col,
        t_col=t_col,
        temp_col=temp_col,
        y_col=y_col,
    )
    theta = theta_features(
        df_long,
        ea_kj_list,
        group_col=group_col,
        t_col=t_col,
        temp_col=temp_col,
        y_col=y_col,
    )
    return pd.merge(agg, theta, on=group_col, how="inner")


__all__ = [
    "aggregate_timeseries",
    "build_feature_table",
    "finite_diff",
    "theta_features",
]
