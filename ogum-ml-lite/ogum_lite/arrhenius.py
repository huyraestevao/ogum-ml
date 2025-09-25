"""Arrhenius regression helpers for sintering kinetics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from scipy.stats import linregress

from .stages import DEFAULT_STAGES, split_by_stages

R_GAS_CONSTANT = 8.314462618  # J/(mol*K)


@dataclass
class ArrheniusResult:
    """Container for Arrhenius linear regressions."""

    Ea_J_mol: float
    slope: float
    intercept: float
    rvalue: float
    n_points: int
    method: Literal["global", "stage", "sliding"]
    meta: dict[str, Any]


def arrhenius_lnT_dy_dt_vs_invT(
    df: pd.DataFrame,
    *,
    T_col: str = "temp_C",
    dy_dt_col: str = "dy_dt",
) -> pd.DataFrame:
    """Prepare dataframe with ``ln(T*dy/dt)`` versus ``1/T`` columns."""

    required = {T_col, dy_dt_col}
    missing = required - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns: {missing_cols}")

    prepared = df.copy()
    T_K = prepared[T_col].astype(float) + 273.15
    dy_dt = prepared[dy_dt_col].astype(float)
    dy_dt_clip = np.clip(dy_dt, 1e-12, None)

    prepared["T_K"] = T_K
    prepared["invT_K"] = 1.0 / T_K
    prepared["ln_T_dy_dt"] = np.log(T_K * dy_dt_clip)
    return prepared


def _linear_fit(df: pd.DataFrame) -> tuple[float, float, float, int]:
    subset = df[["invT_K", "ln_T_dy_dt"]].dropna()
    if subset.shape[0] < 3:
        raise ValueError("At least three valid points are required for Arrhenius fit")
    result = linregress(subset["invT_K"], subset["ln_T_dy_dt"])
    return result.slope, result.intercept, result.rvalue, subset.shape[0]


def _ensure_prepared(
    df: pd.DataFrame, *, T_col: str = "temp_C", dy_dt_col: str = "dy_dt"
) -> pd.DataFrame:
    if {"invT_K", "ln_T_dy_dt"}.issubset(df.columns):
        return df
    return arrhenius_lnT_dy_dt_vs_invT(df, T_col=T_col, dy_dt_col=dy_dt_col)


def fit_arrhenius_global(
    df: pd.DataFrame,
    *,
    T_col: str = "temp_C",
    dy_dt_col: str = "dy_dt",
) -> ArrheniusResult:
    """Fit Arrhenius regression using all available samples."""

    prepared = _ensure_prepared(df, T_col=T_col, dy_dt_col=dy_dt_col)
    slope, intercept, rvalue, n_points = _linear_fit(prepared)
    Ea = -slope * R_GAS_CONSTANT
    return ArrheniusResult(
        Ea_J_mol=float(Ea),
        slope=float(slope),
        intercept=float(intercept),
        rvalue=float(rvalue),
        n_points=int(n_points),
        method="global",
        meta={},
    )


def fit_arrhenius_by_stages(
    df: pd.DataFrame,
    *,
    stages: Iterable[tuple[float, float]] = DEFAULT_STAGES,
    y_col: str = "y",
    group_col: str = "sample_id",
) -> list[ArrheniusResult]:
    """Fit Arrhenius regressions for each densification stage."""

    prepared = _ensure_prepared(df)
    stage_frames = split_by_stages(
        prepared, y_col=y_col, group_col=group_col, stages=stages
    )

    results: list[ArrheniusResult] = []
    for idx, ((lower, upper), label) in enumerate(zip(stages, stage_frames)):
        frame = stage_frames[label]
        if frame.empty:
            continue
        slope, intercept, rvalue, n_points = _linear_fit(frame)
        Ea = -slope * R_GAS_CONSTANT
        results.append(
            ArrheniusResult(
                Ea_J_mol=float(Ea),
                slope=float(slope),
                intercept=float(intercept),
                rvalue=float(rvalue),
                n_points=int(n_points),
                method="stage",
                meta={"stage": label, "lower": float(lower), "upper": float(upper)},
            )
        )
    return results


def fit_arrhenius_sliding(
    df: pd.DataFrame,
    *,
    window_pts: int = 25,
    step: int = 5,
    t_col: str = "time_s",
    group_col: str = "sample_id",
) -> list[ArrheniusResult]:
    """Fit Arrhenius regressions over sliding windows along the time axis."""

    if window_pts < 5:
        raise ValueError("window_pts must be at least 5")
    if step < 1:
        raise ValueError("step must be positive")

    prepared = _ensure_prepared(df)
    results: list[ArrheniusResult] = []

    grouped = (
        prepared.groupby(group_col)
        if group_col in prepared.columns
        else [(None, prepared)]
    )
    for sample_id, group in grouped:
        subset = group.sort_values(t_col)
        if subset.shape[0] < window_pts:
            continue
        values = subset[[t_col, "invT_K", "ln_T_dy_dt"]].dropna()
        if values.shape[0] < window_pts:
            continue

        for start in range(0, values.shape[0] - window_pts + 1, step):
            window_df = values.iloc[start : start + window_pts]
            try:
                slope, intercept, rvalue, n_points = _linear_fit(window_df)
            except ValueError:
                continue
            Ea = -slope * R_GAS_CONSTANT
            results.append(
                ArrheniusResult(
                    Ea_J_mol=float(Ea),
                    slope=float(slope),
                    intercept=float(intercept),
                    rvalue=float(rvalue),
                    n_points=int(n_points),
                    method="sliding",
                    meta={
                        "sample_id": sample_id,
                        "t_start": float(window_df[t_col].iloc[0]),
                        "t_end": float(window_df[t_col].iloc[-1]),
                    },
                )
            )
    return results


__all__ = [
    "ArrheniusResult",
    "arrhenius_lnT_dy_dt_vs_invT",
    "fit_arrhenius_global",
    "fit_arrhenius_by_stages",
    "fit_arrhenius_sliding",
    "R_GAS_CONSTANT",
]
