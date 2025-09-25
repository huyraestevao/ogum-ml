"""Feature engineering helpers for Ogum Lite datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .arrhenius import (
    arrhenius_lnT_dy_dt_vs_invT,
    fit_arrhenius_by_stages,
    fit_arrhenius_global,
)
from .preprocess import derive_all, finite_diff
from .stages import DEFAULT_STAGES, split_by_stages
from .theta_msc import R_GAS_CONSTANT


@dataclass
class _PreparedTimeseries:
    index: np.ndarray
    time_s: np.ndarray
    temp_C: np.ndarray
    y_rel: np.ndarray


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
        index=subset.index.to_numpy(),
        time_s=subset[t_col].to_numpy(dtype=float),
        temp_C=subset[temp_col].to_numpy(dtype=float),
        y_rel=subset[y_col].to_numpy(dtype=float),
    )


def aggregate_per_sample(
    df_long: pd.DataFrame,
    *,
    group_col: str = "sample_id",
    t_col: str = "time_s",
    T_col: str = "temp_C",
    y_col: str = "y",
) -> pd.DataFrame:
    """Aggregate per-sample statistics for long-format sintering datasets."""

    required_columns = {group_col, t_col, T_col, y_col}
    missing = required_columns - set(df_long.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns: {missing_cols}")

    records: list[dict[str, float | str]] = []

    for sample_id, group in df_long.groupby(group_col):
        prepared = _prepare_group(group, t_col=t_col, temp_col=T_col, y_col=y_col)
        index = prepared.index
        time = prepared.time_s
        temp = prepared.temp_C
        y_values = prepared.y_rel

        metrics: dict[str, float | str] = {group_col: sample_id}
        metrics["T_max_C"] = float(np.nanmax(temp)) if temp.size else float("nan")
        metrics["y_final"] = float(y_values[-1]) if y_values.size else float("nan")

        mask_90 = y_values >= 0.90
        metrics["t_to_90pct_s"] = (
            float(time[mask_90][0]) if mask_90.any() else float("nan")
        )

        if time.size >= 2:
            if "dT_dt" in group.columns and group["dT_dt"].notna().any():
                dT_dt = group.loc[index, "dT_dt"].to_numpy(dtype=float)
            else:
                dT_dt = finite_diff(temp, time)

            if "dy_dt" in group.columns and group["dy_dt"].notna().any():
                dy_dt = group.loc[index, "dy_dt"].to_numpy(dtype=float)
            else:
                dy_dt = finite_diff(y_values, time)

            metrics["heating_rate_med_C_per_s"] = float(np.nanmedian(dT_dt))

            idx_max_dy = int(np.nanargmax(dy_dt)) if np.isfinite(dy_dt).any() else None
            if idx_max_dy is not None:
                metrics["dy_dt_max"] = float(dy_dt[idx_max_dy])
                metrics["T_at_dy_dt_max_C"] = float(temp[idx_max_dy])
            else:
                metrics["dy_dt_max"] = float("nan")
                metrics["T_at_dy_dt_max_C"] = float("nan")

            idx_max_dT = int(np.nanargmax(dT_dt)) if np.isfinite(dT_dt).any() else None
            if idx_max_dT is not None:
                metrics["dT_dt_max"] = float(dT_dt[idx_max_dT])
                metrics["t_at_dT_dt_max_s"] = float(time[idx_max_dT])
            else:
                metrics["dT_dt_max"] = float("nan")
                metrics["t_at_dT_dt_max_s"] = float("nan")
        else:
            metrics.update(
                {
                    "heating_rate_med_C_per_s": float("nan"),
                    "dy_dt_max": float("nan"),
                    "T_at_dy_dt_max_C": float("nan"),
                    "dT_dt_max": float("nan"),
                    "t_at_dT_dt_max_s": float("nan"),
                }
            )

        records.append(metrics)

    return pd.DataFrame.from_records(records)


def aggregate_timeseries(
    df: pd.DataFrame,
    *,
    group_col: str = "sample_id",
    t_col: str = "time_s",
    temp_col: str = "temp_C",
    y_col: str = "rho_rel",
) -> pd.DataFrame:
    """Backward compatible wrapper around :func:`aggregate_per_sample`."""

    return aggregate_per_sample(
        df,
        group_col=group_col,
        t_col=t_col,
        T_col=temp_col,
        y_col=y_col,
    )


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
    """Compute θ(Ea) integrals per sample for a list of activation energies."""

    required_columns = {group_col, t_col, temp_col, y_col}
    missing = required_columns - set(df_long.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns: {missing_cols}")

    ea_values = [float(value) for value in ea_kj_list]
    if not ea_values:
        raise ValueError("`ea_kj_list` must contain at least one activation energy")
    if any(value <= 0 for value in ea_values):
        raise ValueError("Activation energies must be positive")

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


def _stage_suffix(lower: float, upper: float) -> str:
    return f"{int(round(lower * 100)):02d}_{int(round(upper * 100)):02d}"


def build_stage_feature_tables(
    df_long: pd.DataFrame,
    *,
    stages: Sequence[tuple[float, float]] = DEFAULT_STAGES,
    group_col: str = "sample_id",
    t_col: str = "time_s",
    T_col: str = "temp_C",
    y_col: str = "y",
) -> dict[str, pd.DataFrame]:
    """Aggregate per-stage statistics using :func:`aggregate_per_sample`."""

    stage_frames = split_by_stages(
        df_long,
        y_col=y_col,
        group_col=group_col,
        stages=stages,
    )

    stage_tables: dict[str, pd.DataFrame] = {}
    for idx, ((lower, upper), label) in enumerate(zip(stages, stage_frames)):
        frame = stage_frames[label]
        if frame.empty:
            continue
        aggregated = aggregate_per_sample(
            frame,
            group_col=group_col,
            t_col=t_col,
            T_col=T_col,
            y_col=y_col,
        )
        suffix = f"_s{idx + 1}"
        rename = {
            column: f"{column}{suffix}"
            for column in aggregated.columns
            if column != group_col
        }
        stage_tables[label] = aggregated.rename(columns=rename)
    return stage_tables


def arrhenius_feature_table(
    df_long: pd.DataFrame,
    *,
    stages: Sequence[tuple[float, float]] = DEFAULT_STAGES,
    group_col: str = "sample_id",
    t_col: str = "time_s",
    T_col: str = "temp_C",
    y_col: str = "y",
    smooth: str = "savgol",
    window: int = 11,
    poly: int = 3,
    moving_k: int = 5,
) -> pd.DataFrame:
    """Compute Arrhenius activation energies per sample and per stage."""

    if "dy_dt" not in df_long.columns:
        derived = derive_all(
            df_long,
            t_col=t_col,
            T_col=T_col,
            y_col=y_col,
            smooth=smooth,
            window=window,
            poly=poly,
            moving_k=moving_k,
        )
    else:
        derived = df_long

    prepared = arrhenius_lnT_dy_dt_vs_invT(derived, T_col=T_col, dy_dt_col="dy_dt")

    records: list[Mapping[str, float | str]] = []
    for sample_id, group in prepared.groupby(group_col):
        row: dict[str, float | str] = {group_col: sample_id}
        try:
            global_res = fit_arrhenius_global(group)
            row["Ea_arr_global_kJ"] = global_res.Ea_J_mol / 1000.0
            row["rvalue_arr_global"] = global_res.rvalue
        except ValueError:
            row["Ea_arr_global_kJ"] = float("nan")
            row["rvalue_arr_global"] = float("nan")

        stage_results = fit_arrhenius_by_stages(
            group,
            stages=stages,
            y_col=y_col,
            group_col=group_col,
        )
        stage_lookup = {res.meta["stage"]: res for res in stage_results}
        for idx, (lower, upper) in enumerate(stages, start=1):
            key = _stage_suffix(lower, upper)
            stage_label = f"stage_{idx}"
            result = stage_lookup.get(stage_label)
            if result is None:
                row[f"Ea_arr_{key}_kJ"] = float("nan")
                row[f"rvalue_arr_{key}"] = float("nan")
            else:
                row[f"Ea_arr_{key}_kJ"] = result.Ea_J_mol / 1000.0
                row[f"rvalue_arr_{key}"] = result.rvalue

        records.append(row)

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


def build_feature_store(
    df_long: pd.DataFrame,
    *,
    stages: Sequence[tuple[float, float]] = DEFAULT_STAGES,
    group_col: str = "sample_id",
    t_col: str = "time_s",
    T_col: str = "temp_C",
    y_col: str = "y",
    smooth: str = "savgol",
    window: int = 11,
    poly: int = 3,
    moving_k: int = 5,
    theta_ea_kj: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Build a consolidated feature store including stage-aware columns."""

    derived = derive_all(
        df_long,
        t_col=t_col,
        T_col=T_col,
        y_col=y_col,
        smooth=smooth,
        window=window,
        poly=poly,
        moving_k=moving_k,
    )

    base = aggregate_per_sample(
        derived,
        group_col=group_col,
        t_col=t_col,
        T_col=T_col,
        y_col=y_col,
    )

    stage_tables = build_stage_feature_tables(
        derived,
        stages=stages,
        group_col=group_col,
        t_col=t_col,
        T_col=T_col,
        y_col=y_col,
    )
    features = base.copy()
    base_feature_cols = [col for col in base.columns if col != group_col]
    for idx, stage in enumerate(stages, start=1):
        label = f"stage_{idx}"
        table = stage_tables.get(label)
        if table is None:
            empty_cols = {f"{col}_s{idx}": np.nan for col in base_feature_cols}
            table = pd.DataFrame({group_col: features[group_col]}).assign(**empty_cols)
        features = features.merge(table, on=group_col, how="left")

    arrhenius_table = arrhenius_feature_table(
        derived,
        stages=stages,
        group_col=group_col,
        t_col=t_col,
        T_col=T_col,
        y_col=y_col,
        smooth=smooth,
        window=window,
        poly=poly,
        moving_k=moving_k,
    )
    features = features.merge(arrhenius_table, on=group_col, how="left")

    if theta_ea_kj is not None:
        theta = theta_features(
            df_long,
            theta_ea_kj,
            group_col=group_col,
            t_col=t_col,
            temp_col=T_col,
            y_col=y_col,
        )
        features = features.merge(theta, on=group_col, how="left")

    return features


__all__ = [
    "aggregate_per_sample",
    "aggregate_timeseries",
    "arrhenius_feature_table",
    "build_feature_store",
    "build_feature_table",
    "build_stage_feature_tables",
    "finite_diff",
    "theta_features",
]
