"""Feature engineering helpers for simulation bundles."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .schema import SimBundle


def _series_mean(series: Sequence[np.ndarray]) -> np.ndarray:
    if not series:
        return np.array([], dtype=float)
    return np.asarray(
        [float(np.mean(np.asarray(arr, dtype=float))) for arr in series],
        dtype=float,
    )


def _series_max(series: Sequence[np.ndarray]) -> tuple[float, int]:
    if not series:
        return float("nan"), -1
    values = np.asarray(
        [float(np.max(np.asarray(arr, dtype=float))) for arr in series],
        dtype=float,
    )
    if not values.size:
        return float("nan"), -1
    idx = int(np.nanargmax(values))
    return float(values[idx]), idx


def _series_percentile(series: Sequence[np.ndarray], q: float) -> float:
    if not series:
        return float("nan")
    stacked = np.concatenate([np.asarray(arr, dtype=float).ravel() for arr in series])
    if stacked.size == 0:
        return float("nan")
    return float(np.percentile(stacked, q))


def _approx_gradient_series(series: Sequence[np.ndarray]) -> np.ndarray:
    if not series:
        return np.array([], dtype=float)
    grads = []
    for arr in series:
        values = np.asarray(arr, dtype=float).ravel()
        if values.size <= 1:
            grads.append(0.0)
            continue
        sorted_values = np.sort(values)
        diffs = np.abs(np.diff(sorted_values))
        grads.append(float(np.mean(diffs)))
    return np.asarray(grads, dtype=float)


def _window_mean(
    series: Sequence[np.ndarray], times: np.ndarray, fraction: float = 0.2
) -> float:
    means = _series_mean(series)
    if means.size == 0:
        return float("nan")
    if times.size == 0:
        return float(np.nanmean(means))
    span = times[-1] - times[0]
    if span <= 0:
        return float(np.nanmean(means))
    threshold = times[-1] - span * fraction
    mask = times >= threshold
    if not mask.any():
        mask[-1] = True
    return float(np.nanmean(means[mask]))


def _integral(times: np.ndarray, values: np.ndarray) -> float:
    if times.size == 0 or values.size == 0:
        return float("nan")
    n = min(times.size, values.size)
    if n < 2:
        return float(np.sum(values[:n]))
    return float(np.trapz(values[:n], times[:n]))


def _field_series(bundle: SimBundle, name: str) -> Sequence[np.ndarray]:
    if name in bundle.node_fields:
        return bundle.node_fields[name]
    if name in bundle.cell_fields:
        return bundle.cell_fields[name]
    return []


def _time_of_index(times: np.ndarray, idx: int) -> float:
    if idx < 0 or idx >= times.size:
        return float("nan")
    return float(times[idx])


def sim_features_global(bundle: SimBundle) -> pd.DataFrame:
    """Compute global features for a simulation bundle."""

    times = np.asarray(bundle.times, dtype=float)
    record: dict[str, float | str] = {"sim_id": bundle.meta.sim_id}

    temp_series = _field_series(bundle, "temp_C")
    temp_means = _series_mean(temp_series)
    temp_max, temp_idx = _series_max(temp_series)
    record["T_max_C"] = temp_max
    record["t_at_T_max_s"] = _time_of_index(times, temp_idx)
    record["T_mean_C_window"] = _window_mean(temp_series, times)
    grad_series = _approx_gradient_series(temp_series)
    record["gradT_mean"] = (
        float(np.nanmean(grad_series)) if grad_series.size else float("nan")
    )
    record["integral_T_dt"] = _integral(times, temp_means)
    record["integral_gradT_dt"] = _integral(times[: grad_series.size], grad_series)

    sigma_series = _field_series(bundle, "von_mises_MPa")
    sigma_max, sigma_idx = _series_max(sigma_series)
    record["sigma_vm_max_MPa"] = sigma_max
    record["sigma_vm_p95"] = _series_percentile(sigma_series, 95.0)
    record["t_at_sigma_vm_max_s"] = _time_of_index(times, sigma_idx)

    e_series = _field_series(bundle, "E_V_per_m")
    e_max, e_idx = _series_max(e_series)
    record["E_max_V_per_m"] = e_max
    record["E_p95_V_per_m"] = _series_percentile(e_series, 95.0)
    record["t_at_E_max_s"] = _time_of_index(times, e_idx)

    return pd.DataFrame([record])


def sim_features_segmented(
    bundle: SimBundle, segments: list[tuple[float, float]]
) -> pd.DataFrame:
    """Compute features on user-defined time segments."""

    if not segments:
        return pd.DataFrame(columns=["sim_id", "segment_start_s", "segment_end_s"])

    times = np.asarray(bundle.times, dtype=float)
    temp_full = list(_field_series(bundle, "temp_C"))
    sigma_full = list(_field_series(bundle, "von_mises_MPa"))
    e_full = list(_field_series(bundle, "E_V_per_m"))

    rows: list[dict[str, float | str]] = []
    for start, end in segments:
        if start > end:
            start, end = end, start
        mask = (times >= start) & (times <= end)
        indices = np.where(mask)[0]
        if indices.size == 0:
            continue
        sub_times = times[indices]
        temp_series = [temp_full[i] for i in indices if i < len(temp_full)]
        sigma_series = [sigma_full[i] for i in indices if i < len(sigma_full)]
        e_series = [e_full[i] for i in indices if i < len(e_full)]

        row: dict[str, float | str] = {
            "sim_id": bundle.meta.sim_id,
            "segment_start_s": float(start),
            "segment_end_s": float(end),
        }

        temp_means = _series_mean(temp_series)
        temp_max, temp_idx = _series_max(temp_series)
        row["T_max_C"] = temp_max
        row["t_at_T_max_s"] = _time_of_index(sub_times, temp_idx)
        row["T_mean_C_window"] = _window_mean(temp_series, sub_times)
        grad_series = _approx_gradient_series(temp_series)
        row["gradT_mean"] = (
            float(np.nanmean(grad_series)) if grad_series.size else float("nan")
        )
        row["integral_T_dt"] = _integral(sub_times, temp_means)
        row["integral_gradT_dt"] = _integral(sub_times[: grad_series.size], grad_series)

        sigma_max, sigma_idx = _series_max(sigma_series)
        row["sigma_vm_max_MPa"] = sigma_max
        row["sigma_vm_p95"] = _series_percentile(sigma_series, 95.0)
        row["t_at_sigma_vm_max_s"] = _time_of_index(sub_times, sigma_idx)

        e_max, e_idx = _series_max(e_series)
        row["E_max_V_per_m"] = e_max
        row["E_p95_V_per_m"] = _series_percentile(e_series, 95.0)
        row["t_at_E_max_s"] = _time_of_index(sub_times, e_idx)

        rows.append(row)

    return pd.DataFrame(rows)


__all__ = ["sim_features_global", "sim_features_segmented"]
