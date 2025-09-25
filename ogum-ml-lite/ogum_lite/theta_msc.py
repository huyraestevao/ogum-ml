"""Utilities for computing Ogum's θ(Ea) tables and Master Sintering Curves.

The implementation mirrors the data processing steps adopted in Ogum 6.4 while
remaining lightweight enough to run inside Google Colab sessions.  Only a small
subset of the full feature extraction pipeline is implemented here – enough to
provide meaningful θ(Ea) values and master sintering curves (MSC) for rapid
experimentation and ML prototyping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

from .stages import DEFAULT_STAGES

R_GAS_CONSTANT = 8.314462618  # J/(mol*K)


@dataclass
class MasterCurveResult:
    """Container with the outcome of a MSC collapse for a given Ea."""

    activation_energy: float
    curve: pd.DataFrame
    per_sample: pd.DataFrame
    mse_global: float
    mse_segmented: float
    segment_mse: Dict[Tuple[float, float], float]
    normalize_theta: Literal["minmax", None]
    segments: Tuple[Tuple[float, float], ...]


@dataclass
class OgumLite:
    """High level API for θ(Ea) and MSC calculations.

    Parameters
    ----------
    data:
        Optional data frame that contains the sintering runs to analyse.  The
        following columns are expected:

        ``time_s``
            Time in seconds since the start of the run.
        ``temp_C``
            Process temperature in Celsius.
        ``rho_rel``
            Relative density or shrinkage fraction between 0 and 1.
        ``datatype`` (optional)
            Stage or label to filter datasets (e.g. ``"heating"``).
    """

    data: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Data handling utilities
    # ------------------------------------------------------------------
    def load_csv(self, path: str | Path, **kwargs) -> pd.DataFrame:
        """Load a CSV file and store the resulting dataframe.

        Parameters
        ----------
        path:
            Path to the CSV file containing sintering data.
        **kwargs:
            Additional arguments forwarded to :func:`pandas.read_csv`.

        Returns
        -------
        pandas.DataFrame
            The loaded dataframe with the expected columns.
        """

        dataframe = pd.read_csv(path, **kwargs)
        self.data = dataframe
        return dataframe

    def select_datatype(self, datatype: str, column: str = "datatype") -> pd.DataFrame:
        """Filter the dataset keeping only rows matching ``datatype``.

        Parameters
        ----------
        datatype:
            Desired value in the ``column`` column.
        column:
            Column name holding stage identifiers.

        Returns
        -------
        pandas.DataFrame
            Filtered view of the stored dataframe.
        """

        if self.data is None:
            raise ValueError("No data loaded. Call load_csv first.")

        if column not in self.data.columns:
            raise KeyError(f"Column '{column}' not found in dataframe")

        filtered = self.data[self.data[column] == datatype].copy()
        self.data = filtered
        return filtered

    def cut_by_time(
        self,
        t_min: float | None = None,
        t_max: float | None = None,
        column: str = "time_s",
    ) -> pd.DataFrame:
        """Restrict the stored dataframe to the given time window.

        Parameters
        ----------
        t_min, t_max:
            Optional lower/upper bounds in seconds.  ``None`` keeps the
            corresponding side open.
        column:
            Name of the time column.
        """

        if self.data is None:
            raise ValueError("No data loaded. Call load_csv first.")

        if column not in self.data.columns:
            raise KeyError(f"Column '{column}' not found in dataframe")

        data = self.data
        if t_min is not None:
            data = data[data[column] >= t_min]
        if t_max is not None:
            data = data[data[column] <= t_max]
        self.data = data.copy()
        return self.data

    def baseline_shift(
        self, column: str = "rho_rel", reference: str = "first"
    ) -> pd.DataFrame:
        """Shift the densification baseline to start at zero.

        Parameters
        ----------
        column:
            Densification column to normalise.
        reference:
            ``"first"`` subtracts the initial value; ``"min"`` subtracts the
            minimum value.
        """

        if self.data is None:
            raise ValueError("No data loaded. Call load_csv first.")

        if column not in self.data.columns:
            raise KeyError(f"Column '{column}' not found in dataframe")

        if reference == "first":
            ref_value = self.data[column].iloc[0]
        elif reference == "min":
            ref_value = self.data[column].min()
        else:
            raise ValueError("reference must be 'first' or 'min'")

        self.data[column] = self.data[column] - ref_value
        return self.data

    # ------------------------------------------------------------------
    # θ(Ea) and MSC computation
    # ------------------------------------------------------------------
    def compute_theta_table(
        self,
        activation_energies: Iterable[float],
        temperature_column: str = "temp_C",
        time_column: str = "time_s",
    ) -> pd.DataFrame:
        """Compute the Ogum θ(Ea) table for the provided activation energies.

        Parameters
        ----------
        activation_energies:
            Sequence with activation energies in kJ/mol.
        temperature_column:
            Column with temperatures in Celsius.
        time_column:
            Column with time stamps in seconds.

        Returns
        -------
        pandas.DataFrame
            A dataframe with columns ``Ea_kJ_mol`` and ``theta``.
        """

        data = self._require_columns([temperature_column, time_column])
        temperatures_K = data[temperature_column].to_numpy() + 273.15
        times = data[time_column].to_numpy()

        if np.any(np.diff(times) < 0):
            order = np.argsort(times)
            times = times[order]
            temperatures_K = temperatures_K[order]

        thetas = []
        for ea in activation_energies:
            integrand = np.exp(-(ea * 1000.0) / (R_GAS_CONSTANT * temperatures_K))
            theta = np.trapezoid(integrand, times)
            thetas.append({"Ea_kJ_mol": float(ea), "theta": float(theta)})
        return pd.DataFrame(thetas)

    def build_msc(
        self,
        activation_energy: float,
        densification_column: str = "rho_rel",
        temperature_column: str = "temp_C",
        time_column: str = "time_s",
    ) -> pd.DataFrame:
        """Construct a Master Sintering Curve (MSC).

        Parameters
        ----------
        activation_energy:
            Activation energy in kJ/mol used as the reference.
        densification_column:
            Column with densification values (0–1).
        temperature_column:
            Column with temperature in Celsius.
        time_column:
            Column with time in seconds.

        Returns
        -------
        pandas.DataFrame
            Dataframe with columns ``theta`` and ``densification`` suitable for
            plotting or exporting.
        """

        data = self._require_columns(
            [densification_column, temperature_column, time_column]
        ).sort_values(time_column)

        temperatures_K = data[temperature_column].to_numpy() + 273.15
        times = data[time_column].to_numpy()
        densification = data[densification_column].to_numpy()

        integrand = np.exp(
            -(activation_energy * 1000.0) / (R_GAS_CONSTANT * temperatures_K)
        )
        theta = cumulative_trapezoid(integrand, times, initial=0.0)

        return pd.DataFrame({"theta": theta, "densification": densification})

    def plot_msc(self, msc_df: pd.DataFrame, *, ax: plt.Axes | None = None) -> plt.Axes:
        """Plot a Master Sintering Curve.

        Parameters
        ----------
        msc_df:
            Output of :meth:`build_msc`.
        ax:
            Optional axis to plot on.  When ``None`` a new figure is created.

        Returns
        -------
        matplotlib.axes.Axes
            Axis with the plotted curve.
        """

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(msc_df["theta"], msc_df["densification"], label="MSC")
        ax.set_xlabel(r"$\Theta(E_a)$")
        ax.set_ylabel("Densification")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return ax

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _require_columns(self, columns: Sequence[str]) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv first.")

        missing = [column for column in columns if column not in self.data.columns]
        if missing:
            raise KeyError(f"Missing columns: {', '.join(missing)}")
        return self.data


def _normalise_theta(theta: np.ndarray, mode: Literal["minmax", None]) -> np.ndarray:
    if mode is None:
        return theta
    if mode == "minmax":
        theta_min = float(theta.min())
        theta_max = float(theta.max())
        scale = theta_max - theta_min
        if scale <= 0:
            return np.zeros_like(theta)
        return (theta - theta_min) / scale
    raise ValueError("normalize_theta must be 'minmax' or None")


def build_master_curve(
    df_long: pd.DataFrame,
    activation_energy: float,
    *,
    group_col: str = "sample_id",
    t_col: str = "time_s",
    temp_col: str = "temp_C",
    y_col: str = "rho_rel",
    normalize_theta: Literal["minmax", None] = "minmax",
    segments: Sequence[Tuple[float, float]] = DEFAULT_STAGES,
    grid_size: int = 200,
) -> MasterCurveResult:
    """Collapse multiple runs into a MSC and compute error metrics.

    Parameters
    ----------
    df_long:
        Long-format dataframe containing multiple samples identified via
        ``group_col``.
    activation_energy:
        Activation energy in kJ/mol used for the θ(Ea) integration.
    group_col, t_col, temp_col, y_col:
        Column names describing the long-format dataset.
    normalize_theta:
        Normalisation strategy applied to the cumulative θ arrays prior to the
        collapse.  ``"minmax"`` scales each curve to [0, 1]; ``None`` keeps the
        original θ values.
    segments:
        Iterable with (lower, upper) densification ranges used to compute the
        segmented MSE.
    grid_size:
        Number of points used to sample the common densification grid.

    Returns
    -------
    MasterCurveResult
        Dataclass containing the collapsed curve, per-sample projections and
        error metrics.
    """

    required_columns = {group_col, t_col, temp_col, y_col}
    missing = required_columns - set(df_long.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns: {missing_cols}")

    if grid_size < 10:
        raise ValueError("grid_size must be at least 10 to sample the MSC")

    segments_tuple: Tuple[Tuple[float, float], ...] = tuple(
        (float(seg[0]), float(seg[1])) for seg in segments
    )
    if not segments_tuple:
        raise ValueError("segments must contain at least one (lower, upper) pair")

    segment_min = min(seg[0] for seg in segments_tuple)
    segment_max = max(seg[1] for seg in segments_tuple)
    if segment_max <= segment_min:
        raise ValueError(
            "Invalid segments: upper bound must be greater than lower bound"
        )

    y_grid = np.linspace(segment_min, segment_max, grid_size)

    per_sample_rows: list[pd.DataFrame] = []
    theta_matrix: list[np.ndarray] = []

    for sample_id, group in df_long.groupby(group_col):
        subset = group[[t_col, temp_col, y_col]].dropna()
        if subset.empty:
            continue
        subset = subset.sort_values(t_col)
        subset = subset.loc[~subset[t_col].duplicated(keep="first")]

        if len(subset) < 2:
            continue

        time = subset[t_col].to_numpy(dtype=float)
        temp = subset[temp_col].to_numpy(dtype=float) + 273.15
        y_values = subset[y_col].to_numpy(dtype=float)

        if np.any(np.diff(time) <= 0):
            order = np.argsort(time)
            time = time[order]
            temp = temp[order]
            y_values = y_values[order]
            mask = np.concatenate(([True], np.diff(time) > 0))
            time = time[mask]
            temp = temp[mask]
            y_values = y_values[mask]
            if time.size < 2:
                continue

        integrand = np.exp(-(activation_energy * 1000.0) / (R_GAS_CONSTANT * temp))
        theta = cumulative_trapezoid(integrand, time, initial=0.0)
        theta_norm = _normalise_theta(theta, normalize_theta)

        order_y = np.argsort(y_values)
        y_sorted = y_values[order_y]
        theta_sorted = theta_norm[order_y]
        y_unique, unique_idx = np.unique(y_sorted, return_index=True)
        theta_unique = theta_sorted[unique_idx]

        if y_unique[0] > y_grid[0] or y_unique[-1] < y_grid[-1]:
            continue

        theta_interp = np.interp(y_grid, y_unique, theta_unique)
        theta_matrix.append(theta_interp)

        per_sample_rows.append(
            pd.DataFrame(
                {
                    group_col: sample_id,
                    "densification": y_grid,
                    "theta": theta_interp,
                }
            )
        )

    if len(theta_matrix) < 2:
        raise ValueError("At least two samples are required to build the MSC")

    theta_stack = np.vstack(theta_matrix)
    theta_mean = theta_stack.mean(axis=0)
    theta_std = theta_stack.std(axis=0, ddof=0)
    errors = (theta_stack - theta_mean) ** 2
    mse_global = float(errors.mean())

    segment_metrics: Dict[Tuple[float, float], float] = {}
    weighted_sum = 0.0
    total_weight = 0.0
    for lower, upper in segments_tuple:
        mask = (y_grid >= lower) & (y_grid <= upper)
        if not np.any(mask):
            segment_metrics[(lower, upper)] = float("nan")
            continue
        mse_value = float(errors[:, mask].mean())
        segment_metrics[(lower, upper)] = mse_value
        if np.isfinite(mse_value):
            weight = upper - lower
            weighted_sum += weight * mse_value
            total_weight += weight

    mse_segmented = weighted_sum / total_weight if total_weight > 0 else float("nan")

    curve_df = pd.DataFrame(
        {
            "densification": y_grid,
            "theta_mean": theta_mean,
            "theta_std": theta_std,
        }
    )
    per_sample_df = pd.concat(per_sample_rows, ignore_index=True)

    return MasterCurveResult(
        activation_energy=float(activation_energy),
        curve=curve_df,
        per_sample=per_sample_df,
        mse_global=mse_global,
        mse_segmented=mse_segmented,
        segment_mse=segment_metrics,
        normalize_theta=normalize_theta,
        segments=segments_tuple,
    )


def score_activation_energies(
    df_long: pd.DataFrame,
    ea_kj_list: Iterable[float],
    *,
    metric: Literal["global", "segmented"] = "segmented",
    **kwargs,
) -> tuple[pd.DataFrame, MasterCurveResult, list[MasterCurveResult]]:
    """Evaluate multiple activation energies and return collapse metrics."""

    ea_values = list(ea_kj_list)
    if not ea_values:
        raise ValueError("ea_kj_list must contain at least one activation energy")

    if metric not in {"global", "segmented"}:
        raise ValueError("metric must be 'global' or 'segmented'")

    results: list[MasterCurveResult] = []
    rows: list[dict[str, float]] = []

    for ea in ea_values:
        result = build_master_curve(df_long, ea, **kwargs)
        results.append(result)

        row: dict[str, float] = {
            "Ea_kJ_mol": float(ea),
            "mse_global": result.mse_global,
            "mse_segmented": result.mse_segmented,
        }
        for (lower, upper), value in result.segment_mse.items():
            key = f"mse_{lower:.2f}_{upper:.2f}"
            row[key] = value
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("Ea_kJ_mol").reset_index(drop=True)

    metric_key = "mse_segmented" if metric == "segmented" else "mse_global"
    best_result = None
    best_value = float("inf")
    for result in results:
        value = getattr(result, metric_key)
        if np.isnan(value):
            continue
        if value < best_value:
            best_value = value
            best_result = result

    if best_result is None:
        raise ValueError("No valid MSC could be computed for the provided Ea values")

    return summary, best_result, results


__all__ = [
    "MasterCurveResult",
    "OgumLite",
    "R_GAS_CONSTANT",
    "build_master_curve",
    "score_activation_energies",
]
