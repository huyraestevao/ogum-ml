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
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

R_GAS_CONSTANT = 8.314462618  # J/(mol*K)


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
        ``temperature_C``
            Process temperature in Celsius.
        ``densification``
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
        self, column: str = "densification", reference: str = "first"
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
        temperature_column: str = "temperature_C",
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
        densification_column: str = "densification",
        temperature_column: str = "temperature_C",
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


__all__ = ["OgumLite", "R_GAS_CONSTANT"]
