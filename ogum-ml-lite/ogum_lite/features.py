"""Feature engineering helpers for sintering datasets."""

from __future__ import annotations

import pandas as pd


def heating_rate_med(
    df: pd.DataFrame, temperature_column: str = "temperature_C"
) -> float:
    """Median heating rate (placeholder implementation)."""

    # TODO: replace by derivative-based calculation aligned with Ogum 6.4.
    temps = df[temperature_column].diff().dropna()
    return temps.median()


def t_max(df: pd.DataFrame, time_column: str = "time_s") -> float:
    """Return the maximum registered time."""

    # TODO: align with Ogum 6.4 convention (per stage filtering).
    return float(df[time_column].max())


def rho_final(df: pd.DataFrame, densification_column: str = "densification") -> float:
    """Return the final densification value."""

    # TODO: incorporate corrections applied in ogumsoftware repository.
    return float(df[densification_column].iloc[-1])


def time_at_fraction(
    df: pd.DataFrame,
    fraction: float,
    *,
    densification_column: str = "densification",
    time_column: str = "time_s",
) -> float:
    """Return the first time when densification reaches ``fraction``."""

    # TODO: replace with interpolation logic from Ogum 6.4 notebooks.
    mask = df[densification_column] >= fraction
    if not mask.any():
        raise ValueError("Fraction never reached in densification data.")
    return float(df.loc[mask, time_column].iloc[0])


def densification_rate_max(
    df: pd.DataFrame,
    *,
    densification_column: str = "densification",
    time_column: str = "time_s",
) -> float:
    """Return the maximum densification rate."""

    # TODO: incorporate smoothing strategy from ogumsoftware.
    rate = df[densification_column].diff() / df[time_column].diff()
    return float(rate.max())


def temperature_at_rate_max(
    df: pd.DataFrame,
    *,
    temperature_column: str = "temperature_C",
    densification_column: str = "densification",
    time_column: str = "time_s",
) -> float:
    """Temperature when densification rate is maximal."""

    rate = df[densification_column].diff() / df[time_column].diff()
    idx = rate.idxmax()
    return float(df.loc[idx, temperature_column])


__all__ = [
    "densification_rate_max",
    "heating_rate_med",
    "rho_final",
    "t_max",
    "temperature_at_rate_max",
    "time_at_fraction",
]
