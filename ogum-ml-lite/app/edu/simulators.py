"""Educational simulators for the training mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import plotly.graph_objects as go

R_GAS = 8.314  # J/(mol*K)


@dataclass
class CollapseResult:
    """Container with results of the MSC collapse simulation."""

    grid_theta: np.ndarray
    mean_curve: np.ndarray
    mse: float
    collapsed_curves: list[dict]
    figure: go.Figure


@dataclass
class BlaineResult:
    """Container with the Blaine linearisation output."""

    n_est: float
    r2: float
    mse: float
    log_theta: np.ndarray
    log_y: np.ndarray
    figure: go.Figure


def simulate_theta(
    temp_profile_C: np.ndarray,
    time_s: np.ndarray,
    Ea_kJ: float,
) -> np.ndarray:
    r"""Compute the discrete Arrhenius integral :math:`\Theta(t)`.

    Parameters
    ----------
    temp_profile_C:
        Temperature profile in Celsius degrees for each timestamp.
    time_s:
        Time samples in seconds, strictly increasing.
    Ea_kJ:
        Activation energy expressed in kJ/mol.

    Returns
    -------
    np.ndarray
        Array with the cumulative Arrhenius integral for each instant.
    """

    temps_K = np.asarray(temp_profile_C, dtype=float) + 273.15
    time_s = np.asarray(time_s, dtype=float)
    if temps_K.shape != time_s.shape:
        raise ValueError("Temperature and time arrays must have the same shape")
    if temps_K.ndim != 1:
        raise ValueError("Temperature and time arrays must be one-dimensional")
    if np.any(np.diff(time_s) <= 0):
        raise ValueError("time_s must be strictly increasing")

    ea_j = Ea_kJ * 1_000.0
    k = np.exp(-ea_j / (R_GAS * temps_K))
    increments = np.concatenate(([0.0], 0.5 * (k[1:] + k[:-1]) * np.diff(time_s)))
    theta = np.cumsum(increments)
    theta -= theta.min()
    return theta


def make_fig_theta(
    time_s: np.ndarray, temp_C: np.ndarray, theta: np.ndarray
) -> go.Figure:
    r"""Build a dual-axis plot showing the thermal cycle and :math:`\Theta(t)`."""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=time_s, y=temp_C, mode="lines", name="T (°C)", yaxis="y1")
    )
    fig.add_trace(go.Scatter(x=time_s, y=theta, mode="lines", name="Θ", yaxis="y2"))
    fig.update_layout(
        xaxis_title="t (s)",
        yaxis=dict(title="T (°C)", rangemode="tozero"),
        yaxis2=dict(
            title="Θ", overlaying="y", side="right", showgrid=False, rangemode="tozero"
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    return fig


def _prepare_curve(curve: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    time = np.asarray(curve["time_s"], dtype=float)
    temp = np.asarray(curve["temp_C"], dtype=float)
    y = np.asarray(curve["y"], dtype=float)
    label = str(curve.get("label", "sample"))
    return time, temp, y, label


def simulate_msc_collapse(curves: Sequence[dict], Ea_kJ: float) -> CollapseResult:
    """Collapse multiple densification curves using the Arrhenius time.

    Parameters
    ----------
    curves:
        Sequence of dictionaries with ``time_s``, ``temp_C`` and ``y`` arrays.
    Ea_kJ:
        Candidate activation energy in kJ/mol.

    Returns
    -------
    CollapseResult
        Structured result with the collapsed curves, MSE and Plotly figure.
    """

    collapsed: list[dict] = []
    grid = np.linspace(0.0, 1.0, 200)
    interpolated: list[np.ndarray] = []

    for curve in curves:
        time, temp, y, label = _prepare_curve(curve)
        theta = simulate_theta(temp, time, Ea_kJ)
        theta_norm = theta / theta[-1] if theta[-1] != 0 else theta
        y_interp = np.interp(grid, theta_norm, y)
        collapsed.append(
            {
                "theta_norm": theta_norm,
                "y": y,
                "time_s": time,
                "label": label,
            }
        )
        interpolated.append(y_interp)

    stacked = np.vstack(interpolated)
    mean_curve = np.mean(stacked, axis=0)
    mse = float(np.mean((stacked - mean_curve) ** 2))
    figure = make_fig_collapse(
        grid,
        stacked,
        mean_curve,
        [c["label"] for c in collapsed],
    )
    return CollapseResult(
        grid_theta=grid,
        mean_curve=mean_curve,
        mse=mse,
        collapsed_curves=collapsed,
        figure=figure,
    )


def make_fig_collapse(
    grid_theta: np.ndarray,
    collapsed_y: Iterable[np.ndarray],
    mean_curve: np.ndarray,
    labels: Sequence[str],
) -> go.Figure:
    """Create a Plotly figure with the collapsed MSC curves."""

    fig = go.Figure()
    for series, label in zip(collapsed_y, labels):
        fig.add_trace(
            go.Scatter(
                x=grid_theta,
                y=series,
                mode="lines",
                name=label,
                hovertemplate="Θ: %{x:.2f}<br>y: %{y:.3f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=grid_theta,
            y=mean_curve,
            mode="lines",
            name="média",
            line=dict(width=4, dash="dash"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Θ normalizado",
        yaxis_title="Densificação",
    )
    return fig


def simulate_blaine_linearization(theta: np.ndarray, y: np.ndarray) -> BlaineResult:
    """Estimate ``n`` via Blaine linearisation from Arrhenius time and densification.

    Parameters
    ----------
    theta:
        Arrhenius time vector (positive values).
    y:
        Densification response aligned with ``theta``.

    Returns
    -------
    BlaineResult
        Result object containing the estimated ``n`` and diagnostic metrics.
    """

    theta = np.asarray(theta, dtype=float)
    y = np.asarray(y, dtype=float)
    if theta.shape != y.shape:
        raise ValueError("theta and y must have the same shape")
    mask = (theta > 0) & (y > 0)
    if mask.sum() < 2:
        raise ValueError("Need at least two valid samples for Blaine linearisation")
    theta = theta[mask]
    y = y[mask]

    log_theta = np.log(theta)
    log_y = np.log(y)
    slope, intercept = np.polyfit(log_theta, log_y, 1)
    y_pred = np.exp(intercept + slope * log_theta)
    residuals = y - y_pred
    mse = float(np.mean(residuals**2))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    figure = make_fig_blaine(log_theta, log_y, slope, intercept)
    return BlaineResult(
        n_est=float(slope),
        r2=float(max(min(r2, 1.0), 0.0)),
        mse=mse,
        log_theta=log_theta,
        log_y=log_y,
        figure=figure,
    )


def make_fig_blaine(
    log_theta: np.ndarray, log_y: np.ndarray, slope: float, intercept: float
) -> go.Figure:
    """Plot the Blaine linearisation along with the fitted line."""

    x_sorted = np.linspace(log_theta.min(), log_theta.max(), 100)
    y_line = intercept + slope * x_sorted

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=log_theta,
            y=log_y,
            mode="markers",
            name="dados",
            hovertemplate="ln Θ: %{x:.2f}<br>ln y: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(go.Scatter(x=x_sorted, y=y_line, mode="lines", name="ajuste"))
    fig.update_layout(
        template="plotly_white",
        xaxis_title="ln Θ",
        yaxis_title="ln densificação",
    )
    return fig
