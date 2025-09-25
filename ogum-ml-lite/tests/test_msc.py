"""Tests for Master Sintering Curve utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ogum_lite.theta_msc import build_master_curve, score_activation_energies


def synthetic_runs(n_samples: int = 3) -> pd.DataFrame:
    time = np.linspace(0, 600, 121)
    frames = []
    for idx in range(n_samples):
        rho = 1 - np.exp(-(time + idx * 20) / 180)
        temp = 25 + 3.5 * time + idx * 4
        frames.append(
            pd.DataFrame(
                {
                    "sample_id": f"S{idx}",
                    "time_s": time,
                    "temp_C": temp,
                    "rho_rel": rho,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_build_master_curve_segmented_metric_bounded() -> None:
    df = synthetic_runs()
    result = build_master_curve(df, 220.0)

    assert {"densification", "theta_mean", "theta_std"} <= set(result.curve.columns)
    assert result.mse_segmented <= 1.5 * result.mse_global


def test_build_master_curve_requires_multiple_samples() -> None:
    df = synthetic_runs(1)
    with pytest.raises(ValueError, match="At least two samples"):
        build_master_curve(df, 200.0)


def test_score_activation_energies_returns_best_result() -> None:
    df = synthetic_runs()
    summary, best_result, results = score_activation_energies(
        df,
        [180.0, 200.0, 240.0],
        metric="global",
    )

    assert len(results) == 3
    assert not summary.empty
    assert best_result.activation_energy in summary["Ea_kJ_mol"].values

    raw_result = build_master_curve(
        df,
        best_result.activation_energy,
        normalize_theta=None,
    )
    assert np.allclose(
        raw_result.curve["densification"],
        best_result.curve["densification"],
    )
