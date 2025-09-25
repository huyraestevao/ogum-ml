"""Unit tests for feature engineering helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from ogum_lite.features import (
    aggregate_timeseries,
    build_feature_table,
    finite_diff,
    theta_features,
)


def synthetic_long_dataframe() -> pd.DataFrame:
    time = np.linspace(0, 400, 81)
    frames = []
    for idx, temp_shift in enumerate([0.0, 5.0]):
        rho = 1 - np.exp(-(time + idx * 15) / 140)
        frames.append(
            pd.DataFrame(
                {
                    "sample_id": f"S{idx}",
                    "time_s": time,
                    "temp_C": 25 + 2.5 * time + temp_shift,
                    "rho_rel": rho,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_finite_diff_linear_profile() -> None:
    x = np.linspace(0, 10, 6)
    y = 3 * x - 5
    derivative = finite_diff(y, x)
    assert np.allclose(derivative, 3.0)


def test_aggregate_timeseries_basic_metrics() -> None:
    df = synthetic_long_dataframe()
    aggregated = aggregate_timeseries(df)

    assert set(aggregated.columns) == {
        "sample_id",
        "heating_rate_med_C_per_s",
        "T_max_C",
        "y_final",
        "t_to_90pct_s",
        "dy_dt_max",
        "T_at_dy_dt_max_C",
    }
    assert len(aggregated) == 2
    assert (aggregated["heating_rate_med_C_per_s"] > 0).all()
    assert np.isfinite(aggregated["t_to_90pct_s"]).all()
    assert (aggregated["T_max_C"] > 25).all()


def test_theta_features_and_feature_table() -> None:
    df = synthetic_long_dataframe()
    thetas = theta_features(df, [200.0, 350.0])

    assert {
        "sample_id",
        "theta_Ea_200kJ",
        "theta_Ea_350kJ",
    } <= set(thetas.columns)

    for _, row in thetas.iterrows():
        assert row["theta_Ea_200kJ"] > row["theta_Ea_350kJ"]

    features_df = build_feature_table(df, [200.0, 350.0])
    assert "theta_Ea_200kJ" in features_df.columns
    assert "heating_rate_med_C_per_s" in features_df.columns
    assert len(features_df) == 2
