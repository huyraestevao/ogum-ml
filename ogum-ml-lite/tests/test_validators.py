from __future__ import annotations

import pandas as pd
from ogum_lite.validators import validate_feature_df, validate_long_df


def test_validate_long_df_detects_out_of_range() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "time_s": [0.0, -1.0],
            "temp_C": [25.0, 3000.0],
            "rho_rel": [0.5, 1.5],
        }
    )

    report = validate_long_df(df)
    assert report["ok"] is False
    joined = "\n".join(report["issues"])
    assert "temp_C" in joined
    assert "time_s" in joined


def test_validate_long_df_accepts_valid_payload() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "time_s": [12.5],
            "temp_C": [1250.0],
            "rho_rel": [0.82],
        }
    )

    report = validate_long_df(df)
    assert report["ok"] is True
    assert report["issues"] == []


def test_validate_feature_df_reports_nans_and_bounds() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["x", "y"],
            "heating_rate_med_C_per_s": [0.5, -0.2],
            "T_max_C": [1350.0, 2600.0],
            "y_final": [0.95, 1.1],
            "t_to_90pct_s": [100.0, None],
            "dy_dt_max": [0.01, float("nan")],
            "T_at_dy_dt_max_C": [900.0, 800.0],
        }
    )

    report = validate_feature_df(df)
    assert report["ok"] is False
    joined = "\n".join(report["issues"])
    assert "NaN ratio" in joined
    assert "T_max_C" in joined


def test_validate_feature_df_accepts_theta_columns() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "heating_rate_med_C_per_s": [1.2],
            "T_max_C": [1450.0],
            "y_final": [0.88],
            "t_to_90pct_s": [95.0],
            "dy_dt_max": [0.04],
            "T_at_dy_dt_max_C": [1420.0],
            "theta_Ea_200": [0.75],
        }
    )

    report = validate_feature_df(df)
    assert report["ok"] is True
    assert report["issues"] == []
