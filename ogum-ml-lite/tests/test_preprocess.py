from __future__ import annotations

import numpy as np
import pandas as pd
from ogum_lite.preprocess import derive_all, smooth_series


def test_savgol_smoothing_preserves_linear_trend() -> None:
    data = np.linspace(0, 10, 21)
    smoothed = smooth_series(data, method="savgol", window=5, poly=2)
    assert np.allclose(smoothed, data)


def test_derive_all_linear_profiles() -> None:
    time = np.linspace(0, 49, 50)
    df = pd.DataFrame(
        {
            "sample_id": "S1",
            "time_s": time,
            "temp_C": 20.0 + 2.0 * time,
            "y": 0.3 + 0.01 * time,
        }
    )

    derived = derive_all(df, smooth="none")

    assert np.allclose(derived["dT_dt"].dropna(), 2.0)
    assert np.allclose(derived["dy_dt"].dropna(), 0.01)
    assert np.allclose(derived["dp_dT"].dropna(), 0.01 / 2.0, atol=1e-6)
    t_kelvin = derived["temp_C"] + 273.15
    assert np.allclose(derived["T_times_dy_dt"].dropna(), t_kelvin.dropna() * 0.01)
