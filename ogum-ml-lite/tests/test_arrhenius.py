from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ogum_lite.arrhenius import (
    R_GAS_CONSTANT,
    arrhenius_lnT_dy_dt_vs_invT,
    fit_arrhenius_by_stages,
    fit_arrhenius_global,
    fit_arrhenius_sliding,
)
from ogum_lite.stages import DEFAULT_STAGES


def synthetic_arrhenius_dataset() -> pd.DataFrame:
    time = np.linspace(0.0, 1500.0, 250)
    temp_C = 25.0 + 0.15 * time
    T_K = temp_C + 273.15
    Ea = 85_000.0  # J/mol
    A = 2.0e7
    dy_dt = A * np.exp(-Ea / (R_GAS_CONSTANT * T_K))
    dt = time[1] - time[0]
    y = np.cumsum(dy_dt) * dt
    y = y / y.max() * 0.95

    return pd.DataFrame(
        {
            "sample_id": "S1",
            "time_s": time,
            "temp_C": temp_C,
            "y": y,
            "dy_dt": dy_dt,
        }
    )


def test_fit_arrhenius_global_recovers_activation_energy() -> None:
    df = synthetic_arrhenius_dataset()
    prepared = arrhenius_lnT_dy_dt_vs_invT(df)
    result = fit_arrhenius_global(prepared)
    assert result.Ea_J_mol == pytest.approx(85_000.0, rel=0.05)


def test_fit_arrhenius_by_stages_produces_results() -> None:
    df = synthetic_arrhenius_dataset()
    prepared = arrhenius_lnT_dy_dt_vs_invT(df)
    results = fit_arrhenius_by_stages(prepared, stages=DEFAULT_STAGES, y_col="y")
    assert len(results) >= 1


def test_fit_arrhenius_sliding_returns_windows() -> None:
    df = synthetic_arrhenius_dataset()
    prepared = arrhenius_lnT_dy_dt_vs_invT(df)
    results = fit_arrhenius_sliding(prepared, window_pts=30, step=10)
    assert results
