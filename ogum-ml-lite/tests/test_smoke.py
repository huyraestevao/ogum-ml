"""Smoke tests for the Ogum Lite processing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from ogum_lite.theta_msc import OgumLite


def synthetic_dataset() -> pd.DataFrame:
    time = np.linspace(0, 600, 301)
    temperature = 25 + 5 * time  # simple ramp
    densification = 1 - np.exp(-time / 200)
    return pd.DataFrame(
        {
            "time_s": time,
            "temp_C": temperature,
            "rho_rel": densification,
            "datatype": "heating",
        }
    )


def test_theta_and_msc_end_to_end(tmp_path) -> None:
    df = synthetic_dataset()

    ogum = OgumLite(df)
    theta_table = ogum.compute_theta_table([200, 300])

    assert list(theta_table["Ea_kJ_mol"]) == [200.0, 300.0]
    assert (theta_table["theta"] > 0).all()

    msc_df = ogum.build_msc(200)
    assert len(msc_df) == len(df)
    assert np.isclose(msc_df["densification"].iloc[-1], df["rho_rel"].iloc[-1])

    msc_path = tmp_path / "msc.csv"
    msc_df.to_csv(msc_path, index=False)
    assert msc_path.exists()
