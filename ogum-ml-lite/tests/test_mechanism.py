import numpy as np
import pandas as pd
import pytest
from ogum_lite.mechanism import detect_mechanism_change


def test_mechanism_change_detected_in_piecewise_curve() -> None:
    theta = np.linspace(0.0, 10.0, 200)
    densification = np.piecewise(
        theta,
        [theta < 5.0, theta >= 5.0],
        [lambda t: 0.2 + 0.05 * t, lambda t: 0.45 + 0.09 * (t - 5.0)],
    )
    df = pd.DataFrame(
        {
            "sample_id": "S1",
            "theta": theta,
            "densification": densification,
        }
    )

    result = detect_mechanism_change(df)
    row = result.iloc[0]
    assert bool(row["change_detected"])
    assert row["n_segments"] == 2
    assert row["breakpoint_theta"] == pytest.approx(5.0, abs=0.3)


def test_mechanism_no_change_for_linear_relation() -> None:
    theta = np.linspace(0.0, 10.0, 150)
    densification = 0.25 + 0.07 * theta
    df = pd.DataFrame(
        {
            "sample_id": "S2",
            "theta": theta,
            "densification": densification,
        }
    )

    result = detect_mechanism_change(df)
    row = result.iloc[0]
    assert not bool(row["change_detected"])
    assert row["n_segments"] == 1
