import numpy as np
import pandas as pd
import pytest
from ogum_lite.segmentation import segment_dataframe


def _build_dataframe() -> pd.DataFrame:
    time = np.linspace(0, 10, 101)
    rho = np.linspace(0.40, 0.95, 101)
    return pd.DataFrame({"sample_id": "S1", "time_s": time, "rho_rel": rho})


def test_fixed_segmentation_generates_expected_ranges() -> None:
    df = _build_dataframe()
    segments = segment_dataframe(
        df,
        method="fixed",
        thresholds=(0.55, 0.70, 0.90),
        group_col="sample_id",
        t_col="time_s",
        y_col="rho_rel",
    )

    assert len(segments) == 4
    assert pytest.approx(segments[0].lower, rel=1e-6) == 0.0
    assert pytest.approx(segments[-1].upper, rel=1e-6) == 1.0
    assert all(segment.sample_id == "S1" for segment in segments)


def test_data_driven_segmentation_detects_breakpoints() -> None:
    time = np.linspace(0, 12, 120)
    y = np.piecewise(
        time,
        [time < 4, (time >= 4) & (time < 8), time >= 8],
        [
            lambda t: 0.40 + 0.05 * t,
            lambda t: 0.60 + 0.03 * (t - 4),
            lambda t: 0.72 + 0.06 * (t - 8),
        ],
    )
    df = pd.DataFrame({"sample_id": "S2", "time_s": time, "rho_rel": y})

    segments = segment_dataframe(
        df,
        method="data",
        n_segments=3,
        min_size=10,
        group_col="sample_id",
        t_col="time_s",
        y_col="rho_rel",
    )

    assert len(segments) == 3
    assert segments[0].end_time_s == pytest.approx(4.0, abs=0.5)
    assert segments[1].end_time_s == pytest.approx(8.0, abs=0.5)
