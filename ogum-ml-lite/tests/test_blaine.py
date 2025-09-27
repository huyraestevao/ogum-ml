import numpy as np
import pandas as pd
import pytest

from ogum_lite.blaine import fit_blaine_by_segments, fit_blaine_segment
from ogum_lite.segmentation import Segment


def test_blaine_linearisation_returns_expected_exponent() -> None:
    time = np.linspace(1.0, 5.0, 20)
    y = 1.0 - 1.0 / (time**2)

    stats = fit_blaine_segment(time, y)

    assert stats["n"] == pytest.approx(2.0, abs=1e-3)
    assert stats["r2"] == pytest.approx(1.0, abs=1e-6)


def test_blaine_by_segments_uses_segment_indices() -> None:
    time = np.linspace(1.0, 4.0, 15)
    y = 1.0 - 1.0 / (time**2)
    df = pd.DataFrame({"sample_id": "A", "time_s": time, "rho_rel": y})

    segment = Segment(
        sample_id="A",
        segment_index=1,
        method="fixed",
        lower=float(y.min()),
        upper=float(y.max()),
        start_time_s=float(time[0]),
        end_time_s=float(time[-1]),
        start_y=float(y[0]),
        end_y=float(y[-1]),
        n_points=len(time),
        indices=df.index.to_numpy(),
    )

    results = fit_blaine_by_segments(df, [segment], t_col="time_s", y_col="rho_rel")
    assert len(results) == 1
    result = results[0]
    assert result.sample_id == "A"
    assert result.n == pytest.approx(2.0, abs=1e-3)
    assert result.r2 == pytest.approx(1.0, abs=1e-6)
    assert result.mse < 1e-6
