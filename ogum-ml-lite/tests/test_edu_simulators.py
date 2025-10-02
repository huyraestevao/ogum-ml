import numpy as np
import pytest
from app.edu import simulators


def _sample_curves():
    time = np.linspace(0, 1000, 50)
    temps = [900 + 50 * np.sin(time / 200 + phase) for phase in (0.0, 0.4, 0.8)]
    curves = []
    for idx, temp in enumerate(temps, start=1):
        theta = simulators.simulate_theta(temp, time, 320.0)
        y = 0.5 + 0.4 * np.power(theta / theta[-1], 1.2 + 0.1 * idx)
        curves.append(
            {
                "time_s": time,
                "temp_C": temp,
                "y": y,
                "label": f"sample {idx}",
            }
        )
    return curves


def test_simulate_theta_is_increasing():
    time = np.linspace(0, 1000, 50)
    temp = np.linspace(900, 1150, 50)
    theta = simulators.simulate_theta(temp, time, 310.0)
    assert np.all(np.diff(theta) >= 0)
    assert theta.shape == time.shape


def test_collapse_produces_mean_curve():
    result = simulators.simulate_msc_collapse(_sample_curves(), 320.0)
    assert result.mean_curve.shape == result.grid_theta.shape
    assert result.mse >= 0
    assert result.figure is not None


def test_blaine_linearisation_metrics():
    theta = np.linspace(0.01, 1.0, 100)
    y = 0.6 * np.power(theta, 1.5)
    result = simulators.simulate_blaine_linearization(theta, y)
    assert 0 <= result.r2 <= 1
    assert result.n_est == pytest.approx(1.5, abs=0.1)
    assert result.figure is not None
