from pathlib import Path

import numpy as np
import pytest
from ogum_lite.sim import adapters


def test_load_csv_timeseries(tmp_path: Path):
    csv_path = tmp_path / "summary.csv"
    csv_path.write_text(
        "time_s,temp_K,E_field\n0,300,10\n10,350,15\n",
        encoding="utf-8",
    )

    bundle = adapters.load_csv_timeseries(csv_path)
    assert bundle.meta.sim_id == "summary"
    assert np.allclose(bundle.times, [0.0, 10.0])
    assert bundle.meta.units["temp_C"] == "C"
    assert bundle.node_fields["temp_C"][0][0] == pytest.approx(26.85, rel=1e-5)
    assert bundle.meta.units["E_V_per_m"] == "V/m"
    assert bundle.node_fields["E_V_per_m"][1][0] == pytest.approx(15.0)


def test_vtk_requires_meshio(monkeypatch):
    def _missing():
        raise RuntimeError("meshio missing")

    monkeypatch.setattr(adapters, "_require_meshio", _missing)

    with pytest.raises(RuntimeError, match="meshio"):
        adapters.load_vtk_series(Path("/tmp/not-there"))
