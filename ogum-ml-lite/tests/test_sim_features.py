import numpy as np
from ogum_lite.sim.features import sim_features_global, sim_features_segmented
from ogum_lite.sim.schema import MeshInfo, SimBundle, SimMeta


def _bundle() -> SimBundle:
    meta = SimMeta(sim_id="sim01", solver="csv", units={"time": "s", "temp_C": "C"})
    mesh = MeshInfo(num_nodes=1, num_cells=0)
    times = np.array([0.0, 5.0, 10.0])
    temp = [np.array([20.0]), np.array([30.0]), np.array([40.0])]
    sigma = [np.array([100.0]), np.array([120.0]), np.array([150.0])]
    efield = [np.array([1.0]), np.array([0.5]), np.array([0.2])]
    return SimBundle(
        meta=meta,
        mesh=mesh,
        times=times,
        node_fields={"temp_C": temp},
        cell_fields={"von_mises_MPa": sigma, "E_V_per_m": efield},
    )


def test_sim_features_global():
    bundle = _bundle()
    df = sim_features_global(bundle)
    assert list(df.columns)[0] == "sim_id"
    assert df.loc[0, "sim_id"] == "sim01"
    assert df.loc[0, "T_max_C"] == 40.0
    assert df.loc[0, "t_at_T_max_s"] == 10.0
    assert df.loc[0, "integral_T_dt"] == 300.0
    assert df.loc[0, "sigma_vm_max_MPa"] == 150.0
    assert df.loc[0, "E_max_V_per_m"] == 1.0


def test_sim_features_segmented():
    bundle = _bundle()
    segments = [(0.0, 5.0), (5.0, 10.0)]
    df = sim_features_segmented(bundle, segments)
    assert len(df) == 2
    assert set(df.columns) >= {
        "sim_id",
        "segment_start_s",
        "segment_end_s",
        "T_max_C",
        "sigma_vm_max_MPa",
    }
    first = df.iloc[0]
    assert first["segment_start_s"] == 0.0
    assert first["segment_end_s"] == 5.0
    assert first["T_max_C"] == 30.0
