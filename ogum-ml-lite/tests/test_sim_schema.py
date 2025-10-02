from pathlib import Path

import numpy as np
from ogum_lite.sim.schema import (
    MeshInfo,
    SimBundle,
    SimMeta,
    from_disk,
    to_disk,
    validate_bundle,
)


def _make_bundle() -> SimBundle:
    times = np.array([0.0, 5.0, 10.0])
    node_fields = {"temp_C": [np.array([20.0]), np.array([25.0]), np.array([30.0])]}
    meta = SimMeta(sim_id="run_a", solver="csv", units={"time": "s", "temp_C": "C"})
    mesh = MeshInfo(num_nodes=0, num_cells=0)
    return SimBundle(
        meta=meta,
        mesh=mesh,
        times=times,
        node_fields=node_fields,
        cell_fields={},
    )


def test_validate_bundle_ok():
    bundle = _make_bundle()
    report = validate_bundle(bundle)
    assert report["ok"]
    assert not report["issues"]


def test_roundtrip_to_disk(tmp_path: Path):
    bundle = _make_bundle()
    target = tmp_path / "bundle"
    to_disk(bundle, target)
    loaded = from_disk(target)
    assert loaded.meta.sim_id == bundle.meta.sim_id
    assert np.allclose(loaded.times, bundle.times)
    assert loaded.meta.units == bundle.meta.units
    assert np.allclose(loaded.node_fields["temp_C"][1], bundle.node_fields["temp_C"][1])
