"""Canonical schema definitions for simulation bundles."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


@dataclass(slots=True)
class SimMeta:
    """Metadata describing a simulation run.

    Parameters
    ----------
    sim_id:
        Unique identifier for the simulation bundle.
    solver:
        Name of the solver or exporter used to generate the data (``vtk``, ``csv``...).
    version:
        Solver version string when available.
    units:
        Mapping of canonical field names to the unit system (``{"temp_C": "C"}``).
    """

    sim_id: str
    solver: str
    version: str | None = None
    units: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class MeshInfo:
    """Summary information about the mesh supporting the fields.

    Parameters
    ----------
    num_nodes:
        Number of nodes in the mesh. Zero when the bundle stores global scalars only.
    num_cells:
        Number of cells/elements in the mesh.
    cell_type:
        Canonical cell name (``tri``, ``tet``, ``hex``...) if known.
    dimension:
        Spatial dimensionality (1, 2, 3). ``None`` when unknown.
    """

    num_nodes: int
    num_cells: int
    cell_type: str | None = None
    dimension: int | None = None


@dataclass(slots=True)
class TimeSeriesInfo:
    """Representation of the temporal grid in seconds."""

    t: np.ndarray


@dataclass(slots=True)
class SimBundle:
    """Container holding simulation data in the Ogum canonical format."""

    meta: SimMeta
    mesh: MeshInfo
    times: np.ndarray
    node_fields: dict[str, list[np.ndarray]] = field(default_factory=dict)
    cell_fields: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def copy(self) -> "SimBundle":
        """Return a shallow copy of the bundle."""

        return SimBundle(
            meta=replace(self.meta, units=dict(self.meta.units)),
            mesh=replace(self.mesh),
            times=self.times.copy(),
            node_fields={
                k: [arr.copy() for arr in v] for k, v in self.node_fields.items()
            },
            cell_fields={
                k: [arr.copy() for arr in v] for k, v in self.cell_fields.items()
            },
        )


def _ensure_1d_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Times must be a one-dimensional array")
    return arr


def validate_bundle(bundle: SimBundle) -> dict[str, Any]:
    """Validate structural consistency of a :class:`SimBundle`.

    Parameters
    ----------
    bundle:
        Bundle to validate.

    Returns
    -------
    dict
        Dictionary containing ``ok`` (``bool``) and ``issues`` (``list[str]``).
    """

    issues: list[str] = []

    try:
        times = _ensure_1d_array(bundle.times)
    except ValueError as exc:
        issues.append(str(exc))
        times = np.asarray([], dtype=float)

    if times.size and np.any(np.diff(times) < 0):
        issues.append("Times must be monotonic increasing")

    expected_steps = len(times)
    for field_name, series in bundle.node_fields.items():
        if len(series) not in {expected_steps, 0}:
            issues.append(
                (
                    f"Node field '{field_name}' has {len(series)} steps, expected "
                    f"{expected_steps}"
                )
            )
        _check_consistent_shapes(series, issues, f"node field '{field_name}'")

    for field_name, series in bundle.cell_fields.items():
        if len(series) not in {expected_steps, 0}:
            issues.append(
                (
                    f"Cell field '{field_name}' has {len(series)} steps, expected "
                    f"{expected_steps}"
                )
            )
        _check_consistent_shapes(series, issues, f"cell field '{field_name}'")

    if bundle.mesh.num_nodes < 0 or bundle.mesh.num_cells < 0:
        issues.append("Mesh counts must be non-negative")

    ok = not issues
    return {"ok": ok, "issues": issues}


def _check_consistent_shapes(
    series: list[np.ndarray], issues: list[str], label: str
) -> None:
    if not series:
        return
    first_shape = series[0].shape
    for idx, arr in enumerate(series):
        if not isinstance(arr, np.ndarray):
            issues.append(f"{label} step {idx} is not an ndarray")
            continue
        if arr.shape != first_shape:
            issues.append(
                (
                    f"{label} step {idx} shape {arr.shape} differs from first "
                    f"{first_shape}"
                )
            )


def infer_units(bundle: SimBundle) -> SimBundle:
    """Return a bundle with heuristically normalised units.

    Currently converts temperature fields expressed in Kelvin to Celsius when the
    unit map declares ``K`` or ``kelvin``.
    """

    normalised = bundle.copy()
    units_lower = {
        k: v.lower() for k, v in normalised.meta.units.items() if isinstance(v, str)
    }

    for field_name, series in normalised.node_fields.items():
        unit = units_lower.get(field_name)
        if field_name.startswith("temp") and unit in {"k", "kelvin"}:
            for idx, arr in enumerate(series):
                series[idx] = arr.astype(float) - 273.15
            normalised.meta.units[field_name] = "C"

    return normalised


def to_disk(bundle: SimBundle, outdir: Path) -> None:
    """Serialise a :class:`SimBundle` to ``outdir``.

    Parameters
    ----------
    bundle:
        Bundle to persist.
    outdir:
        Destination directory. It will be created if necessary.
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / "meta.json").write_text(
        json.dumps(asdict(bundle.meta), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (outdir / "mesh.json").write_text(
        json.dumps(asdict(bundle.mesh), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(outdir / "times.npy", np.asarray(bundle.times, dtype=float))

    manifest: dict[str, dict[str, List[str]]] = {
        "node_fields": {},
        "cell_fields": {},
    }
    for attr in ("node_fields", "cell_fields"):
        base_dir = outdir / attr
        base_dir.mkdir(parents=True, exist_ok=True)
        fields: Dict[str, List[np.ndarray]] = getattr(bundle, attr)
        for field_name, series in fields.items():
            stored: list[str] = []
            for idx, arr in enumerate(series):
                file_name = f"{field_name}_{idx:03d}.npy"
                path = base_dir / file_name
                np.save(path, arr)
                stored.append(file_name)
            manifest[attr][field_name] = stored

    (outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def from_disk(outdir: Path) -> SimBundle:
    """Load a :class:`SimBundle` from ``outdir``."""

    outdir = Path(outdir)
    meta = SimMeta(**json.loads((outdir / "meta.json").read_text(encoding="utf-8")))
    mesh = MeshInfo(**json.loads((outdir / "mesh.json").read_text(encoding="utf-8")))
    times = np.load(outdir / "times.npy")

    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    node_fields = _load_field_series(
        outdir / "node_fields", manifest.get("node_fields", {})
    )
    cell_fields = _load_field_series(
        outdir / "cell_fields", manifest.get("cell_fields", {})
    )

    return SimBundle(
        meta=meta,
        mesh=mesh,
        times=times,
        node_fields=node_fields,
        cell_fields=cell_fields,
    )


def _load_field_series(
    base_dir: Path, manifest: dict[str, list[str]]
) -> dict[str, list[np.ndarray]]:
    series_map: dict[str, list[np.ndarray]] = {}
    for field_name, file_names in manifest.items():
        data_list: list[np.ndarray] = []
        for file_name in file_names:
            path = base_dir / file_name
            data_list.append(np.load(path))
        series_map[field_name] = data_list
    return series_map


__all__ = [
    "SimMeta",
    "MeshInfo",
    "TimeSeriesInfo",
    "SimBundle",
    "validate_bundle",
    "infer_units",
    "to_disk",
    "from_disk",
]
