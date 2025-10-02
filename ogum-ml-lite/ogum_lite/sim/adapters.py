"""Adapters to ingest simulation data into the canonical schema."""

from __future__ import annotations

import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .schema import MeshInfo, SimBundle, SimMeta, infer_units

_TIME_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _require_meshio():
    try:
        import meshio  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(
            "meshio is required for VTK/XDMF ingestion. Install via 'pip install "
            '"ogum-ml[sim]"\'.'
        ) from exc
    return meshio


def load_vtk_series(dir_path: Path, pattern: str = "*.vtu") -> SimBundle:
    """Load a directory of VTK files into a :class:`SimBundle`."""

    meshio = _require_meshio()
    paths = sorted(Path(dir_path).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No VTK files matching {pattern} under {dir_path}")

    times: list[float] = []
    node_fields: dict[str, list[np.ndarray]] = defaultdict(list)
    cell_fields: dict[str, list[np.ndarray]] = defaultdict(list)
    mesh_info: MeshInfo | None = None
    units: dict[str, str] = {"time": "s"}

    for idx, path in enumerate(paths):
        mesh = meshio.read(path)
        if mesh_info is None:
            mesh_info = _mesh_info_from_mesh(mesh)
        times.append(_extract_time(mesh, path, default=float(idx)))
        _ingest_point_data(mesh.point_data, node_fields, units)
        _ingest_cell_data(mesh.cell_data, cell_fields, units)

    if mesh_info is None:
        mesh_info = MeshInfo(num_nodes=0, num_cells=0)

    bundle = SimBundle(
        meta=SimMeta(
            sim_id=Path(dir_path).name or "vtk_series",
            solver="vtk",
            units=units,
        ),
        mesh=mesh_info,
        times=_sorted_times(times, node_fields, cell_fields),
        node_fields=node_fields,
        cell_fields=cell_fields,
    )
    return infer_units(bundle)


def load_xdmf(xdmf_path: Path) -> SimBundle:
    """Load a single XDMF file into a :class:`SimBundle`."""

    meshio = _require_meshio()
    mesh = meshio.read(xdmf_path)
    units: dict[str, str] = {"time": "s"}

    times = [_extract_time(mesh, xdmf_path, default=0.0)]
    node_fields: dict[str, list[np.ndarray]] = defaultdict(list)
    cell_fields: dict[str, list[np.ndarray]] = defaultdict(list)

    _ingest_point_data(mesh.point_data, node_fields, units)
    _ingest_cell_data(mesh.cell_data, cell_fields, units)

    bundle = SimBundle(
        meta=SimMeta(sim_id=Path(xdmf_path).stem, solver="xdmf", units=units),
        mesh=_mesh_info_from_mesh(mesh),
        times=np.asarray(times, dtype=float),
        node_fields=node_fields,
        cell_fields=cell_fields,
    )
    return infer_units(bundle)


def load_csv_timeseries(csv_path: Path) -> SimBundle:
    """Load scalar time series from a CSV file into a :class:`SimBundle`."""

    df = pd.read_csv(csv_path)
    time_col = _find_time_column(df.columns)
    if time_col is None:
        raise ValueError("CSV must contain a time column (time_s, time, or t)")

    times = np.asarray(df[time_col], dtype=float)
    other_columns = [col for col in df.columns if col != time_col]
    if not other_columns:
        raise ValueError("CSV must contain at least one field besides time")

    node_fields: dict[str, list[np.ndarray]] = {}
    units: dict[str, str] = {"time": "s"}

    for column in other_columns:
        canonical, values, unit = _canonicalise_series(
            column, df[column].to_numpy(dtype=float)
        )
        node_fields[canonical] = [np.asarray([value], dtype=float) for value in values]
        units[canonical] = unit

    bundle = SimBundle(
        meta=SimMeta(sim_id=Path(csv_path).stem, solver="csv", units=units),
        mesh=MeshInfo(num_nodes=0, num_cells=0),
        times=times,
        node_fields=node_fields,
        cell_fields={},
    )
    return infer_units(bundle)


def load_fenicsx(*_: object, **__: object) -> Optional[SimBundle]:
    """Stub loader for FEniCSx users."""

    try:
        import dolfinx  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        warnings.warn(
            "FEniCSx not installed. Install dolfinx to enable this loader.",
            stacklevel=2,
        )
        return None

    warnings.warn("FEniCSx integration is not yet implemented.", stacklevel=2)
    return None


def load_abaqus_odb(*_: object, **__: object) -> Optional[SimBundle]:
    """Stub loader for Abaqus ODB files."""

    try:
        import odbAccess  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        warnings.warn(
            (
                "Abaqus Python environment not detected. Run inside Abaqus to "
                "enable this loader."
            ),
            stacklevel=2,
        )
        return None

    warnings.warn("Abaqus ODB ingestion is not yet implemented.", stacklevel=2)
    return None


def _mesh_info_from_mesh(mesh: object) -> MeshInfo:
    points = getattr(mesh, "points", np.empty((0, 0)))
    cells = getattr(mesh, "cells", [])
    num_nodes = int(points.shape[0]) if isinstance(points, np.ndarray) else 0
    if cells:
        first = cells[0]
        cell_type = getattr(first, "type", None)
        cell_data = getattr(first, "data", np.empty((0, 0)))
        num_cells = int(cell_data.shape[0]) if isinstance(cell_data, np.ndarray) else 0
    else:
        cell_type = None
        num_cells = 0
    dimension = (
        int(points.shape[1]) if isinstance(points, np.ndarray) and points.size else None
    )
    return MeshInfo(
        num_nodes=num_nodes,
        num_cells=num_cells,
        cell_type=cell_type,
        dimension=dimension,
    )


def _extract_time(mesh: object, path: Path, default: float) -> float:
    field_data = getattr(mesh, "field_data", {}) or {}
    for key in ("time", "Time", "TimeValue", "TIME"):
        if key in field_data:
            value = np.asarray(field_data[key]).ravel()
            if value.size:
                try:
                    return float(value[0])
                except (TypeError, ValueError):
                    continue
    match = _TIME_PATTERN.findall(path.stem)
    if match:
        try:
            return float(match[-1])
        except ValueError:
            pass
    return default


def _ingest_point_data(
    point_data: Dict[str, np.ndarray],
    node_fields: dict[str, list[np.ndarray]],
    units: dict[str, str],
) -> None:
    for name, values in point_data.items():
        canonical, arr, unit = _canonicalise_field(name, values)
        node_fields[canonical].append(arr)
        units.setdefault(canonical, unit)


def _ingest_cell_data(
    cell_data: Dict[str, Iterable[np.ndarray]],
    cell_fields: dict[str, list[np.ndarray]],
    units: dict[str, str],
) -> None:
    for name, blocks in cell_data.items():
        for block in blocks:
            canonical, arr, unit = _canonicalise_field(name, block)
            cell_fields[canonical].append(arr)
            units.setdefault(canonical, unit)
            break  # only consider first block for canonical bundle


def _canonicalise_field(name: str, values: np.ndarray) -> tuple[str, np.ndarray, str]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim > 1 and arr.shape[-1] > 1:
        arr = np.linalg.norm(arr, axis=-1)
    canonical, unit = _canonical_field_and_unit(name)
    if canonical == "temp_C" and unit == "K":
        arr = arr - 273.15
        unit = "C"
    elif canonical == "temp_C" and unit == "F":
        arr = (arr - 32.0) * 5.0 / 9.0
        unit = "C"
    elif canonical == "von_mises_MPa" and unit == "Pa":
        arr = arr / 1e6
        unit = "MPa"
    return canonical, arr, unit


def _canonicalise_series(name: str, values: np.ndarray) -> tuple[str, np.ndarray, str]:
    arr = np.asarray(values, dtype=float)
    canonical, unit = _canonical_field_and_unit(name)
    if canonical == "temp_C" and unit == "K":
        arr = arr - 273.15
        unit = "C"
    elif canonical == "temp_C" and unit == "F":
        arr = (arr - 32.0) * 5.0 / 9.0
        unit = "C"
    elif canonical == "von_mises_MPa" and unit == "Pa":
        arr = arr / 1e6
        unit = "MPa"
    return canonical, arr, unit


def _canonical_field_and_unit(name: str) -> tuple[str, str]:
    lower = name.lower()
    if lower in {"t", "temperature"} or "temp" in lower:
        if any(token in lower for token in ["_k", "[k", " kelvin"]):
            return "temp_C", "K"
        if lower.endswith("_f"):
            return "temp_C", "F"
        return "temp_C", "C"
    if "sigma" in lower and "vm" in lower:
        if "mpa" in lower:
            return "von_mises_MPa", "MPa"
        if "pa" in lower:
            return "von_mises_MPa", "Pa"
        return "von_mises_MPa", "MPa"
    if lower == "e" or "electric" in lower or lower.startswith("e_"):
        return "E_V_per_m", "V/m"
    return name, "unknown"


def _find_time_column(columns: Iterable[str]) -> Optional[str]:
    for candidate in ("time_s", "time", "t"):
        if candidate in columns:
            return candidate
    for column in columns:
        if column.lower() in {"time[s]", "tempo", "tempo_s"}:
            return column
    return None


def _sorted_times(
    times: list[float],
    node_fields: dict[str, list[np.ndarray]],
    cell_fields: dict[str, list[np.ndarray]],
) -> np.ndarray:
    order = np.argsort(times)
    sorted_times = np.asarray(times, dtype=float)[order]
    if not np.all(order == np.arange(len(times))):
        for field_dict in (node_fields, cell_fields):
            for key, series in field_dict.items():
                field_dict[key] = [series[idx] for idx in order]
    return sorted_times


__all__ = [
    "load_vtk_series",
    "load_xdmf",
    "load_csv_timeseries",
    "load_fenicsx",
    "load_abaqus_odb",
]
