"""I/O helpers to map heterogeneous spreadsheets into Ogum's canonical schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal

import pandas as pd

TimeUnit = Literal["s", "min"]
TemperatureUnit = Literal["C", "K"]


@dataclass(frozen=True)
class ColumnMap:
    """Mapping between heterogeneous column names and Ogum's canonical schema."""

    sample_id: str
    time_col: str
    temp_col: str
    y_col: str
    composition: str
    technique: str
    time_unit: TimeUnit = "s"
    temp_unit: TemperatureUnit = "C"


ALIASES: Dict[str, Iterable[str]] = {
    "sample_id": ["sample_id", "id", "sample", "amostra", "run_id"],
    "time": [
        "time_s",
        "time",
        "tempo",
        "t_s",
        "time_min",
        "t",
        "tempo_min",
    ],
    "temperature": [
        "temp_c",
        "temperature",
        "temp",
        "t_c",
        "temp_k",
        "temperature_c",
        "temperature_k",
        "t_k",
    ],
    "response": [
        "rho_rel",
        "densification",
        "densidade_relativa",
        "shrinkage_rel",
        "y",
        "response",
    ],
    "composition": ["composition", "alloy", "material", "comp", "liga"],
    "technique": ["technique", "process", "route", "tecnica", "method"],
}


TECHNIQUE_CHOICES = [
    "Conventional",
    "UHS",
    "Flash",
    "SPS",
    "Two-Step",
    "Cold",
    "HeatingOnly",
]


def _normalise(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace("/", "")
    )


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV/XLS/XLSX files using pandas based on the file extension."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file extension: {suffix}")


def _match_column(df: pd.DataFrame, key: str) -> str:
    aliases = ALIASES.get(key, [])
    norm_aliases = {_normalise(alias): alias for alias in aliases}
    for column in df.columns:
        norm = _normalise(str(column))
        if norm in norm_aliases:
            return str(column)
    raise KeyError(f"Unable to infer column for '{key}'")


def _detect_time_unit(column_name: str) -> TimeUnit:
    lowered = column_name.lower()
    if "min" in lowered and not lowered.endswith("s"):
        return "min"
    if lowered.endswith("_min") or lowered.endswith("min"):
        return "min"
    return "s"


def _detect_temperature_unit(column_name: str) -> TemperatureUnit:
    lowered = column_name.lower()
    if lowered.endswith("k") or "kelvin" in lowered:
        return "K"
    if lowered.endswith("_k"):
        return "K"
    return "C"


def infer_mapping(df: pd.DataFrame) -> ColumnMap:
    """Infer :class:`ColumnMap` for a dataframe based on column aliases."""

    sample_id = _match_column(df, "sample_id")
    time_col = _match_column(df, "time")
    temp_col = _match_column(df, "temperature")
    y_col = _match_column(df, "response")
    composition = _match_column(df, "composition")
    technique = _match_column(df, "technique")

    time_unit = _detect_time_unit(time_col)
    temp_unit = _detect_temperature_unit(temp_col)

    return ColumnMap(
        sample_id=sample_id,
        time_col=time_col,
        temp_col=temp_col,
        y_col=y_col,
        composition=composition,
        technique=technique,
        time_unit=time_unit,
        temp_unit=temp_unit,
    )


def apply_mapping(df: pd.DataFrame, cmap: ColumnMap) -> pd.DataFrame:
    """Project ``df`` into Ogum's canonical schema using ``cmap``."""

    required_columns = {
        cmap.sample_id,
        cmap.time_col,
        cmap.temp_col,
        cmap.y_col,
    }
    missing = required_columns - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise KeyError(f"Missing columns required by mapping: {missing_cols}")

    dataframe = df.copy()

    time_series = dataframe[cmap.time_col].astype(float)
    if cmap.time_unit == "min":
        time_series = time_series * 60.0

    temp_series = dataframe[cmap.temp_col].astype(float)
    if cmap.temp_unit == "K":
        temp_series = temp_series - 273.15

    y_series = dataframe[cmap.y_col].astype(float)

    result = pd.DataFrame(
        {
            "sample_id": dataframe[cmap.sample_id].astype(str),
            "time_s": time_series,
            "temp_C": temp_series,
            "y": y_series,
        }
    )

    if cmap.composition in dataframe.columns:
        result["composition"] = dataframe[cmap.composition].astype(str)
    else:
        result["composition"] = cmap.composition

    if cmap.technique in dataframe.columns:
        tech_values = dataframe[cmap.technique].fillna(TECHNIQUE_CHOICES[0]).astype(str)
    else:
        tech_values = pd.Series(cmap.technique, index=result.index, dtype=str)
    if not tech_values.isin(TECHNIQUE_CHOICES).all():
        invalid = sorted(set(tech_values) - set(TECHNIQUE_CHOICES))
        raise ValueError(
            "Invalid technique detected. Expected one of "
            f"{', '.join(TECHNIQUE_CHOICES)}; got {invalid}"
        )
    result["technique"] = tech_values

    return result


__all__ = [
    "ColumnMap",
    "ALIASES",
    "TECHNIQUE_CHOICES",
    "read_table",
    "infer_mapping",
    "apply_mapping",
]
