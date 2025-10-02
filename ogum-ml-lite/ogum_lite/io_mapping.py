"""I/O helpers to map heterogeneous spreadsheets into Ogum's canonical schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional

import pandas as pd
import yaml

from . import __path__ as OGUM_PACKAGE_PATH

TimeUnit = Literal["s", "min"]
TemperatureUnit = Literal["C", "K"]


@dataclass(frozen=True)
class ColumnMap:
    """Mapping between heterogeneous column names and Ogum's canonical schema."""

    sample_id: str
    time_col: str
    temp_col: str
    y_col: str
    composition: str | None
    technique: str | None
    composition_default: Optional[str] = None
    technique_default: Optional[str] = None
    tech_comment: Optional[str] = None
    user: Optional[str] = None
    timestamp: Optional[str] = None
    time_unit: TimeUnit = "s"
    temp_unit: TemperatureUnit = "C"
    extra_metadata: dict[str, str] = field(default_factory=dict)


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
    "SPS",
    "Flash",
    "UHS",
    "Two-Step",
    "Cold",
    "Heating-Only",
    "Outro",
]


def _profiles_dir() -> Path:
    package_root = Path(list(OGUM_PACKAGE_PATH)[0])
    return package_root.parent / "app" / "config" / "profiles"


def load_technique_profiles(directory: str | Path | None = None) -> Dict[str, dict]:
    """Load technique presets from YAML files bundled with the application."""

    if directory is None:
        directory = _profiles_dir()
    path = Path(directory)
    profiles: Dict[str, dict] = {}
    if not path.exists():
        return profiles

    for yaml_path in sorted(path.glob("*.yaml")):
        try:
            data = yaml.safe_load(yaml_path.read_text())
        except Exception:  # pragma: no cover - defensive fallback
            continue
        if not isinstance(data, dict):
            continue
        profile = data.get("technique_profile")
        if not isinstance(profile, dict):
            continue
        technique_name = profile.get("name") or data.get("name")
        if not technique_name:
            continue
        technique_name = str(technique_name)
        profiles[technique_name] = profile
    return profiles


TECHNIQUE_PROFILES = load_technique_profiles()


CANONICAL_COLUMNS = (
    "sample_id",
    "time_s",
    "temp_C",
    "response",
    "composition",
    "technique",
)


def _normalise(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace("/", "")
    )


def _detect_header_start(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        for idx, line in enumerate(handle):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if any(delim in stripped for delim in (",", ";", "\t")):
                return idx
    return 0


def read_table(path: str | Path) -> pd.DataFrame:
    """Read delimited or spreadsheet files handling metadata prologues."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        header_row = _detect_header_start(path)
        return pd.read_csv(
            path,
            sep=None,
            engine="python",
            comment="#",
            skip_blank_lines=True,
            header=header_row,
        )
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


def infer_mapping(
    df: pd.DataFrame,
    *,
    default_composition: str | None = None,
    default_technique: str | None = None,
    tech_comment: str | None = None,
    user: str | None = None,
    timestamp: str | None = None,
    extra_metadata: dict[str, str] | None = None,
) -> ColumnMap:
    """Infer :class:`ColumnMap` for a dataframe based on column aliases."""

    sample_id = _match_column(df, "sample_id")
    time_col = _match_column(df, "time")
    temp_col = _match_column(df, "temperature")
    y_col = _match_column(df, "response")
    composition_default = None
    try:
        composition = _match_column(df, "composition")
    except KeyError:
        if default_composition is None:
            raise
        composition = None
        composition_default = default_composition

    technique_default = None
    try:
        technique = _match_column(df, "technique")
    except KeyError:
        if default_technique is None:
            raise
        technique = None
        technique_default = default_technique

    time_unit = _detect_time_unit(time_col)
    temp_unit = _detect_temperature_unit(temp_col)

    return ColumnMap(
        sample_id=sample_id,
        time_col=time_col,
        temp_col=temp_col,
        y_col=y_col,
        composition=composition,
        technique=technique,
        composition_default=composition_default,
        technique_default=technique_default,
        tech_comment=tech_comment,
        user=user,
        timestamp=timestamp,
        extra_metadata=dict(extra_metadata or {}),
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

    if cmap.composition and cmap.composition in dataframe.columns:
        result["composition"] = dataframe[cmap.composition].astype(str)
    elif cmap.composition_default is not None:
        result["composition"] = str(cmap.composition_default)
    else:
        raise KeyError("Composition column missing and no default provided")

    if cmap.technique and cmap.technique in dataframe.columns:
        tech_values = (
            dataframe[cmap.technique]
            .fillna(cmap.technique_default or TECHNIQUE_CHOICES[0])
            .astype(str)
        )
    elif cmap.technique_default is not None:
        tech_values = pd.Series(cmap.technique_default, index=result.index, dtype=str)
    else:
        raise KeyError("Technique column missing and no default provided")

    valid_choices = set(TECHNIQUE_CHOICES)
    if not tech_values.isin(valid_choices).all():
        invalid = sorted(set(tech_values) - valid_choices)
        raise ValueError(
            "Invalid technique detected. Expected one of "
            f"{', '.join(TECHNIQUE_CHOICES)}; got {invalid}"
        )
    result["technique"] = tech_values

    if cmap.tech_comment:
        result["tech_comment"] = str(cmap.tech_comment)
    if cmap.user:
        result["import_user"] = str(cmap.user)
    if cmap.timestamp:
        result["import_timestamp"] = str(cmap.timestamp)
    for key, value in cmap.extra_metadata.items():
        if key in result.columns:
            continue
        result[key] = value

    result["response"] = y_series
    result["rho_rel"] = y_series
    result["y"] = y_series

    return result


__all__ = [
    "ColumnMap",
    "ALIASES",
    "TECHNIQUE_CHOICES",
    "read_table",
    "load_technique_profiles",
    "TECHNIQUE_PROFILES",
    "CANONICAL_COLUMNS",
    "infer_mapping",
    "apply_mapping",
]
