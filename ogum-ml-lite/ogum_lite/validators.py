"""Validation helpers for Ogum Lite datasets."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class LongRow(BaseModel):
    """Schema for the long-format experimental table."""

    sample_id: int | str
    time_s: float = Field(ge=0)
    temp_C: float = Field(ge=-50, le=2500)
    rho_rel: float | None = None
    shrinkage_rel: float | None = None

    model_config = ConfigDict(extra="ignore")

    @field_validator("time_s", "temp_C", mode="before")
    @classmethod
    def _cast_float(cls, value: Any) -> float:
        if value is None:
            raise ValueError("value cannot be null")
        return float(value)

    @field_validator("rho_rel", "shrinkage_rel", mode="before")
    @classmethod
    def _validate_relative(cls, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        value = float(value)
        if not 0 <= value <= 1:
            raise ValueError("must be between 0 and 1 inclusive")
        return value

    @model_validator(mode="after")
    def _check_targets(self) -> "LongRow":
        if self.rho_rel is None and self.shrinkage_rel is None:
            raise ValueError("expected rho_rel and/or shrinkage_rel per row")
        return self


class FeatureRow(BaseModel):
    """Schema for per-sample feature tables."""

    sample_id: int | str
    heating_rate_med_C_per_s: float = Field(ge=0)
    T_max_C: float = Field(ge=-50, le=2500)
    y_final: float = Field(ge=0, le=1)
    t_to_90pct_s: float = Field(ge=0)
    dy_dt_max: float = Field(ge=0)
    T_at_dy_dt_max_C: float = Field(ge=-50, le=2500)

    model_config = ConfigDict(extra="allow")

    @field_validator(
        "heating_rate_med_C_per_s",
        "T_max_C",
        "y_final",
        "t_to_90pct_s",
        "dy_dt_max",
        "T_at_dy_dt_max_C",
        mode="before",
    )
    @classmethod
    def _cast_numeric(cls, value: Any) -> float:
        if value is None:
            raise ValueError("value cannot be null")
        return float(value)

    @model_validator(mode="after")
    def _normalise_theta_columns(self) -> "FeatureRow":
        for key, value in list(self.__dict__.items()):
            if not key.startswith("theta_Ea_"):
                continue
            if value is None or (isinstance(value, float) and math.isnan(value)):
                setattr(self, key, None)
                continue
            try:
                setattr(self, key, float(value))
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise ValueError(f"{key} must be numeric") from exc
        return self


def _nan_report(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    issues: list[str] = []
    for column in columns:
        if column not in df.columns:
            continue
        pct = float(df[column].isna().mean() * 100)
        if pct > 0:
            issues.append(f"NaN ratio for column '{column}': {pct:.2f}%")
    return issues


def _sample_indices(length: int) -> list[int]:
    if length == 0:
        return []
    step = max(length // 50, 1)
    indices = list(range(0, length, step))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return indices


def validate_long_df(df: pd.DataFrame, *, y_col: str = "rho_rel") -> dict[str, Any]:
    """Validate long-format dataframe and collect issues."""

    issues: list[str] = []
    required_columns = {"sample_id", "time_s", "temp_C"}
    missing = sorted(col for col in required_columns if col not in df.columns)
    if missing:
        issues.append("Missing columns: " + ", ".join(missing))

    target_columns = {y_col, "rho_rel", "shrinkage_rel"} & set(df.columns)
    if not target_columns:
        issues.append(
            "Missing densification column: expected one of "
            + ", ".join(sorted({y_col, "rho_rel", "shrinkage_rel"}))
        )

    issues.extend(
        _nan_report(df, required_columns | {y_col, "rho_rel", "shrinkage_rel"})
    )

    for idx in _sample_indices(len(df)):
        row = df.iloc[idx].to_dict()
        row_payload = {
            "sample_id": row.get("sample_id"),
            "time_s": row.get("time_s"),
            "temp_C": row.get("temp_C"),
            "rho_rel": row.get("rho_rel"),
            "shrinkage_rel": row.get("shrinkage_rel"),
        }
        if y_col in row:
            # Override rho_rel with the selected target column when applicable.
            if y_col == "shrinkage_rel":
                row_payload["shrinkage_rel"] = row.get(y_col)
            else:
                row_payload["rho_rel"] = row.get(y_col)
        try:
            LongRow.model_validate(row_payload)
        except ValidationError as exc:
            for error in exc.errors():
                location = "->".join(str(piece) for piece in error.get("loc", ()))
                issues.append(f"Row {idx}: {location} {error.get('msg')}")

    return {"ok": not issues, "issues": issues}


def validate_feature_df(df: pd.DataFrame) -> dict[str, Any]:
    """Validate per-sample feature dataframe."""

    issues: list[str] = []
    required_columns = {
        "sample_id",
        "heating_rate_med_C_per_s",
        "T_max_C",
        "y_final",
        "t_to_90pct_s",
        "dy_dt_max",
        "T_at_dy_dt_max_C",
    }
    missing = sorted(col for col in required_columns if col not in df.columns)
    if missing:
        issues.append("Missing columns: " + ", ".join(missing))

    issues.extend(
        _nan_report(
            df,
            required_columns
            | {col for col in df.columns if col.startswith("theta_Ea_")},
        )
    )

    for idx in _sample_indices(len(df)):
        row = df.iloc[idx].to_dict()
        try:
            FeatureRow.model_validate(row)
        except ValidationError as exc:
            for error in exc.errors():
                location = "->".join(str(piece) for piece in error.get("loc", ()))
                issues.append(f"Row {idx}: {location} {error.get('msg')}")

    return {"ok": not issues, "issues": issues}
