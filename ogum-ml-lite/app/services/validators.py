"""Validation helpers bridging CLI validators to UX messages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from ogum_lite.validators import validate_feature_df, validate_long_df


@dataclass(slots=True)
class ValidationSummary:
    """Structured validation result for display."""

    ok: bool
    issues: list[str]


def _format_issues(issues: Iterable[str]) -> list[str]:
    return [f"[warn] {issue}" for issue in issues]


def validate_long(path: Path, *, y_col: str = "rho_rel") -> ValidationSummary:
    """Validate a long-format CSV file using the shared validators."""

    df = pd.read_csv(path)
    result: dict[str, Any] = validate_long_df(df, y_col=y_col)
    return ValidationSummary(
        ok=bool(result.get("ok", False)),
        issues=_format_issues(result.get("issues", [])),
    )


def validate_features(path: Path) -> ValidationSummary:
    """Validate the engineered features CSV file."""

    df = pd.read_csv(path)
    result: dict[str, Any] = validate_feature_df(df)
    return ValidationSummary(
        ok=bool(result.get("ok", False)),
        issues=_format_issues(result.get("issues", [])),
    )
