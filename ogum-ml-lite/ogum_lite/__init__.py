"""Ogum Lite package initialization."""

from .features import (
    aggregate_timeseries,
    build_feature_table,
    finite_diff,
    theta_features,
)
from .theta_msc import (
    R_GAS_CONSTANT,
    MasterCurveResult,
    OgumLite,
    build_master_curve,
    score_activation_energies,
)

__all__ = [
    "MasterCurveResult",
    "OgumLite",
    "R_GAS_CONSTANT",
    "aggregate_timeseries",
    "build_feature_table",
    "build_master_curve",
    "finite_diff",
    "score_activation_energies",
    "theta_features",
]
