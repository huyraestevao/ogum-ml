"""Ogum Lite package initialization."""

from .blaine import fit_blaine_by_segments, fit_blaine_segment
from .features import (
    aggregate_timeseries,
    arrhenius_feature_table,
    build_feature_store,
    build_feature_table,
    build_stage_feature_tables,
    finite_diff,
    segment_feature_table,
    theta_features,
)
from .maps import (
    prepare_segment_heatmap,
    render_segment_heatmap,
    save_heatmap,
)
from .mechanism import detect_mechanism_change
from .ml_hooks import (
    kmeans_explore,
    predict_from_artifact,
    train_classifier,
    train_regressor,
)
from .segmentation import segment_dataframe
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
    "arrhenius_feature_table",
    "build_feature_table",
    "build_feature_store",
    "build_stage_feature_tables",
    "detect_mechanism_change",
    "fit_blaine_by_segments",
    "fit_blaine_segment",
    "build_master_curve",
    "finite_diff",
    "kmeans_explore",
    "prepare_segment_heatmap",
    "predict_from_artifact",
    "render_segment_heatmap",
    "save_heatmap",
    "segment_dataframe",
    "segment_feature_table",
    "score_activation_energies",
    "theta_features",
    "train_classifier",
    "train_regressor",
]
