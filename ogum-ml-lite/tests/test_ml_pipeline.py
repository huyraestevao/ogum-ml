"""End-to-end smoke tests for the ML pipelines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ogum_lite.features import build_feature_table
from ogum_lite.ml_hooks import (
    kmeans_explore,
    predict_from_artifact,
    train_classifier,
    train_regressor,
)


def _make_long_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    records: list[dict[str, object]] = []
    sample_ids = [f"S{i:02d}" for i in range(8)]
    techniques = [
        "Conventional",
        "UHS",
        "FS",
        "SPS",
        "Hybrid",
        "Laser",
        "HIP",
        "FAST",
    ]
    for idx, (sample_id, technique) in enumerate(
        zip(sample_ids, techniques, strict=True)
    ):
        times = np.linspace(0.0, 1800.0, 60)
        base_temp = 650.0 + 20.0 * idx
        heating_rate = 0.25 + 0.05 * idx
        temps = base_temp + heating_rate * times
        dens_curve = 0.15 + 0.8 / (1 + np.exp(-(times - 900.0) / 120.0))
        noise = rng.normal(scale=0.01, size=times.size)
        dens = np.clip(dens_curve + 0.02 * idx + noise, 0.0, 0.995)
        for t, temp, rho in zip(times, temps, dens, strict=True):
            records.append(
                {
                    "sample_id": sample_id,
                    "time_s": float(t),
                    "temp_C": float(temp),
                    "rho_rel": float(rho),
                    "technique": technique,
                }
            )
    return pd.DataFrame.from_records(records)


def _make_feature_table() -> pd.DataFrame:
    df_long = _make_long_dataframe()
    features = build_feature_table(df_long, ea_kj_list=[200.0, 300.0, 400.0])
    target_rows = []
    for idx, sample_id in enumerate(sorted(features["sample_id"].tolist())):
        target_rows.append(
            {
                "sample_id": sample_id,
                "technique": df_long.loc[
                    df_long["sample_id"] == sample_id, "technique"
                ].iloc[0],
                "T90_C": 720.0 + 12.0 * idx,
            }
        )
    target_df = pd.DataFrame(target_rows)
    return features.merge(target_df, on="sample_id", how="left")


def _default_features() -> list[str]:
    return [
        "heating_rate_med_C_per_s",
        "T_max_C",
        "y_final",
        "t_to_90pct_s",
        "theta_Ea_200kJ",
        "theta_Ea_300kJ",
    ]


def test_train_classifier_persists_artifacts(tmp_path: Path) -> None:
    feature_table = _make_feature_table()
    outdir = tmp_path / "cls"
    result = train_classifier(
        feature_table,
        target_col="technique",
        group_col="sample_id",
        feature_cols=_default_features(),
        outdir=outdir,
    )
    artifacts = result["artifacts"]
    for key in ("model", "feature_cols", "target", "model_card"):
        assert artifacts[key].exists(), f"Missing artifact: {key}"
    cv_metrics = result["cv"]
    assert "accuracy_mean" in cv_metrics
    assert "f1_macro_mean" in cv_metrics


def test_train_regressor_and_predict(tmp_path: Path) -> None:
    feature_table = _make_feature_table()
    outdir = tmp_path / "reg"
    result = train_regressor(
        feature_table,
        target_col="T90_C",
        group_col="sample_id",
        feature_cols=_default_features(),
        outdir=outdir,
    )
    model_path = result["artifacts"]["model"]
    preds = predict_from_artifact(model_path, feature_table)
    assert set(preds.columns) == {"sample_id", "y_pred"}
    assert len(preds) == len(feature_table)


def test_kmeans_explore(tmp_path: Path) -> None:
    feature_table = _make_feature_table()
    clusters = kmeans_explore(feature_table, _default_features(), k=3)
    assert clusters["cluster"].nunique() == 3
    assert set(clusters.columns) == {"sample_id", "cluster"}
