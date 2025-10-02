"""Smoke tests for the benchmark experimentation utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from ogum_lite.ml_experiments import (
    compare_models,
    list_available_models,
    run_benchmark_matrix,
    run_experiment,
)


def _example_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["g1", "g1", "g2", "g2", "g3", "g3"],
            "feature_num": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "feature_cat": ["a", "b", "a", "b", "a", "b"],
            "technique": ["press", "press", "mill", "mill", "press", "press"],
        }
    )


def test_available_models_includes_rf() -> None:
    models = list_available_models("cls")
    assert "rf" in models


def test_run_experiment_and_compare(tmp_path: Path) -> None:
    dataframe = _example_dataframe()
    outdir = tmp_path / "artifacts"

    result = run_experiment(
        df_features=dataframe,
        target_col="technique",
        group_col="sample_id",
        feature_cols=["feature_num", "feature_cat"],
        model_key="rf",
        outdir=outdir,
        task="cls",
    )
    assert not result["skipped"]
    artifacts = result["artifacts"]
    assert artifacts["model"].exists()
    assert artifacts["cv_metrics"].exists()

    feature_sets = {"basic": ["feature_num", "feature_cat"]}
    matrix_df = run_benchmark_matrix(
        df_features=dataframe,
        task="cls",
        targets=["technique"],
        feature_sets=feature_sets,
        models=["rf"],
        group_col="sample_id",
        base_outdir=outdir,
    )
    bench_csv = outdir / "bench_results.csv"
    assert bench_csv.exists()
    assert not matrix_df.empty

    summary = compare_models(bench_csv, "cls")
    assert not summary.empty
    assert (outdir / "bench_summary.csv").exists()
    assert (outdir / "ranking.png").exists()
