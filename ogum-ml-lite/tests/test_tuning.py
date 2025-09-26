from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ogum_lite.cli import cmd_ml_report
from ogum_lite.ml_hooks import random_search_classifier, random_search_regressor


@pytest.fixture()
def synthetic_classifier_df() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    groups = np.repeat(np.arange(6), 5)
    n_samples = groups.size
    x1 = rng.normal(size=n_samples)
    x2 = rng.normal(size=n_samples)
    logits = 0.8 * x1 - 0.5 * x2
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)
    return pd.DataFrame(
        {
            "sample_id": groups,
            "f1": x1,
            "f2": x2,
            "target": y,
        }
    )


@pytest.fixture()
def synthetic_regressor_df() -> pd.DataFrame:
    rng = np.random.default_rng(321)
    groups = np.repeat(np.arange(5), 6)
    n_samples = groups.size
    x1 = rng.normal(size=n_samples)
    x2 = rng.normal(size=n_samples)
    noise = rng.normal(scale=0.1, size=n_samples)
    y = 3.0 * x1 - 2.0 * x2 + noise
    return pd.DataFrame(
        {
            "sample_id": groups,
            "f1": x1,
            "f2": x2,
            "target": y,
        }
    )


def test_random_search_classifier_creates_artifacts(
    tmp_path: Path, synthetic_classifier_df: pd.DataFrame
) -> None:
    outdir = tmp_path / "cls"
    result = random_search_classifier(
        synthetic_classifier_df,
        target_col="target",
        group_col="sample_id",
        feature_cols=["f1", "f2"],
        outdir=outdir,
        n_iter=5,
        cv_splits=3,
        random_state=7,
    )

    assert (outdir / "classifier_tuned.joblib").exists()
    assert (outdir / "param_grid.json").exists()
    assert (outdir / "cv_results.json").exists()
    card_path = outdir / "model_card.json"
    assert card_path.exists()
    payload = json.loads(card_path.read_text(encoding="utf-8"))
    assert "best_params" in payload
    assert result["cv"]["accuracy_mean"] >= 0.0


def test_random_search_regressor_and_report(
    tmp_path: Path,
    synthetic_regressor_df: pd.DataFrame,
    capsys: pytest.CaptureFixture[str],
) -> None:
    outdir = tmp_path / "reg"
    table_path = tmp_path / "features.csv"
    synthetic_regressor_df.to_csv(table_path, index=False)

    result = random_search_regressor(
        synthetic_regressor_df,
        target_col="target",
        group_col="sample_id",
        feature_cols=["f1", "f2"],
        outdir=outdir,
        n_iter=5,
        cv_splits=3,
        random_state=11,
    )
    assert "mae_mean" in result["cv"]

    args = Namespace(
        table=table_path,
        target="target",
        group_col="sample_id",
        model=outdir / "regressor_tuned.joblib",
        outdir=outdir,
        n_repeats=3,
        random_state=0,
        notes="Smoke test",
    )
    cmd_ml_report(args)
    captured = capsys.readouterr()
    assert "Report metrics" in captured.out
    report_path = outdir / "report.html"
    assert report_path.exists()
    html = report_path.read_text(encoding="utf-8")
    assert "MAE" in html
    assert (outdir / "regression_scatter.png").exists()
    assert (outdir / "feature_importance.png").exists()
