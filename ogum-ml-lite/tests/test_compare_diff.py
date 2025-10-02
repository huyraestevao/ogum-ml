"""Tests for semantic diff helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from ogum_lite.compare import diff_core


def _write_csv(path: Path, data: pd.DataFrame) -> None:
    data.to_csv(path, index=False)


def test_diff_presets_detects_changes() -> None:
    left = {"ea": [1, 2], "profile": "alpha"}
    right = {"ea": [2, 3], "profile": "beta", "new": 1}
    diff = diff_core.diff_presets(left, right)
    assert "profile" in diff["changed"]
    assert diff["added"] == {"new": 1}
    assert diff["lists"]["ea"]["only_a"] == [1]


def test_diff_msc_metrics(tmp_path: Path) -> None:
    a_csv = tmp_path / "msc_a.csv"
    b_csv = tmp_path / "msc_b.csv"
    _write_csv(
        a_csv,
        pd.DataFrame(
            {
                "mse_global": [1.0, 1.0],
                "mse_segmented": [0.5, 0.5],
                "theta_norm": [0.0, 1.0],
                "prediction": [0.1, 0.2],
                "target": [0.1, 0.2],
            }
        ),
    )
    _write_csv(
        b_csv,
        pd.DataFrame(
            {
                "mse_global": [0.8, 0.8],
                "mse_segmented": [0.4, 0.4],
                "theta_norm": [0.0, 1.0],
                "prediction": [0.2, 0.3],
                "target": [0.1, 0.2],
            }
        ),
    )
    diff = diff_core.diff_msc(a_csv, b_csv)
    assert diff["metrics"]["mse_global"]["delta"] == -0.19999999999999996
    assert diff["curve"]["n_points"] == 2


def test_diff_segments(tmp_path: Path) -> None:
    a_csv = tmp_path / "segments_a.csv"
    b_csv = tmp_path / "segments_b.csv"
    _write_csv(
        a_csv,
        pd.DataFrame(
            {
                "sample_id": ["A"],
                "segment_id": [1],
                "n_est": [10],
            }
        ),
    )
    _write_csv(
        b_csv,
        pd.DataFrame(
            {
                "sample_id": ["A"],
                "segment_id": [1],
                "n_est": [12],
            }
        ),
    )
    diff = diff_core.diff_segments(a_csv, b_csv)
    assert diff["changed"][("A", 1)]["n_est"]["b"] == 12.0


def test_diff_mechanism(tmp_path: Path) -> None:
    a_csv = tmp_path / "mechanism_a.csv"
    b_csv = tmp_path / "mechanism_b.csv"
    _write_csv(
        a_csv,
        pd.DataFrame(
            {
                "sample_id": ["S"],
                "has_change": [False],
                "tau": [10.0],
            }
        ),
    )
    _write_csv(
        b_csv,
        pd.DataFrame(
            {
                "sample_id": ["S"],
                "has_change": [True],
                "tau": [10.0],
            }
        ),
    )
    diff = diff_core.diff_mechanism(a_csv, b_csv)
    assert diff["changed"][("S",)]["has_change"]["b"] is True


def test_diff_ml(tmp_path: Path) -> None:
    card_a = tmp_path / "model_a.json"
    card_b = tmp_path / "model_b.json"
    card_a.write_text(
        json.dumps({"algorithm": "rf", "hyperparameters": {"max_depth": 3}}),
        encoding="utf-8",
    )
    card_b.write_text(
        json.dumps({"algorithm": "rf", "hyperparameters": {"max_depth": 4}}),
        encoding="utf-8",
    )
    cv_a = tmp_path / "cv_a.json"
    cv_b = tmp_path / "cv_b.json"
    cv_a.write_text(json.dumps({"accuracy": 0.8}), encoding="utf-8")
    cv_b.write_text(json.dumps({"accuracy": 0.9}), encoding="utf-8")

    diff = diff_core.diff_ml(card_a, card_b, cv_a, cv_b)
    assert diff["hyperparameters"]["max_depth"]["b"] == 4
    assert diff["metrics"]["accuracy"]["delta"] == 0.09999999999999998


def test_compose_diff_summary() -> None:
    summary = diff_core.compose_diff_summary(
        presets={"added": {"a": 1}},
        msc={"metrics": {"mse": {"metric": "mse", "a": 1, "b": 0.5}}},
        segments={"changed": {"foo": {}}},
    )
    assert summary["alerts"]
    assert summary["kpis"][0]["metric"] == "mse"
