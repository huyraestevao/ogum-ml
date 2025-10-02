"""Tests for the comparison loaders module."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
from ogum_lite.compare import loaders


def test_find_run_root_directory(tmp_path: Path) -> None:
    target = tmp_path / "run"
    target.mkdir()
    assert loaders.find_run_root(target) == target


def test_find_run_root_zip(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    inner = root / "export"
    inner.mkdir(parents=True)
    (inner / "preset.yaml").write_text("foo: 1", encoding="utf-8")
    archive = tmp_path / "artifact.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(inner / "preset.yaml", arcname="export/preset.yaml")
    extracted = loaders.find_run_root(archive)
    assert extracted.is_dir()
    assert (extracted / "preset.yaml").exists()


def test_scan_run_detects_artifacts(tmp_path: Path) -> None:
    (tmp_path / "preset.yaml").write_text("foo: 1", encoding="utf-8")
    (tmp_path / "msc.csv").write_text("mse_global\n1.0\n", encoding="utf-8")
    manifest = loaders.scan_run(tmp_path)
    assert manifest["presets"].name == "preset.yaml"
    assert manifest["msc_csv"].name == "msc.csv"


def test_loaders_helpers(tmp_path: Path) -> None:
    json_path = tmp_path / "model_card.json"
    json_path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    yaml_path = tmp_path / "preset.yaml"
    yaml_path.write_text("foo: bar", encoding="utf-8")
    csv_path = tmp_path / "table.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(csv_path, index=False)

    assert loaders.load_json(json_path) == {"a": 1}
    assert loaders.load_yaml(yaml_path) == {"foo": "bar"}
    csv = loaders.load_csv(csv_path)
    assert list(csv.columns) == ["x"]
