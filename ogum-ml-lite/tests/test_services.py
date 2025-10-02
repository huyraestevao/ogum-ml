"""Tests for the service orchestration layer."""

from __future__ import annotations

import subprocess
from pathlib import Path

from app.services import run_cli
from ogum_lite.ui.workspace import Workspace


def test_run_prep_wraps_orchestrator(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OGUML_TELEMETRY", "0")
    ws = Workspace(tmp_path)

    def fake_run_prep(input_csv: Path, preset: dict, workspace: Workspace) -> Path:
        assert workspace is ws
        return tmp_path / "prep.csv"

    monkeypatch.setattr(run_cli.orchestrator, "run_prep", fake_run_prep)
    result = run_cli.run_prep(tmp_path / "input.csv", {}, ws)
    assert result.outputs["prep_csv"].name == "prep.csv"


def test_export_onnx_invokes_cli(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OGUML_TELEMETRY", "0")
    ws = Workspace(tmp_path)

    def fake_execute(command: list[str]) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(run_cli, "_execute", fake_execute)
    preset = {"ml": {"features": ["f1"], "onnx_source": "model.joblib"}}
    (tmp_path / "model.joblib").write_bytes(b"binary")
    result = run_cli.export_onnx(tmp_path, preset, ws)
    assert result.stdout == "ok"
    assert result.outputs["onnx"].suffix == ".onnx"
