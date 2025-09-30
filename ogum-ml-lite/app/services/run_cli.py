"""Service layer bridging the UI with the Ogum Lite CLI."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

from ogum_lite.ui import orchestrator
from ogum_lite.ui.workspace import Workspace
from tenacity import RetryError, retry, stop_after_attempt, wait_fixed

from . import telemetry


@dataclass(slots=True)
class RunResult:
    """Container for CLI execution outputs."""

    stdout: str
    stderr: str
    outputs: dict[str, Path]


@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.4))
def _execute(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def _log_tail(workspace: Workspace, lines: int = 12) -> str:
    log_path = workspace.resolve(workspace.log_name)
    if not log_path.exists():
        return ""
    return "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-lines:])


def _wrap_orchestrator(
    func: Callable[..., Any],
    workspace: Workspace,
    event: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.2))
    def _call() -> Any:
        return func(*args, **kwargs)

    try:
        result = _call()
    except Exception as exc:
        telemetry.log_event(workspace, f"{event}.error", {"message": str(exc)})
        raise
    telemetry.log_event(workspace, event, {})
    return result


def run_prep(
    input_csv: Path, preset: Mapping[str, Any], workspace: Workspace
) -> RunResult:
    output = _wrap_orchestrator(
        orchestrator.run_prep, workspace, "prep", input_csv, preset, workspace
    )
    return RunResult(
        stdout=_log_tail(workspace),
        stderr="",
        outputs={"prep_csv": Path(output)},
    )


def run_features(
    prep_csv: Path,
    preset: Mapping[str, Any],
    workspace: Workspace,
    *,
    use_prep: bool = True,
) -> RunResult:
    csv = prep_csv
    if not use_prep:
        csv = Path(preset.get("features", {}).get("input", prep_csv))
    output = _wrap_orchestrator(
        orchestrator.run_features, workspace, "features", csv, preset, workspace
    )
    return RunResult(
        stdout=_log_tail(workspace),
        stderr="",
        outputs={"features_csv": Path(output)},
    )


def run_theta_msc(
    prep_csv: Path, preset: Mapping[str, Any], workspace: Workspace
) -> RunResult:
    payload: Dict[str, Any] = _wrap_orchestrator(
        orchestrator.run_theta_msc, workspace, "theta_msc", prep_csv, preset, workspace
    )
    outputs = {
        "theta_table": Path(payload["theta_table"]),
        "theta_zip": Path(payload["theta_zip"]),
        "msc_curve": Path(payload["msc_curve"]),
        "msc_plot": Path(payload["msc_plot"]),
    }
    if payload.get("best_ea") is not None:
        telemetry.log_event(
            workspace, "theta_msc.best_ea", {"best_ea": payload["best_ea"]}
        )
    return RunResult(stdout=_log_tail(workspace), stderr="", outputs=outputs)


def run_segmentation(
    prep_csv: Path, preset: Mapping[str, Any], workspace: Workspace
) -> RunResult:
    path = _wrap_orchestrator(
        orchestrator.run_segmentation,
        workspace,
        "segmentation",
        prep_csv,
        preset,
        workspace,
    )
    return RunResult(
        stdout=_log_tail(workspace),
        stderr="",
        outputs={"segments": Path(path)},
    )


def run_mechanism(
    theta_csv: Path, preset: Mapping[str, Any], workspace: Workspace
) -> RunResult:
    path = _wrap_orchestrator(
        orchestrator.run_mechanism, workspace, "mechanism", theta_csv, preset, workspace
    )
    return RunResult(
        stdout=_log_tail(workspace),
        stderr="",
        outputs={"mechanism": Path(path)},
    )


def run_ml_train_cls(
    features_csv: Path, preset: Mapping[str, Any], workspace: Workspace
) -> RunResult:
    payload: Dict[str, Any] = _wrap_orchestrator(
        orchestrator.run_ml_train_cls,
        workspace,
        "ml_train_cls",
        features_csv,
        preset,
        workspace,
    )
    outdir = Path(payload["outdir"])
    outputs = {"outdir": outdir, "model_card": outdir / "model_card.json"}
    return RunResult(
        stdout=payload.get("stdout", _log_tail(workspace)),
        stderr="",
        outputs=outputs,
    )


def run_ml_train_reg(
    features_csv: Path, preset: Mapping[str, Any], workspace: Workspace
) -> RunResult:
    payload: Dict[str, Any] = _wrap_orchestrator(
        orchestrator.run_ml_train_reg,
        workspace,
        "ml_train_reg",
        features_csv,
        preset,
        workspace,
    )
    outdir = Path(payload["outdir"])
    outputs = {"outdir": outdir, "model_card": outdir / "model_card.json"}
    return RunResult(
        stdout=payload.get("stdout", _log_tail(workspace)),
        stderr="",
        outputs=outputs,
    )


def run_ml_predict(
    features_csv: Path,
    model_path: Path,
    preset: Mapping[str, Any],
    workspace: Workspace,
) -> RunResult:
    prediction = _wrap_orchestrator(
        orchestrator.run_ml_predict,
        workspace,
        "ml_predict",
        features_csv,
        model_path,
        preset,
        workspace,
    )
    return RunResult(
        stdout=_log_tail(workspace),
        stderr="",
        outputs={"predictions": Path(prediction)},
    )


def export_report(
    artifacts_dir: Path, preset: Mapping[str, Any], workspace: Workspace
) -> RunResult:
    zip_path = _wrap_orchestrator(
        orchestrator.build_report, workspace, "export", artifacts_dir, preset, workspace
    )
    export_cfg = preset.get("export", {})
    report_path = workspace.resolve(
        export_cfg.get("report", "reports/ogum_report.xlsx")
    )
    outputs = {"zip": Path(zip_path), "report": report_path}
    return RunResult(stdout=_log_tail(workspace), stderr="", outputs=outputs)


def export_onnx(
    model_dir: Path, preset: Mapping[str, Any], workspace: Workspace
) -> RunResult:
    ml_cfg = preset.get("ml", {})
    feature_list = [str(item) for item in ml_cfg.get("features", [])]
    model_path = model_dir / ml_cfg.get("onnx_model", "model.onnx")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        orchestrator.CLI_MODULE,
        "export",
        "onnx",
        "--model",
        str(model_dir / ml_cfg.get("onnx_source", "model.joblib")),
        "--out",
        str(model_path),
    ]
    if feature_list:
        command.extend(["--features", *feature_list])
    try:
        completed = _execute(command)
    except RetryError as exc:  # pragma: no cover - defensive surface
        telemetry.log_event(workspace, "export_onnx.error", {"error": str(exc)})
        raise
    telemetry.log_event(workspace, "export_onnx", {"model": str(model_path)})
    return RunResult(
        stdout=completed.stdout,
        stderr=completed.stderr,
        outputs={"onnx": model_path},
    )
