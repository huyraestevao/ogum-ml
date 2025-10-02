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

from . import cache, profiling, telemetry


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


@cache.cache_result(
    "prep",
    inputs=lambda input_csv, *_, **__: [input_csv],
    params=lambda input_csv, preset, **_: preset.get("preprocess", {}),
)
@profiling.profile_step("prep")
def _run_prep_cached(
    input_csv: Path, preset: Mapping[str, Any], *, workspace: Workspace
) -> str:
    return orchestrator.run_prep(input_csv, preset, workspace)


@cache.cache_result(
    "features",
    inputs=lambda csv, *_, **__: [csv],
    params=lambda csv, preset, use_prep=True, **_: {
        "features": preset.get("features", {}),
        "use_prep": use_prep,
    },
)
@profiling.profile_step("features")
def _run_features_cached(
    csv: Path,
    preset: Mapping[str, Any],
    *,
    workspace: Workspace,
    use_prep: bool = True,
) -> str:
    return orchestrator.run_features(csv, preset, workspace)


@cache.cache_result(
    "theta_msc",
    inputs=lambda prep_csv, *_, **__: [prep_csv],
    params=lambda prep_csv, preset, **_: preset.get("msc", {}),
)
@profiling.profile_step("theta_msc")
def _run_theta_cached(
    prep_csv: Path, preset: Mapping[str, Any], *, workspace: Workspace
) -> Dict[str, Any]:
    return orchestrator.run_theta_msc(prep_csv, preset, workspace)


@profiling.profile_step("segmentation")
def _run_segmentation(
    prep_csv: Path, preset: Mapping[str, Any], *, workspace: Workspace
) -> str:
    return orchestrator.run_segmentation(prep_csv, preset, workspace)


@profiling.profile_step("mechanism")
def _run_mechanism(
    theta_csv: Path, preset: Mapping[str, Any], *, workspace: Workspace
) -> str:
    return orchestrator.run_mechanism(theta_csv, preset, workspace)


@profiling.profile_step("ml_train_cls")
def _run_ml_cls(
    features_csv: Path, preset: Mapping[str, Any], *, workspace: Workspace
) -> Dict[str, Any]:
    return orchestrator.run_ml_train_cls(features_csv, preset, workspace)


@profiling.profile_step("ml_train_reg")
def _run_ml_reg(
    features_csv: Path, preset: Mapping[str, Any], *, workspace: Workspace
) -> Dict[str, Any]:
    return orchestrator.run_ml_train_reg(features_csv, preset, workspace)


@profiling.profile_step("ml_predict")
def _run_ml_predict(
    features_csv: Path,
    model_path: Path,
    preset: Mapping[str, Any],
    *,
    workspace: Workspace,
) -> str:
    return orchestrator.run_ml_predict(features_csv, model_path, preset, workspace)


@profiling.profile_step("export")
def _run_export(
    artifacts_dir: Path, preset: Mapping[str, Any], *, workspace: Workspace
) -> str:
    return orchestrator.build_report(artifacts_dir, preset, workspace)


def _wrap_orchestrator(
    func: Callable[..., Any],
    workspace: Workspace,
    event: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    telemetry_props = kwargs.pop("telemetry_props", None)

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.2))
    def _call() -> Any:
        profiling.set_last_profile(None)
        cache.reset_last_cache()
        return func(*args, workspace=workspace, **kwargs)

    try:
        result = _call()
    except Exception as exc:
        telemetry.log_event(
            f"{event}.error", {"message": str(exc)}, workspace=workspace
        )
        raise

    props: Dict[str, Any] = dict(telemetry_props or {})
    profile = profiling.get_last_profile() or {}
    if profile:
        props.setdefault("duration_ms", profile.get("duration_ms"))
        if profile.get("memory_mb") is not None:
            props.setdefault("memory_mb", profile.get("memory_mb"))
    cache_hit = cache.last_cache_hit()
    if cache_hit is not None:
        props.setdefault("cache_hit", cache_hit)
    telemetry.log_event(event, props, workspace=workspace)
    return result


def run_prep(
    input_csv: Path, preset: Mapping[str, Any], workspace: Workspace
) -> RunResult:
    output = _wrap_orchestrator(
        _run_prep_cached,
        workspace,
        "prep",
        input_csv,
        preset,
        telemetry_props={"input": str(input_csv)},
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
        _run_features_cached,
        workspace,
        "features",
        csv,
        preset,
        use_prep=use_prep,
        telemetry_props={"use_prep": use_prep},
    )
    return RunResult(
        stdout=_log_tail(workspace),
        stderr="",
        outputs={"features_csv": Path(output)},
    )


def run_theta_msc(
    prep_csv: Path, preset: Mapping[str, Any], workspace: Workspace
) -> RunResult:
    msc_cfg = preset.get("msc", {})
    payload: Dict[str, Any] = _wrap_orchestrator(
        _run_theta_cached,
        workspace,
        "theta_msc",
        prep_csv,
        preset,
        telemetry_props={
            "ea": msc_cfg.get("ea_kj"),
            "metric": msc_cfg.get("metric"),
        },
    )
    outputs = {
        "theta_table": Path(payload["theta_table"]),
        "theta_zip": Path(payload["theta_zip"]),
        "msc_curve": Path(payload["msc_curve"]),
        "msc_plot": Path(payload["msc_plot"]),
    }
    if payload.get("best_ea") is not None:
        telemetry.log_event(
            "theta_msc.best_ea",
            {"best_ea": payload["best_ea"]},
            workspace=workspace,
        )
    return RunResult(stdout=_log_tail(workspace), stderr="", outputs=outputs)


def run_segmentation(
    prep_csv: Path, preset: Mapping[str, Any], workspace: Workspace
) -> RunResult:
    path = _wrap_orchestrator(
        _run_segmentation,
        workspace,
        "segmentation",
        prep_csv,
        preset,
        telemetry_props={"bounds": preset.get("segmentation", {}).get("bounds")},
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
        _run_mechanism,
        workspace,
        "mechanism",
        theta_csv,
        preset,
    )
    return RunResult(
        stdout=_log_tail(workspace),
        stderr="",
        outputs={"mechanism": Path(path)},
    )


def run_ml_train_cls(
    features_csv: Path, preset: Mapping[str, Any], workspace: Workspace
) -> RunResult:
    ml_cfg = preset.get("ml", {})
    payload: Dict[str, Any] = _wrap_orchestrator(
        _run_ml_cls,
        workspace,
        "ml_train_cls",
        features_csv,
        preset,
        telemetry_props={"targets": ml_cfg.get("targets"), "task": "cls"},
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
    ml_cfg = preset.get("ml", {})
    payload: Dict[str, Any] = _wrap_orchestrator(
        _run_ml_reg,
        workspace,
        "ml_train_reg",
        features_csv,
        preset,
        telemetry_props={"targets": ml_cfg.get("targets"), "task": "reg"},
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
        _run_ml_predict,
        workspace,
        "ml_predict",
        features_csv,
        model_path,
        preset,
        telemetry_props={"model": str(model_path)},
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
        _run_export,
        workspace,
        "export",
        artifacts_dir,
        preset,
        telemetry_props={"artifacts_dir": str(artifacts_dir)},
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
        telemetry.log_event(
            "export_onnx.error", {"error": str(exc)}, workspace=workspace
        )
        raise
    telemetry.log_event("export_onnx", {"model": str(model_path)}, workspace=workspace)
    return RunResult(
        stdout=completed.stdout,
        stderr=completed.stderr,
        outputs={"onnx": model_path},
    )
