"""Thin wrappers around the Ogum Lite CLI used by the UIs."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..validators import validate_feature_df, validate_long_df
from .workspace import Workspace

CLI_MODULE = "ogum_lite.cli"


def _format_list(values: Iterable[Any]) -> str:
    return ",".join(str(item) for item in values)


def _run_cli(
    args: list[str], ws: Workspace, event: str, extra: dict | None = None
) -> subprocess.CompletedProcess:
    command = [sys.executable, "-m", CLI_MODULE, *args]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = {
        "command": command,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }
    if extra:
        payload.update(extra)
    ws.log_event(event, payload)
    return completed


def _mapping_from_preset(cfg: dict[str, Any]) -> dict[str, Any]:
    columns = cfg.get("columns", {})
    return {
        "sample_id": columns.get("group_col", "sample_id"),
        "time_col": columns.get("t_col", "time_s"),
        "temp_col": columns.get("temp_col", "temp_C"),
        "y_col": columns.get("y_col", "rho_rel"),
        "composition": columns.get("composition_col", "composition"),
        "technique": columns.get("technique_col", "technique"),
        "time_unit": columns.get("time_unit", "s"),
        "temp_unit": columns.get("temp_unit", "C"),
    }


def run_prep(input_csv: Path, cfg: dict[str, Any], ws: Workspace) -> Path:
    """Execute ``preprocess derive`` using CLI based on the preset."""

    prep_cfg = cfg.get("prep", {})
    mapping = prep_cfg.get("mapping") or _mapping_from_preset(cfg)
    mapping_path = ws.resolve(prep_cfg.get("mapping_path", "cache/column_mapping.json"))
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))

    output_path = ws.resolve(prep_cfg.get("output", "prep/ensaios_prep.csv"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "preprocess",
        "derive",
        "--input",
        str(input_csv),
        "--map",
        str(mapping_path),
        "--out",
        str(output_path),
    ]
    for key in ("smooth", "window", "poly", "moving_k"):
        if key in prep_cfg and prep_cfg[key] is not None:
            args.extend([f"--{key.replace('_', '-')}", str(prep_cfg[key])])

    _run_cli(args, ws, "preprocess.derive", {"output": str(output_path)})
    return output_path


def run_features(input_csv: Path, cfg: dict[str, Any], ws: Workspace) -> Path:
    """Generate feature table via CLI ``features`` command."""

    features_cfg = cfg.get("features", {})
    msc_cfg = cfg.get("msc", {})
    columns = cfg.get("columns", {})

    output_path = ws.resolve(features_cfg.get("output", "features/features.csv"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ea_values = msc_cfg.get("ea_kj") or []
    if not ea_values:
        raise ValueError("Preset must provide msc.ea_kj for feature generation")
    args = [
        "features",
        "--input",
        str(input_csv),
        "--output",
        str(output_path),
    ]
    args.extend(["--ea", _format_list(ea_values)])
    args.extend(
        [
            "--group-col",
            columns.get("group_col", "sample_id"),
            "--time-column",
            columns.get("t_col", "time_s"),
            "--temperature-column",
            columns.get("temp_col", "temp_C"),
            "--y-column",
            columns.get("y_col", "rho_rel"),
        ]
    )

    _run_cli(args, ws, "features.legacy", {"output": str(output_path)})
    return output_path


def run_theta_msc(
    input_csv: Path, cfg: dict[str, Any], ws: Workspace
) -> dict[str, Any]:
    """Compute θ(Ea) table and MSC artifacts."""

    msc_cfg = cfg.get("msc", {})
    columns = cfg.get("columns", {})
    ea_values = msc_cfg.get("ea_kj", [])
    if not ea_values:
        raise ValueError("Preset must provide msc.ea_kj for θ/MSC computation")
    outdir = ws.resolve(msc_cfg.get("outdir", "theta_msc"))
    outdir.mkdir(parents=True, exist_ok=True)

    theta_args = [
        "theta",
        "--input",
        str(input_csv),
        "--ea",
        _format_list(ea_values),
        "--time-column",
        columns.get("t_col", "time_s"),
        "--temperature-column",
        columns.get("temp_col", "temp_C"),
        "--outdir",
        str(outdir),
    ]
    _run_cli(theta_args, ws, "theta.compute", {"outdir": str(outdir)})

    theta_table = outdir / "theta_table.csv"
    theta_zip = outdir / "theta_curves.zip"
    if theta_table.exists():
        df = pd.read_csv(theta_table)
        with zipfile.ZipFile(theta_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            if "sample_id" in df.columns:
                for sample_id, group in df.groupby("sample_id"):
                    buffer = group.to_csv(index=False).encode("utf-8")
                    zf.writestr(f"theta_{sample_id}.csv", buffer)
            else:  # pragma: no cover - fallback for unexpected schema
                zf.writestr("theta_table.csv", df.to_csv(index=False).encode("utf-8"))

    msc_curve = outdir / "msc_curve.csv"
    msc_png = outdir / "msc_plot.png"
    msc_args = [
        "msc",
        "--input",
        str(input_csv),
        "--ea",
        _format_list(ea_values),
        "--metric",
        msc_cfg.get("metric", "segmented"),
        "--group-col",
        columns.get("group_col", "sample_id"),
        "--time-column",
        columns.get("t_col", "time_s"),
        "--temperature-column",
        columns.get("temp_col", "temp_C"),
        "--y-column",
        columns.get("y_col", "rho_rel"),
        "--csv",
        str(msc_curve),
        "--png",
        str(msc_png),
    ]
    result = _run_cli(msc_args, ws, "msc.score", {"outdir": str(outdir)})

    match = re.search(r"best_Ea_kJ_mol=([0-9.]+)", result.stdout)
    best_ea = float(match.group(1)) if match else None

    return {
        "best_ea": best_ea,
        "theta_table": theta_table,
        "theta_zip": theta_zip,
        "msc_curve": msc_curve,
        "msc_plot": msc_png,
    }


def run_segmentation(input_csv: Path, cfg: dict[str, Any], ws: Workspace) -> Path:
    """Execute CLI segmentation with preset options."""

    seg_cfg = cfg.get("segmentation", {})
    columns = cfg.get("columns", {})
    output_path = ws.resolve(seg_cfg.get("output", "segments/segments.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "segmentation",
        "--input",
        str(input_csv),
        "--out",
        str(output_path),
        "--mode",
        seg_cfg.get("mode", "fixed"),
        "--group-col",
        columns.get("group_col", "sample_id"),
        "--time-column",
        columns.get("t_col", "time_s"),
        "--y-column",
        columns.get("y_col", "rho_rel"),
    ]
    if seg_cfg.get("mode", "fixed") == "fixed" and seg_cfg.get("bounds"):
        args.extend(["--thresholds", ",".join(seg_cfg.get("bounds", []))])
    else:
        if "segments" in seg_cfg:
            args.extend(["--segments", str(seg_cfg["segments"])])
        if "min_size" in seg_cfg:
            args.extend(["--min-size", str(seg_cfg["min_size"])])

    _run_cli(args, ws, "segmentation.run", {"output": str(output_path)})
    return output_path


def run_mechanism(theta_csv: Path, cfg: dict[str, Any], ws: Workspace) -> Path:
    """Detect mechanism change based on θ(Ea) trajectories."""

    mech_cfg = cfg.get("mechanism", {})
    columns = cfg.get("columns", {})
    output_path = ws.resolve(mech_cfg.get("output", "mechanism/mechanism.csv"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "mechanism",
        "--theta",
        str(theta_csv),
        "--out",
        str(output_path),
        "--group-col",
        columns.get("group_col", "sample_id"),
    ]
    if "theta_column" in mech_cfg:
        args.extend(["--theta-column", mech_cfg["theta_column"]])
    if "y_column" in mech_cfg:
        args.extend(["--y-column", mech_cfg["y_column"]])
    if "segments" in mech_cfg:
        args.extend(["--segments", str(mech_cfg["segments"])])
    if "min_size" in mech_cfg:
        args.extend(["--min-size", str(mech_cfg["min_size"])])
    if "criterion" in mech_cfg:
        args.extend(["--criterion", mech_cfg["criterion"]])
    if "threshold" in mech_cfg:
        args.extend(["--threshold", str(mech_cfg["threshold"])])
    if "slope_delta" in mech_cfg:
        args.extend(["--slope-delta", str(mech_cfg["slope_delta"])])

    _run_cli(args, ws, "mechanism.run", {"output": str(output_path)})
    return output_path


def run_ml_train_cls(
    features_csv: Path, cfg: dict[str, Any], ws: Workspace
) -> dict[str, Any]:
    """Train classification model via CLI ``ml train-cls``."""

    ml_cfg = cfg.get("ml", {})
    columns = cfg.get("columns", {})
    outdir = ws.resolve(ml_cfg.get("cls_outdir", "ml/classifier"))
    outdir.mkdir(parents=True, exist_ok=True)

    features = list(ml_cfg.get("features", []))
    if not features:
        raise ValueError("Preset must provide ml.features for classification training")

    args = [
        "ml",
        "train-cls",
        "--table",
        str(features_csv),
        "--target",
        ml_cfg.get("cls_target", "technique"),
        "--group-col",
        columns.get("group_col", "sample_id"),
        "--outdir",
        str(outdir),
    ]
    args.extend(["--features", *features])

    result = _run_cli(args, ws, "ml.train_cls", {"outdir": str(outdir)})
    model_card = outdir / "model_card.json"
    metrics = json.loads(model_card.read_text()) if model_card.exists() else {}
    return {
        "outdir": outdir,
        "stdout": result.stdout,
        "model_card": metrics,
    }


def run_ml_train_reg(
    features_csv: Path, cfg: dict[str, Any], ws: Workspace
) -> dict[str, Any]:
    """Train regression model via CLI ``ml train-reg``."""

    ml_cfg = cfg.get("ml", {})
    columns = cfg.get("columns", {})
    outdir = ws.resolve(ml_cfg.get("reg_outdir", "ml/regressor"))
    outdir.mkdir(parents=True, exist_ok=True)

    features = list(ml_cfg.get("features", []))
    if not features:
        raise ValueError("Preset must provide ml.features for regression training")

    args = [
        "ml",
        "train-reg",
        "--table",
        str(features_csv),
        "--target",
        ml_cfg.get("reg_target", "T90_C"),
        "--group-col",
        columns.get("group_col", "sample_id"),
        "--outdir",
        str(outdir),
    ]
    args.extend(["--features", *features])

    result = _run_cli(args, ws, "ml.train_reg", {"outdir": str(outdir)})
    model_card = outdir / "model_card.json"
    metrics = json.loads(model_card.read_text()) if model_card.exists() else {}
    return {
        "outdir": outdir,
        "stdout": result.stdout,
        "model_card": metrics,
    }


def run_ml_predict(
    features_csv: Path, model_path: Path, cfg: dict[str, Any], ws: Workspace
) -> Path:
    """Generate predictions from a stored model artifact."""

    predict_cfg = cfg.get("ml", {})
    output_path = ws.resolve(predict_cfg.get("predict_output", "ml/predictions.csv"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "ml",
        "predict",
        "--table",
        str(features_csv),
        "--model",
        str(model_path),
        "--out",
        str(output_path),
    ]
    _run_cli(args, ws, "ml.predict", {"output": str(output_path)})
    return output_path


def run_validation(input_csv: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate long-format datasets using the shared validators."""

    columns = cfg.get("columns", {})
    df = pd.read_csv(input_csv)
    return validate_long_df(df, y_col=columns.get("y_col", "rho_rel"))


def run_feature_validation(features_csv: Path) -> dict[str, Any]:
    """Validate feature tables using the shared validators."""

    df = pd.read_csv(features_csv)
    return validate_feature_df(df)


def build_report(artifacts_dir: Path, cfg: dict[str, Any], ws: Workspace) -> Path:
    """Generate the consolidated XLSX report and bundle workspace outputs."""

    export_cfg = cfg.get("export", {})
    report_path = ws.resolve(export_cfg.get("report", "reports/ogum_report.xlsx"))
    report_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "export",
        "xlsx",
        "--out",
        str(report_path),
    ]
    optional_flags = {
        "msc": export_cfg.get("msc_csv"),
        "features": export_cfg.get("features_csv"),
        "metrics": export_cfg.get("metrics_json"),
        "dataset": export_cfg.get("dataset_name"),
        "notes": export_cfg.get("notes"),
        "img-msc": export_cfg.get("img_msc"),
        "img-cls": export_cfg.get("img_cls"),
        "img-reg": export_cfg.get("img_reg"),
    }
    for flag, value in optional_flags.items():
        if value:
            args.extend([f"--{flag}", str(value)])

    _run_cli(args, ws, "export.xlsx", {"report": str(report_path)})

    zip_target = ws.resolve(export_cfg.get("zip", "exports/session_artifacts.zip"))
    ws.zip_outputs(artifacts_dir, zip_target)
    return zip_target
