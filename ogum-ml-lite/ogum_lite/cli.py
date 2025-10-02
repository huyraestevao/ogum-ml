"""Command line interface for the Ogum Lite toolkit."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Literal, Sequence

import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

from .arrhenius import arrhenius_lnT_dy_dt_vs_invT, fit_arrhenius_global
from .exporters import export_onnx, export_xlsx
from .features import arrhenius_feature_table, build_feature_store, build_feature_table
from .io_mapping import (
    TECHNIQUE_CHOICES,
    TECHNIQUE_PROFILES,
    ColumnMap,
    apply_mapping,
    infer_mapping,
    read_table,
)
from .maps import prepare_segment_heatmap, render_segment_heatmap
from .mechanism import detect_mechanism_change
from .ml_experiments import compare_models, list_available_models, run_benchmark_matrix
from .ml_hooks import (
    compute_permutation_importance,
    filter_features_by_importance,
    kmeans_explore,
    predict_from_artifact,
    random_search_classifier,
    random_search_regressor,
    train_classifier,
    train_regressor,
)
from .preprocess import convert_response, derive_all
from .reports import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_regression_scatter,
    render_html_report,
)
from .segmentation import aggregate_max_rate_bounds, segment_dataframe
from .stages import DEFAULT_STAGES
from .theta_msc import OgumLite, score_activation_energies
from .validators import validate_feature_df, validate_long_df


def parse_ea_list(raw: str) -> List[float]:
    """Parse a comma separated list of activation energies."""

    try:
        return [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(
            "Activation energies must be numeric."
        ) from exc


def parse_ea_range(raw: str) -> List[float]:
    """Parse activation energy ranges in the form ``start:end:step``."""

    try:
        start_str, end_str, step_str = raw.split(":", 2)
        start = float(start_str)
        end = float(end_str)
        step = float(step_str)
    except ValueError as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(
            "Ea range must be provided as start:end:step"
        ) from exc
    if step <= 0:
        raise argparse.ArgumentTypeError("Ea range step must be positive")
    if end <= start:
        raise argparse.ArgumentTypeError("Ea range end must be greater than start")

    values: List[float] = []
    current = start
    # Protect against floating point drift by iterating up to an inclusive bound
    while current <= end + 1e-9:
        values.append(round(current, 6))
        current += step
    return values


def parse_stage_ranges(raw: str | None) -> list[tuple[float, float]]:
    """Parse densification ranges like ``0.55-0.70,0.70-0.90``."""

    if raw is None or not raw.strip():
        return list(DEFAULT_STAGES)

    ranges: list[tuple[float, float]] = []
    for chunk in raw.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        if "-" not in piece:
            raise argparse.ArgumentTypeError(
                f"Invalid stage definition '{piece}'. Expected format 'lower-upper'."
            )
        lower_str, upper_str = piece.split("-", 1)
        try:
            lower = float(lower_str)
            upper = float(upper_str)
        except ValueError as exc:  # pragma: no cover - defensive
            raise argparse.ArgumentTypeError(
                f"Stage bounds must be numeric: '{piece}'"
            ) from exc
        if upper <= lower:
            raise argparse.ArgumentTypeError(
                f"Stage upper bound must exceed lower bound: '{piece}'"
            )
        ranges.append((lower, upper))

    if not ranges:
        return list(DEFAULT_STAGES)
    return ranges


def parse_thresholds(raw: str | None) -> Sequence[float]:
    """Parse comma separated densification thresholds."""

    if raw is None or not raw.strip():
        return (0.55, 0.70, 0.90)

    values: list[float] = []
    for chunk in raw.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        try:
            value = float(piece)
        except ValueError as exc:  # pragma: no cover - defensive
            raise argparse.ArgumentTypeError(
                f"Invalid threshold '{piece}'. Must be numeric."
            ) from exc
        if not 0.0 < value < 1.0:
            raise argparse.ArgumentTypeError(
                f"Thresholds must be within (0, 1). Received {value}."
            )
        values.append(value)

    if not values:
        return (0.55, 0.70, 0.90)
    return tuple(values)


def parse_targets(raw: str) -> list[str]:
    """Parse target column names from CLI input."""

    targets = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if not targets:
        raise argparse.ArgumentTypeError("Provide at least one target column")
    return targets


def parse_feature_sets(definitions: Sequence[str]) -> dict[str, list[str]]:
    """Parse feature set definitions like ``basic="f1,f2"``."""

    result: dict[str, list[str]] = {}
    for item in definitions:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid feature set '{item}'. Use alias=col1,col2"
            )
        alias, cols = item.split("=", 1)
        alias = alias.strip()
        if not alias:
            raise argparse.ArgumentTypeError("Feature set alias cannot be empty")
        columns = [col.strip() for col in cols.split(",") if col.strip()]
        if not columns:
            raise argparse.ArgumentTypeError(
                f"Feature set '{alias}' must include at least one column"
            )
        result[alias] = columns
    if not result:
        raise argparse.ArgumentTypeError("Provide at least one feature set definition")
    return result


def parse_model_aliases(raw: str | None) -> list[str] | None:
    """Parse a comma separated list of model aliases."""

    if raw is None:
        return None
    models = [piece.strip() for piece in raw.split(",") if piece.strip()]
    return models or None


def resolve_stage_segments(
    dataframe: pd.DataFrame,
    *,
    segmentation_mode: Literal["fixed", "max-rate"] = "fixed",
    stage_text: str | None = None,
    group_col: str = "sample_id",
    t_col: str = "time_s",
    y_col: str = "rho_rel",
    n_segments: int = 3,
    min_size: int = 5,
) -> list[tuple[float, float]]:
    """Resolve densification segments based on CLI parameters."""

    if segmentation_mode == "fixed":
        return parse_stage_ranges(stage_text)
    if segmentation_mode == "max-rate":
        bounds = aggregate_max_rate_bounds(
            dataframe,
            group_col=group_col,
            t_col=t_col,
            y_col=y_col,
            n_segments=n_segments,
            min_size=min_size,
        )
        if not bounds:
            raise ValueError(
                "Unable to derive max-rate bounds. Provide cleaner data or "
                "adjust min_size."
            )
        return bounds
    raise ValueError("segmentation_mode must be 'fixed' or 'max-rate'")


def _mapping_from_json(path: Path) -> ColumnMap:
    data = json.loads(Path(path).read_text())
    return ColumnMap(**data)


def _edit_mapping(mapping: ColumnMap, df: pd.DataFrame) -> ColumnMap:
    current = asdict(mapping)
    columns = [str(col) for col in df.columns]

    print("\nDetected mapping:")
    for key, value in current.items():
        print(f"  {key}: {value}")

    print("\nEnter adjustments as field=value. Press ENTER to keep current mapping.")
    print("Available columns:")
    print("  " + ", ".join(columns))
    print("Technique options: " + ", ".join(TECHNIQUE_CHOICES))
    print(
        "Text fields: composition_default, technique_default, tech_comment, "
        "user, timestamp"
    )

    while True:
        try:
            line = input("mapping> ").strip()
        except EOFError:  # pragma: no cover - interactive safeguard
            break
        if not line:
            break
        if "=" not in line:
            print("Please use the format field=value")
            continue
        field, value = [item.strip() for item in line.split("=", 1)]
        if field not in current:
            print(f"Unknown field '{field}'. Valid keys: {', '.join(current)}")
            continue
        if field == "time_unit":
            if value not in {"s", "min"}:
                print("time_unit must be 's' or 'min'")
                continue
        elif field == "temp_unit":
            if value not in {"C", "K"}:
                print("temp_unit must be 'C' or 'K'")
                continue
        elif field in {"composition_default", "tech_comment", "user", "timestamp"}:
            current[field] = value
            continue
        elif field == "technique_default":
            if value not in TECHNIQUE_CHOICES:
                print("technique_default must match one of the predefined choices")
                continue
            current[field] = value
            continue
        elif field == "extra_metadata":
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                print("extra_metadata must be a valid JSON object")
                continue
            if not isinstance(parsed, dict):
                print("extra_metadata JSON must represent an object")
                continue
            current[field] = parsed
            continue
        elif field == "technique":
            if value not in TECHNIQUE_CHOICES and value not in columns:
                print(
                    "Technique must be a column name or one of the predefined choices"
                )
                continue
        elif value not in columns:
            print(f"Value '{value}' not found among dataframe columns")
            continue
        current[field] = value

    return ColumnMap(**current)


def cmd_segmentation(args: argparse.Namespace) -> None:
    if args.input is None or args.out is None:
        raise SystemExit("--input and --out are required")

    df = pd.read_csv(args.input)
    thresholds = parse_thresholds(args.thresholds)

    segments = segment_dataframe(
        df,
        group_col=args.group_col,
        t_col=args.time_column,
        y_col=args.y_column,
        method=args.mode,
        thresholds=thresholds,
        n_segments=args.segments,
        min_size=args.min_size,
    )

    grouped: dict[str | int | float | None, list[dict[str, float | int | str]]] = {}
    for segment in segments:
        key = segment.sample_id
        if isinstance(key, np.generic):
            key = key.item()
        grouped.setdefault(key, []).append(segment.to_dict())

    output_records = []
    for sample_key, segment_list in grouped.items():
        output_records.append(
            {
                args.group_col: sample_key,
                "mode": args.mode,
                "segments": segment_list,
            }
        )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_records, indent=2, default=float))

    print(f"Saved {len(segments)} segments to {output_path}")


def cmd_features(args: argparse.Namespace) -> None:
    required = ["input", "ea"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        missing_opts = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise SystemExit(f"Missing required options: {missing_opts}")

    dataframe = pd.read_csv(args.input)
    features_df = build_feature_table(
        dataframe,
        args.ea,
        group_col=args.group_col,
        t_col=args.time_column,
        temp_col=args.temperature_column,
        y_col=args.y_column,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    print(f"Feature table saved to {output_path}")
    if args.print:
        print(features_df.to_string(index=False))


def cmd_mechanism(args: argparse.Namespace) -> None:
    if args.theta is None or args.out is None:
        raise SystemExit("--theta and --out are required")

    df = pd.read_csv(args.theta)
    results = detect_mechanism_change(
        df,
        group_col=args.group_col,
        theta_col=args.theta_column,
        y_col=args.y_column,
        max_segments=args.segments,
        min_size=args.min_size,
        criterion=args.criterion,
        threshold=args.threshold,
        slope_delta=args.slope_delta,
    )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Mechanism summary saved to {output_path}")


def cmd_features_build(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input)
    stages = parse_stage_ranges(args.stages)
    theta_ea = parse_ea_list(args.theta_ea) if args.theta_ea else None
    if theta_ea is not None and not theta_ea:
        theta_ea = None

    feature_store = build_feature_store(
        df,
        stages=stages,
        group_col=args.group_col,
        t_col=args.time_column,
        T_col=args.temperature_column,
        y_col=args.y_column,
        smooth=args.smooth,
        window=args.window,
        poly=args.poly,
        moving_k=args.moving_k,
        theta_ea_kj=theta_ea,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_store.to_csv(output_path, index=False)
    print(f"Feature store saved to {output_path}")
    if args.print:
        print(feature_store.to_string(index=False))


def cmd_ml_features(args: argparse.Namespace) -> None:
    cmd_features(args)


def cmd_io_map(args: argparse.Namespace) -> None:
    df = read_table(str(args.input))

    default_composition = getattr(args, "default_composition", None)
    default_technique = getattr(args, "default_technique", None)
    tech_comment = getattr(args, "tech_comment", None)
    user = getattr(args, "user", None)
    timestamp = getattr(args, "timestamp", None)

    if default_technique and default_technique not in TECHNIQUE_CHOICES:
        raise SystemExit(
            "--default-technique must be one of: " + ", ".join(TECHNIQUE_CHOICES)
        )

    if user and timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    mapping = infer_mapping(
        df,
        default_composition=default_composition,
        default_technique=default_technique,
        tech_comment=tech_comment,
        user=user,
        timestamp=timestamp,
    )
    if args.edit:
        mapping = _edit_mapping(mapping, df)

    updates: dict[str, object] = {}
    if tech_comment is not None:
        updates["tech_comment"] = tech_comment
    if user is not None:
        updates["user"] = user
    if timestamp is not None:
        updates["timestamp"] = timestamp
    if default_composition is not None:
        updates.setdefault("composition_default", default_composition)
    if default_technique is not None:
        updates.setdefault("technique_default", default_technique)
    if updates:
        mapping = replace(mapping, **updates)

    if mapping.technique is None and mapping.technique_default:
        profile = TECHNIQUE_PROFILES.get(mapping.technique_default)
        if profile:
            extra = dict(mapping.extra_metadata)
            for key, value in profile.items():
                if key == "name":
                    continue
                extra[f"tech_{key}"] = str(value)
            mapping = replace(mapping, extra_metadata=extra)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(mapping), indent=2))
    print(f"Column mapping saved to {args.out}")


def cmd_preprocess_derive(args: argparse.Namespace) -> None:
    df = read_table(str(args.input))
    mapping = _mapping_from_json(args.map)
    mapped = apply_mapping(df, mapping)
    response = convert_response(
        mapped,
        column="rho_rel",
        response_type=args.response_type,
        L0=args.L0,
        rho0=args.rho0,
    )
    mapped["rho_rel"] = response
    mapped["response"] = response
    mapped["y"] = response
    derived = derive_all(
        mapped,
        smooth=args.smooth,
        window=args.window,
        poly=args.poly,
        moving_k=args.moving_k,
        y_col="rho_rel",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    derived.to_csv(args.out, index=False)
    print(f"Derived table saved to {args.out}")
    if args.print:
        print(derived.head().to_string(index=False))


def cmd_arrhenius_fit(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input)
    if "dy_dt" not in df.columns:
        df = derive_all(
            df,
            t_col=args.time_column,
            T_col=args.temperature_column,
            y_col=args.y_column,
            smooth=args.smooth,
            window=args.window,
            poly=args.poly,
            moving_k=args.moving_k,
        )

    try:
        stages = resolve_stage_segments(
            df,
            segmentation_mode=args.segmentation_mode,
            stage_text=args.stages,
            group_col=args.group_col,
            t_col=args.time_column,
            y_col=args.y_column,
            n_segments=args.segments,
            min_size=args.min_size,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    arrhenius_table = arrhenius_feature_table(
        df,
        stages=stages,
        group_col=args.group_col,
        t_col=args.time_column,
        T_col=args.temperature_column,
        y_col=args.y_column,
        smooth=args.smooth,
        window=args.window,
        poly=args.poly,
        moving_k=args.moving_k,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    arrhenius_table.to_csv(args.out, index=False)
    print(arrhenius_table.to_string(index=False))

    if args.png:
        prepared = arrhenius_lnT_dy_dt_vs_invT(
            df,
            T_col=args.temperature_column,
            dy_dt_col="dy_dt",
        )
        result = fit_arrhenius_global(prepared)
        fig, ax = plt.subplots()
        ax.scatter(prepared["invT_K"], prepared["ln_T_dy_dt"], s=10, alpha=0.6)
        x_grid = np.linspace(prepared["invT_K"].min(), prepared["invT_K"].max(), 200)
        y_grid = result.slope * x_grid + result.intercept
        ax.plot(x_grid, y_grid, color="red", linewidth=2.0, label="Global fit")
        ax.set_xlabel("1/T (1/K)")
        ax.set_ylabel("ln(T·dy/dt)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        args.png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Arrhenius plot saved to {args.png}")


def _print_cv_metrics(cv_metrics: dict[str, float]) -> None:
    n_splits = cv_metrics.get("n_splits")
    n_groups = cv_metrics.get("n_groups")
    header = "Cross-validation metrics"
    if n_splits:
        header += f" (n_splits={int(n_splits)})"
    if n_groups:
        header += f" | n_groups={int(n_groups)}"
    print(header)
    for key in sorted(cv_metrics):
        if key in {"n_splits", "n_groups"}:
            continue
        print(f"  {key}: {cv_metrics[key]:.4f}")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_feature_columns_from_dir(outdir: Path) -> list[str]:
    payload = _load_json(outdir / "feature_cols.json")
    features = payload.get("features")
    if not isinstance(features, list):  # pragma: no cover - defensive
        raise SystemExit("feature_cols.json must contain a list under 'features'")
    return [str(item) for item in features]


def _print_validation_report(label: str, report: dict) -> None:
    status = "OK" if report.get("ok") else "FAILED"
    print(f"{label}: {status}")
    issues = report.get("issues", [])
    if not issues:
        return
    limit = min(len(issues), 10)
    print(f"  showing {limit} issue(s) (see JSON for full list)")
    for issue in issues[:limit]:
        print(f"  - {issue}")
    remaining = len(issues) - limit
    if remaining > 0:
        print(f"  ... {remaining} more")


def cmd_validate_long(args: argparse.Namespace) -> None:
    dataframe = pd.read_csv(args.input)
    result = validate_long_df(dataframe, y_col=args.y_col)
    payload = {
        "input": str(args.input),
        "rows": int(len(dataframe)),
        "y_col": args.y_col,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **result,
    }
    _print_validation_report("Long table validation", payload)
    _write_json(args.out, payload)
    print(f"Validation report saved to {args.out}")


def cmd_validate_features(args: argparse.Namespace) -> None:
    dataframe = pd.read_csv(args.table)
    result = validate_feature_df(dataframe)
    payload = {
        "input": str(args.table),
        "rows": int(len(dataframe)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **result,
    }
    _print_validation_report("Feature table validation", payload)
    _write_json(args.out, payload)
    print(f"Validation report saved to {args.out}")


def _load_metrics(path: Path | None) -> tuple[dict, pd.DataFrame]:
    if path is None:
        return {}, pd.DataFrame()
    payload = _load_json(path)
    if isinstance(payload, dict):
        rows = [
            {"metric": key, "value": value} for key, value in sorted(payload.items())
        ]
        frame = pd.DataFrame(rows)
    else:
        frame = pd.DataFrame(payload)
    return payload, frame


def cmd_export_xlsx(args: argparse.Namespace) -> None:
    sources: dict[str, str] = {}
    context: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
    }
    if args.dataset:
        context["dataset"] = {"name": args.dataset}
    if args.notes:
        context["notes"] = args.notes

    tables: dict[str, pd.DataFrame] = {}

    if args.msc:
        msc_df = pd.read_csv(args.msc)
        tables["MSC"] = msc_df
        sources["msc"] = str(args.msc)
        msc_summary = context.setdefault("msc_summary", {})
        if isinstance(msc_summary, dict):
            msc_summary["rows"] = int(len(msc_df))
        if "activation_energy" in msc_df.columns:
            if isinstance(msc_summary, dict):
                msc_summary["top_activation_energy"] = msc_df.iloc[0][
                    "activation_energy"
                ]

    if args.features:
        features_df = pd.read_csv(args.features)
        tables["Features"] = features_df
        sources["features"] = str(args.features)
        features_summary = context.setdefault("features_summary", {})
        if isinstance(features_summary, dict):
            features_summary["rows"] = int(len(features_df))

    metrics_payload, metrics_df = _load_metrics(args.metrics)
    if not metrics_df.empty:
        tables["Metrics"] = metrics_df
    if metrics_payload:
        context["metrics"] = metrics_payload
        sources["metrics"] = str(args.metrics)

    if args.msc is None:
        tables.setdefault("MSC", pd.DataFrame())
    if args.features is None:
        tables.setdefault("Features", pd.DataFrame())
    if args.metrics is None:
        tables.setdefault("Metrics", pd.DataFrame())

    image_payload: dict[str, bytes] = {}
    if args.img_msc:
        image_payload["msc.png"] = Path(args.img_msc).read_bytes()
    if args.img_cls:
        image_payload["confusion.png"] = Path(args.img_cls).read_bytes()
    if args.img_reg:
        image_payload["scatter.png"] = Path(args.img_reg).read_bytes()

    export_xlsx(
        args.out,
        context=context,
        tables=tables,
        images=image_payload or None,
    )
    print(f"Excel report saved to {args.out}")


def _load_feature_names(path: Path) -> list[str]:
    payload = _load_json(path)
    if isinstance(payload, dict):
        features = payload.get("features", payload.get("columns"))
    else:
        features = payload
    if not isinstance(features, list):
        raise SystemExit("feature list JSON must be an array or contain 'features'")
    return [str(name) for name in features]


def cmd_export_onnx(args: argparse.Namespace) -> None:
    model = joblib.load(args.model)
    feature_names = _load_feature_names(args.features_json)
    out_path = export_onnx(model, feature_names, args.out)
    if out_path is None:
        print("ignorado (deps ausentes ou modelo incompatível)")
    else:
        print(f"ONNX model exported to {out_path}")


def cmd_ml_train_classifier(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.table)
    result = train_classifier(
        df,
        target_col=args.target,
        group_col=args.group_col,
        feature_cols=args.features,
        outdir=args.outdir,
    )
    _print_cv_metrics(result["cv"])
    print(f"Artifacts saved to {args.outdir}")


def cmd_ml_tune_classifier(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.table)
    result = random_search_classifier(
        df,
        target_col=args.target,
        group_col=args.group_col,
        feature_cols=args.features,
        outdir=args.outdir,
        n_iter=args.n_iter,
        cv_splits=args.cv,
        random_state=args.random_state,
    )
    print("Best hyperparameters:")
    for key, value in result["best_params"].items():
        print(f"  {key}: {value}")
    cv_metrics = result["cv"]
    acc_mean = cv_metrics.get("accuracy_mean", 0.0)
    acc_std = cv_metrics.get("accuracy_std", 0.0)
    f1_mean = cv_metrics.get("f1_macro_mean", 0.0)
    f1_std = cv_metrics.get("f1_macro_std", 0.0)
    print(f"accuracy = {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"f1_macro = {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"Artifacts saved to {args.outdir}")


def cmd_ml_train_regressor(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.table)
    result = train_regressor(
        df,
        target_col=args.target,
        group_col=args.group_col,
        feature_cols=args.features,
        outdir=args.outdir,
    )
    _print_cv_metrics(result["cv"])
    print(f"Artifacts saved to {args.outdir}")


def cmd_ml_tune_regressor(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.table)
    result = random_search_regressor(
        df,
        target_col=args.target,
        group_col=args.group_col,
        feature_cols=args.features,
        outdir=args.outdir,
        n_iter=args.n_iter,
        cv_splits=args.cv,
        random_state=args.random_state,
    )
    print("Best hyperparameters:")
    for key, value in result["best_params"].items():
        print(f"  {key}: {value}")
    cv_metrics = result["cv"]
    mae_mean = cv_metrics.get("mae_mean", 0.0)
    mae_std = cv_metrics.get("mae_std", 0.0)
    rmse_mean = cv_metrics.get("rmse_mean", 0.0)
    rmse_std = cv_metrics.get("rmse_std", 0.0)
    print(f"MAE = {mae_mean:.4f} ± {mae_std:.4f}")
    print(f"RMSE = {rmse_mean:.4f} ± {rmse_std:.4f}")
    print(f"Artifacts saved to {args.outdir}")


def cmd_ml_predict(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.table)
    predictions = predict_from_artifact(args.model, df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out, index=False)
    print(predictions.to_string(index=False))
    print(f"Predictions saved to {args.out}")


def cmd_ml_cluster(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.table)
    clusters = kmeans_explore(df, args.features, k=args.k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    clusters.to_csv(args.out, index=False)
    print(clusters.to_string(index=False))
    print(f"Cluster assignments saved to {args.out}")


def cmd_ml_report(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.table)
    model = joblib.load(args.model)
    if not hasattr(model, "named_steps"):
        raise SystemExit("Loaded object is not a scikit-learn Pipeline")
    estimator = model.named_steps.get("model")
    if isinstance(estimator, RandomForestClassifier):
        task = "classification"
        scoring = "accuracy"
    elif isinstance(estimator, RandomForestRegressor):
        task = "regression"
        scoring = "neg_mean_absolute_error"
    else:  # pragma: no cover - defensive
        raise SystemExit("Unsupported estimator for reporting")

    feature_cols = _load_feature_columns_from_dir(args.outdir)
    required = [args.target, *feature_cols]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in table: {', '.join(missing)}")
    if args.group_col not in df.columns:
        raise SystemExit(f"Missing group column '{args.group_col}' in table")

    cleaned = df.dropna(subset=required)
    if cleaned.empty:
        raise SystemExit("No rows available after dropping missing values")

    X = cleaned[feature_cols]
    y_true = cleaned[args.target]
    y_pred = model.predict(X)

    if task == "classification":
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        metrics = {"accuracy": f"{accuracy:.4f}", "f1_macro": f"{f1_macro:.4f}"}
        labels = (
            pd.Index(pd.Series(y_true))
            .append(pd.Index(pd.Series(y_pred)))
            .unique()
            .tolist()
        )
        figures = {
            "confusion_matrix": plot_confusion_matrix(
                np.array(y_true),
                np.array(y_pred),
                labels=labels,
            )
        }
    else:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics = {"MAE": f"{mae:.4f}", "RMSE": f"{rmse:.4f}"}
        figures = {
            "regression_scatter": plot_regression_scatter(
                np.array(y_true), np.array(y_pred)
            )
        }

    importance_df = compute_permutation_importance(
        model,
        X,
        y_true,
        scoring=scoring,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )
    if task == "regression":
        importance_df["importance_mean"] = importance_df["importance_mean"].abs()
    figures["feature_importance"] = plot_feature_importance(importance_df)

    model_card = _load_json(args.outdir / "model_card.json")
    cv_metrics = model_card.get("cross_validation", {})
    best_params = model_card.get("best_params", {})
    dataset_info = {
        "n_samples": int(cleaned.shape[0]),
        "n_features": len(feature_cols),
    }
    if args.group_col in cleaned.columns:
        dataset_info["n_groups"] = int(cleaned[args.group_col].nunique())

    context = {
        "title": f"Ogum ML Report — {args.target}",
        "task": task,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "cv_metrics": cv_metrics,
        "best_params": best_params,
        "dataset": dataset_info,
        "features": feature_cols,
        "observations": args.notes or "",
    }
    report_path = render_html_report(args.outdir, context=context, figures=figures)
    print("Report metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    top_features = filter_features_by_importance(importance_df, k_top=5)
    if top_features:
        print("Top features (permutation importance): " + ", ".join(top_features))
    print(f"HTML report saved to {report_path}")


def cmd_ml_bench(args: argparse.Namespace) -> None:
    dataframe = pd.read_csv(args.table)
    try:
        feature_sets = parse_feature_sets(args.feature_sets)
        targets = parse_targets(args.targets)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc
    models = parse_model_aliases(args.models)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    available = list_available_models(args.task)
    if models is not None:
        for alias in models:
            if alias not in available:
                print(f"[skip] modelo '{alias}' indisponível (dependência ausente)")

    results = run_benchmark_matrix(
        df_features=dataframe,
        task=args.task,
        targets=targets,
        feature_sets=feature_sets,
        models=models,
        group_col=args.group_col,
        base_outdir=outdir,
    )

    bench_csv = outdir / "bench_results.csv"
    completed = sum(
        not json.loads(raw).get("skipped", False) for raw in results["metrics_json"]
    )
    print(
        "Benchmark finished: "
        f"{completed} completed / {len(results)} total. Results saved to {bench_csv}"
    )


def cmd_ml_compare(args: argparse.Namespace) -> None:
    bench_csv = Path(args.bench_csv)
    outdir = Path(args.outdir)
    if not bench_csv.exists():
        raise SystemExit(f"Benchmark CSV not found: {bench_csv}")
    outdir.mkdir(parents=True, exist_ok=True)

    summary = compare_models(bench_csv, args.task)
    summary_path = bench_csv.parent / "bench_summary.csv"
    ranking_path = bench_csv.parent / "ranking.png"

    if bench_csv.parent != outdir:
        target_summary = outdir / "bench_summary.csv"
        target_ranking = outdir / "ranking.png"
        shutil.copy2(summary_path, target_summary)
        shutil.copy2(ranking_path, target_ranking)
        summary_path = target_summary
        ranking_path = target_ranking

    print(summary.to_string(index=False))
    print(f"Summary saved to {summary_path}")
    print(f"Ranking plot saved to {ranking_path}")


def cmd_theta(args: argparse.Namespace) -> None:
    ogum = OgumLite()
    ogum.load_csv(args.input)

    theta_df = ogum.compute_theta_table(
        args.ea,
        temperature_column=args.temperature_column,
        time_column=args.time_column,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / "theta_table.csv"
    theta_df.to_csv(output_path, index=False)

    print(f"θ(Ea) table saved to {output_path}")
    if args.print:
        print(theta_df.to_string(index=False))


def cmd_msc(args: argparse.Namespace) -> None:
    dataframe = pd.read_csv(args.input)
    normalize_theta = None if args.normalize_theta == "none" else args.normalize_theta
    try:
        segments = resolve_stage_segments(
            dataframe,
            segmentation_mode=args.segmentation_mode,
            stage_text=args.stages,
            group_col=args.group_col,
            t_col=args.time_column,
            y_col=args.y_column,
            n_segments=args.segments,
            min_size=args.min_size,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.segmentation_mode == "max-rate":
        bounds_text = ", ".join(f"{lower:.3f}-{upper:.3f}" for lower, upper in segments)
        print(f"Max-rate segments: {bounds_text}")

    ea_values: list[float] = []
    if args.ea:
        ea_values.extend(args.ea)
    if args.ea_range:
        ea_values.extend(args.ea_range)
    if not ea_values:
        raise SystemExit("Provide --ea or --ea-range to evaluate MSC")
    ea_values = sorted({float(value) for value in ea_values})

    summary, best_result, _ = score_activation_energies(
        dataframe,
        ea_values,
        metric=args.metric,
        group_col=args.group_col,
        t_col=args.time_column,
        temp_col=args.temperature_column,
        y_col=args.y_column,
        normalize_theta=normalize_theta,
        segments=segments,
    )

    print(summary.to_string(index=False))
    print(
        "best_Ea_kJ_mol={:.3f} mse_global={:.6f} mse_segmented={:.6f}".format(
            best_result.activation_energy,
            best_result.mse_global,
            best_result.mse_segmented,
        )
    )

    auto_enabled = args.auto_ea or args.auto_ea_plot is not None
    if auto_enabled:
        metric_key = "mse_segmented" if args.metric == "segmented" else "mse_global"
        if metric_key not in summary.columns:
            raise SystemExit("Summary does not contain the selected metric")
        best_idx = summary[metric_key].idxmin()
        best_ea = summary.loc[best_idx, "Ea_kJ_mol"]
        print(f"[auto-ea] Suggested Ea ({metric_key}) = {best_ea:.3f} kJ/mol")
        if args.auto_ea_plot:
            fig, ax = plt.subplots()
            ax.plot(summary["Ea_kJ_mol"], summary["mse_global"], label="MSE global")
            ax.plot(
                summary["Ea_kJ_mol"],
                summary["mse_segmented"],
                label="MSE segmentado",
            )
            ax.set_xlabel("Ea (kJ/mol)")
            ax.set_ylabel("Erro médio")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            args.auto_ea_plot.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.auto_ea_plot, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Auto-Ea scan saved to {args.auto_ea_plot}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        best_result.curve.to_csv(args.csv, index=False)
        print(f"MSC curve exported to {args.csv}")

    if args.png:
        fig, ax = plt.subplots()
        for sample_id, sample_df in best_result.per_sample.groupby(args.group_col):
            ax.plot(
                sample_df["theta"],
                sample_df["densification"],
                alpha=0.4,
                linewidth=1.0,
                label=str(sample_id),
            )
        ax.plot(
            best_result.curve["theta_mean"],
            best_result.curve["densification"],
            color="black",
            linewidth=2.0,
            label="Mean MSC",
        )
        ax.set_xlabel(r"$\Theta(E_a)$ (normalised)")
        ax.set_ylabel("Densification")
        title = (
            f"MSC collapse (Ea={best_result.activation_energy:.1f} kJ/mol) "
            f"[metric={args.metric}]"
        )
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) > 10:
            ax.legend(handles[:1] + [handles[-1]], ["samples", "Mean MSC"], loc="best")
        else:
            ax.legend(loc="best")
        args.png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"MSC plot saved to {args.png}")


def cmd_maps(args: argparse.Namespace) -> None:
    dataframe = pd.read_csv(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    configurations = [
        (
            "blaine_n",
            "Blaine n heatmap",
            "magma",
            "blaine_n_heatmap.png",
            ".2f",
        ),
        (
            "blaine_mse",
            "Blaine MSE heatmap",
            "viridis",
            "blaine_mse_heatmap.png",
            ".3f",
        ),
    ]

    generated = False
    for metric, title, cmap, filename, fmt in configurations:
        matrix = prepare_segment_heatmap(
            dataframe,
            metric=metric,
            group_col=args.group_col,
        )
        if matrix.empty:
            print(f"No columns ending with _{metric}; skipping heatmap.")
            continue
        png = render_segment_heatmap(matrix, title=title, cmap=cmap, fmt=fmt)
        path = outdir / filename
        path.write_bytes(png)
        print(f"Saved {metric} heatmap to {path}")
        generated = True

    if not generated:
        raise SystemExit(
            "No Blaine segmentation metrics found. Provide a segment feature table."
        )


def cmd_api(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise SystemExit("uvicorn is required to launch the API") from exc

    uvicorn.run(
        "ogum_lite.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def launch_ui() -> None:
    """Launch a Gradio UI covering mapping, derivatives, Arrhenius and MSC."""

    def _to_temp_csv(df: pd.DataFrame, prefix: str) -> str:
        tmpdir = Path(tempfile.mkdtemp(prefix="ogum_ml_ui_"))
        path = tmpdir / f"{prefix}.csv"
        df.to_csv(path, index=False)
        return str(path)

    def load_file(
        file: gr.File | None,
    ) -> tuple[
        gr.Dataframe | None,
        pd.DataFrame | None,
        gr.Dropdown,
        gr.Dropdown,
        gr.Dropdown,
        gr.Dropdown,
        gr.Dropdown,
        gr.Textbox,
        gr.Dropdown,
        gr.Textbox,
        gr.Radio,
        gr.Radio,
        gr.Textbox,
        gr.Textbox,
    ]:
        if file is None:
            empty_update = gr.Dropdown.update(choices=[], value=None)
            radio_update = gr.Radio.update(value="s")
            return (
                gr.Dataframe.update(value=None),
                None,
                empty_update,
                empty_update,
                empty_update,
                empty_update,
                empty_update,
                gr.Textbox.update(value=""),
                gr.Dropdown.update(
                    choices=TECHNIQUE_CHOICES, value=TECHNIQUE_CHOICES[0]
                ),
                gr.Textbox.update(value=""),
                radio_update,
                gr.Radio.update(value="C"),
                gr.Textbox.update(value=""),
                gr.Textbox.update(value=""),
            )

        dataframe = read_table(file.name)
        mapping = infer_mapping(
            dataframe,
            default_composition="",
            default_technique=TECHNIQUE_CHOICES[0],
        )
        columns = [str(col) for col in dataframe.columns]
        technique_choices = columns + TECHNIQUE_CHOICES
        composition_value = (
            mapping.composition if mapping.composition in columns else None
        )
        technique_selection = (
            mapping.technique
            if mapping.technique in columns
            else mapping.technique_default or TECHNIQUE_CHOICES[0]
        )

        return (
            gr.Dataframe.update(value=dataframe.head(10)),
            dataframe,
            gr.Dropdown.update(choices=columns, value=mapping.sample_id),
            gr.Dropdown.update(choices=columns, value=mapping.time_col),
            gr.Dropdown.update(choices=columns, value=mapping.temp_col),
            gr.Dropdown.update(choices=columns, value=mapping.y_col),
            gr.Dropdown.update(choices=columns, value=composition_value),
            gr.Textbox.update(value=mapping.composition_default or ""),
            gr.Dropdown.update(choices=technique_choices, value=technique_selection),
            gr.Textbox.update(value=mapping.tech_comment or ""),
            gr.Radio.update(value=mapping.time_unit),
            gr.Radio.update(value=mapping.temp_unit),
            gr.Textbox.update(value=mapping.user or ""),
            gr.Textbox.update(value=mapping.timestamp or ""),
        )

    def derive_callback(
        dataframe: pd.DataFrame | None,
        sample_col: str,
        time_col: str,
        temp_col: str,
        y_col: str,
        composition_col: str,
        composition_default: str,
        technique_value: str,
        tech_comment: str,
        time_unit: str,
        temp_unit: str,
        user_name: str,
        timestamp_text: str,
        smooth: str,
        window: float,
        poly: float,
        moving_k: float,
    ) -> tuple[gr.Dataframe | None, pd.DataFrame | None, gr.File]:
        if dataframe is None:
            return gr.Dataframe.update(value=None), None, gr.File.update(value=None)

        columns = [str(col) for col in dataframe.columns]
        composition_value = composition_col if composition_col in columns else None
        technique_column = technique_value if technique_value in columns else None
        technique_default = (
            technique_value if technique_value in TECHNIQUE_CHOICES else None
        )
        extra = {}
        mapping = ColumnMap(
            sample_id=sample_col,
            time_col=time_col,
            temp_col=temp_col,
            y_col=y_col,
            composition=composition_value,
            technique=technique_column,
            composition_default=composition_default.strip() or None,
            technique_default=technique_default,
            tech_comment=tech_comment.strip() or None,
            user=user_name.strip() or None,
            timestamp=timestamp_text.strip() or None,
            time_unit=time_unit,  # type: ignore[arg-type]
            temp_unit=temp_unit,  # type: ignore[arg-type]
            extra_metadata=extra,
        )
        if mapping.technique is None and mapping.technique_default:
            profile = TECHNIQUE_PROFILES.get(mapping.technique_default)
            if profile:
                enriched = dict(mapping.extra_metadata)
                for key, value in profile.items():
                    if key == "name":
                        continue
                    enriched[f"tech_{key}"] = str(value)
                mapping = replace(mapping, extra_metadata=enriched)

        mapping = ColumnMap(**asdict(mapping))
        mapped = apply_mapping(dataframe, mapping)
        response = convert_response(mapped, column="rho_rel")
        mapped["rho_rel"] = response
        mapped["response"] = response
        mapped["y"] = response
        derived = derive_all(
            mapped,
            smooth=smooth,
            window=int(window),
            poly=int(poly),
            moving_k=int(moving_k),
            y_col="rho_rel",
        )
        csv_path = _to_temp_csv(derived, "derivatives")
        return (
            gr.Dataframe.update(value=derived.head(15)),
            derived,
            gr.File.update(value=csv_path),
        )

    def arrhenius_callback(
        derived: pd.DataFrame | None,
        stages_text: str,
    ) -> tuple[gr.Dataframe | None, gr.File, plt.Figure | None]:
        if derived is None:
            return gr.Dataframe.update(value=None), gr.File.update(value=None), None

        stages = parse_stage_ranges(stages_text)
        arrhenius_table = arrhenius_feature_table(derived, stages=stages)
        csv_path = _to_temp_csv(arrhenius_table, "arrhenius")

        figure: plt.Figure | None = None
        try:
            prepared = arrhenius_lnT_dy_dt_vs_invT(derived)
            fit = fit_arrhenius_global(prepared)
            fig, ax = plt.subplots()
            ax.scatter(prepared["invT_K"], prepared["ln_T_dy_dt"], s=12, alpha=0.6)
            x_grid = np.linspace(
                prepared["invT_K"].min(), prepared["invT_K"].max(), 200
            )
            y_grid = fit.slope * x_grid + fit.intercept
            ax.plot(x_grid, y_grid, color="red", linewidth=2.0, label="Global fit")
            ax.set_xlabel("1/T (1/K)")
            ax.set_ylabel("ln(T·dy/dt)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            figure = fig
        except ValueError:
            figure = None

        return (
            gr.Dataframe.update(value=arrhenius_table),
            gr.File.update(value=csv_path),
            figure,
        )

    def features_callback(
        derived: pd.DataFrame | None,
        stages_text: str,
        theta_text: str,
    ) -> tuple[gr.Dataframe | None, gr.File]:
        if derived is None:
            return gr.Dataframe.update(value=None), gr.File.update(value=None)

        stages = parse_stage_ranges(stages_text)
        theta_values = parse_ea_list(theta_text) if theta_text else None
        feature_store = build_feature_store(
            derived,
            stages=stages,
            theta_ea_kj=theta_values,
        )
        csv_path = _to_temp_csv(feature_store, "feature_store")
        return gr.Dataframe.update(value=feature_store), gr.File.update(value=csv_path)

    def msc_process(
        file: gr.File,
        t_min: float | None,
        t_max: float | None,
        shift: float,
        scale: float,
        ea_list: str,
        metric: str,
    ) -> tuple[pd.DataFrame, gr.File | None, gr.File | None, gr.File | None]:
        dataframe = pd.read_csv(file.name)
        if t_min is not None:
            dataframe = dataframe[dataframe["time_s"] >= t_min]
        if t_max is not None:
            dataframe = dataframe[dataframe["time_s"] <= t_max]

        dataframe = dataframe.copy()
        dataframe["rho_rel"] = (dataframe["rho_rel"] + shift) * scale

        ea_values = parse_ea_list(ea_list)
        summary, best_result, _ = score_activation_energies(
            dataframe,
            ea_values,
            metric=metric,
        )

        tmpdir = Path(tempfile.mkdtemp(prefix="ogum_ml_ui_"))

        summary_path = tmpdir / "best_ea_mse.csv"
        summary.to_csv(summary_path, index=False)

        curve_path = tmpdir / "msc_curve.csv"
        best_result.curve.to_csv(curve_path, index=False)

        zip_path = tmpdir / "theta_curves.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for sample_id, sample_df in best_result.per_sample.groupby("sample_id"):
                buffer = sample_df.to_csv(index=False).encode("utf-8")
                zf.writestr(f"theta_{sample_id}.csv", buffer)

        return (
            summary,
            gr.File.update(value=str(summary_path)),
            gr.File.update(value=str(curve_path)),
            gr.File.update(value=str(zip_path)),
        )

    with gr.Blocks() as demo:
        gr.Markdown("## Ogum ML Lite – Pipeline & MSC")

        raw_state = gr.State()
        derived_state = gr.State()

        with gr.Tab("Pipeline"):
            file_input = gr.File(label="Planilha (.csv/.xlsx)")
            preview = gr.Dataframe(label="Pré-visualização", interactive=False)

            with gr.Row():
                sample_dd = gr.Dropdown(label="sample_id")
                time_dd = gr.Dropdown(label="time")
                temp_dd = gr.Dropdown(label="temperature")
                y_dd = gr.Dropdown(label="response")
            with gr.Row():
                comp_dd = gr.Dropdown(label="composition")
                comp_default_tb = gr.Textbox(
                    label="composition (default)", placeholder="Ex.: 316L"
                )
                technique_dd = gr.Dropdown(label="technique", choices=TECHNIQUE_CHOICES)
                tech_comment_tb = gr.Textbox(
                    label="tech_comment", placeholder="Detalhes quando 'Outro'"
                )
            with gr.Row():
                time_unit_radio = gr.Radio(
                    label="Unidade de tempo",
                    choices=["s", "min"],
                    value="s",
                    interactive=True,
                )
                temp_unit_radio = gr.Radio(
                    label="Unidade de temperatura",
                    choices=["C", "K"],
                    value="C",
                    interactive=True,
                )
                user_tb = gr.Textbox(label="Operador", placeholder="Nome do operador")
                timestamp_tb = gr.Textbox(
                    label="Timestamp ISO", placeholder="2024-03-18T12:34:56Z"
                )

            smooth_radio = gr.Radio(
                label="Suavização",
                choices=["savgol", "moving", "none"],
                value="savgol",
            )
            window_slider = gr.Slider(5, 51, step=2, value=11, label="Janela")
            poly_slider = gr.Slider(1, 5, step=1, value=3, label="Ordem polinômio")
            moving_slider = gr.Slider(
                3, 25, step=2, value=5, label="Janela média móvel"
            )
            stages_text = gr.Textbox(
                label="Estágios",
                value="0.55-0.70,0.70-0.90",
            )
            theta_text = gr.Textbox(
                label="Ea para θ(Ea) (opcional)",
                value="",
            )

            derive_button = gr.Button("Gerar derivadas")
            derived_preview = gr.Dataframe(label="Derivadas", interactive=False)
            derived_file = gr.File(label="derivatives.csv")

            arrhenius_button = gr.Button("Arrhenius fit")
            arrhenius_df = gr.Dataframe(label="Arrhenius", interactive=False)
            arrhenius_file = gr.File(label="arrhenius.csv")
            arrhenius_plot = gr.Plot(label="ln(T·dy/dt) vs 1/T")

            features_button = gr.Button("Feature store")
            features_df = gr.Dataframe(label="Feature store", interactive=False)
            features_file = gr.File(label="feature_store.csv")

            file_input.change(
                load_file,
                inputs=file_input,
                outputs=[
                    preview,
                    raw_state,
                    sample_dd,
                    time_dd,
                    temp_dd,
                    y_dd,
                    comp_dd,
                    comp_default_tb,
                    technique_dd,
                    tech_comment_tb,
                    time_unit_radio,
                    temp_unit_radio,
                    user_tb,
                    timestamp_tb,
                ],
            )

            derive_button.click(
                derive_callback,
                inputs=[
                    raw_state,
                    sample_dd,
                    time_dd,
                    temp_dd,
                    y_dd,
                    comp_dd,
                    comp_default_tb,
                    technique_dd,
                    tech_comment_tb,
                    time_unit_radio,
                    temp_unit_radio,
                    user_tb,
                    timestamp_tb,
                    smooth_radio,
                    window_slider,
                    poly_slider,
                    moving_slider,
                ],
                outputs=[derived_preview, derived_state, derived_file],
            )

            arrhenius_button.click(
                arrhenius_callback,
                inputs=[derived_state, stages_text],
                outputs=[arrhenius_df, arrhenius_file, arrhenius_plot],
            )

            features_button.click(
                features_callback,
                inputs=[derived_state, stages_text, theta_text],
                outputs=[features_df, features_file],
            )

        with gr.Tab("MSC"):
            file_input_msc = gr.File(label="Ensaios CSV (long format)")
            with gr.Row():
                t_min = gr.Number(label="t_min (s)", value=None)
                t_max = gr.Number(label="t_max (s)", value=None)
                shift = gr.Number(label="shift", value=0.0)
                scale = gr.Number(label="scale", value=1.0)
            ea_box = gr.Textbox(label="Ea list (kJ/mol)", value="200,300,400")
            metric_radio = gr.Radio(
                label="Métrica de colapso",
                choices=["segmented", "global"],
                value="segmented",
            )

            process_button = gr.Button("Calcular MSC")
            summary_df = gr.Dataframe(label="Resumo θ(Ea)")
            summary_file = gr.File(label="best_ea_mse.csv")
            curve_file = gr.File(label="msc_curve.csv")
            theta_zip = gr.File(label="theta_curves.zip")

            process_button.click(
                msc_process,
                inputs=[
                    file_input_msc,
                    t_min,
                    t_max,
                    shift,
                    scale,
                    ea_box,
                    metric_radio,
                ],
                outputs=[summary_df, summary_file, curve_file, theta_zip],
            )

    demo.launch()


def cmd_ui(_: argparse.Namespace) -> None:
    launch_ui()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ogum ML Lite CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_validate = subparsers.add_parser("validate", help="Data validation helpers")
    validate_subparsers = parser_validate.add_subparsers(
        dest="validate_command", required=True
    )
    parser_validate_long = validate_subparsers.add_parser(
        "long", help="Validar tabela em formato long"
    )
    parser_validate_long.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV longo com colunas sample_id,time_s,temp_C e densificação.",
    )
    parser_validate_long.add_argument(
        "--y-col",
        default="rho_rel",
        help="Coluna usada como densificação (padrão: rho_rel).",
    )
    parser_validate_long.add_argument(
        "--out",
        type=Path,
        default=Path("validation_long.json"),
        help="Arquivo JSON com o relatório da validação.",
    )
    parser_validate_long.set_defaults(func=cmd_validate_long)

    parser_validate_features = validate_subparsers.add_parser(
        "features", help="Validar tabela de features por amostra"
    )
    parser_validate_features.add_argument(
        "--table",
        type=Path,
        required=True,
        help="CSV com features derivadas por amostra.",
    )
    parser_validate_features.add_argument(
        "--out",
        type=Path,
        default=Path("validation_features.json"),
        help="Arquivo JSON com o relatório da validação.",
    )
    parser_validate_features.set_defaults(func=cmd_validate_features)

    parser_io = subparsers.add_parser("io", help="I/O utilities")
    io_subparsers = parser_io.add_subparsers(dest="io_command", required=True)
    parser_io_map = io_subparsers.add_parser(
        "map", help="Infer column mapping from spreadsheets"
    )
    parser_io_map.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Arquivo .csv/.xls/.xlsx com dados brutos.",
    )
    parser_io_map.add_argument(
        "--out",
        type=Path,
        default=Path("column_mapping.json"),
        help="Arquivo JSON onde o mapeamento será salvo.",
    )
    parser_io_map.add_argument(
        "--edit",
        action="store_true",
        help="Permite ajustar o mapeamento inferido manualmente.",
    )
    parser_io_map.set_defaults(func=cmd_io_map)

    parser_pre = subparsers.add_parser("preprocess", help="Pre-processing helpers")
    pre_subparsers = parser_pre.add_subparsers(dest="preprocess_command", required=True)
    parser_pre_derive = pre_subparsers.add_parser(
        "derive", help="Aplicar mapeamento e calcular derivadas"
    )
    parser_pre_derive.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Arquivo .csv/.xls/.xlsx com dados brutos.",
    )
    parser_pre_derive.add_argument(
        "--map",
        type=Path,
        required=True,
        help="Arquivo JSON com o mapeamento de colunas.",
    )
    parser_pre_derive.add_argument(
        "--response-type",
        choices=["auto", "shrinkage", "delta", "density"],
        default="auto",
        help="Tipo da coluna de resposta para normalização automática.",
    )
    parser_pre_derive.add_argument(
        "--L0",
        type=float,
        help="Comprimento inicial (necessário para response-type=delta).",
    )
    parser_pre_derive.add_argument(
        "--rho0",
        type=float,
        help="Densidade inicial (necessária para response-type=density).",
    )
    parser_pre_derive.add_argument(
        "--smooth",
        choices=["savgol", "moving", "none"],
        default="savgol",
        help="Método de suavização aplicado antes das derivadas.",
    )
    parser_pre_derive.add_argument(
        "--window",
        type=int,
        default=11,
        help="Janela do filtro de suavização.",
    )
    parser_pre_derive.add_argument(
        "--poly",
        type=int,
        default=3,
        help="Ordem do polinômio no Savitzky-Golay.",
    )
    parser_pre_derive.add_argument(
        "--moving-k",
        type=int,
        default=5,
        help="Tamanho da janela para média móvel.",
    )
    parser_pre_derive.add_argument(
        "--out",
        type=Path,
        default=Path("derivatives.csv"),
        help="Arquivo CSV com as derivadas normalizadas.",
    )
    parser_pre_derive.add_argument(
        "--print",
        action="store_true",
        help="Imprime as primeiras linhas da tabela derivada.",
    )
    parser_pre_derive.set_defaults(func=cmd_preprocess_derive)

    parser_arr = subparsers.add_parser("arrhenius", help="Arrhenius regressions")
    arr_subparsers = parser_arr.add_subparsers(dest="arrhenius_command", required=True)
    parser_arr_fit = arr_subparsers.add_parser(
        "fit", help="Ajustar Ea global e por estágios"
    )
    parser_arr_fit.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV com colunas sample_id,time_s,temp_C,y e derivadas.",
    )
    parser_arr_fit.add_argument(
        "--out",
        type=Path,
        default=Path("arrhenius.csv"),
        help="Arquivo CSV com os resultados de Arrhenius.",
    )
    parser_arr_fit.add_argument(
        "--png",
        type=Path,
        help="Figura opcional ln(T·dy/dt) vs 1/T.",
    )
    parser_arr_fit.add_argument(
        "--stages",
        default=None,
        help="Faixas de densificação (ex.: '0.55-0.70,0.70-0.90').",
    )
    parser_arr_fit.add_argument(
        "--segmentation-mode",
        choices=["fixed", "max-rate"],
        default="fixed",
        help="Modo de definição dos estágios para Arrhenius.",
    )
    parser_arr_fit.add_argument(
        "--segments",
        type=int,
        default=3,
        help="Número de segmentos usados no modo max-rate.",
    )
    parser_arr_fit.add_argument(
        "--min-size",
        type=int,
        default=5,
        help="Número mínimo de pontos por segmento (max-rate).",
    )
    parser_arr_fit.add_argument(
        "--group-col",
        default="sample_id",
        help="Coluna de identificação da amostra.",
    )
    parser_arr_fit.add_argument(
        "--time-column",
        default="time_s",
        help="Nome da coluna de tempo (s).",
    )
    parser_arr_fit.add_argument(
        "--temperature-column",
        default="temp_C",
        help="Nome da coluna de temperatura (°C).",
    )
    parser_arr_fit.add_argument(
        "--y-column",
        default="y",
        help="Coluna de densificação/response normalizada.",
    )
    parser_arr_fit.add_argument(
        "--smooth",
        choices=["savgol", "moving", "none"],
        default="savgol",
        help="Método de suavização usado caso seja necessário recalcular.",
    )
    parser_arr_fit.add_argument(
        "--window",
        type=int,
        default=11,
        help="Janela do filtro de suavização.",
    )
    parser_arr_fit.add_argument(
        "--poly",
        type=int,
        default=3,
        help="Ordem do polinômio no Savitzky-Golay.",
    )
    parser_arr_fit.add_argument(
        "--moving-k",
        type=int,
        default=5,
        help="Tamanho da janela da média móvel.",
    )
    parser_arr_fit.set_defaults(func=cmd_arrhenius_fit)

    parser_segmentation = subparsers.add_parser(
        "segmentation", help="Segmentar curvas de densificação"
    )
    parser_segmentation.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV longo com colunas sample_id,time_s,temp_C,rho_rel.",
    )
    parser_segmentation.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Arquivo JSON com segmentos detectados.",
    )
    parser_segmentation.add_argument(
        "--mode",
        choices=["fixed", "data", "max-rate"],
        default="fixed",
        help="Modo de segmentação: limiares fixos, data-driven ou pico de taxa.",
    )
    parser_segmentation.add_argument(
        "--group-col",
        default="sample_id",
        help="Coluna de agrupamento das amostras.",
    )
    parser_segmentation.add_argument(
        "--time-column",
        default="time_s",
        help="Coluna de tempo em segundos.",
    )
    parser_segmentation.add_argument(
        "--y-column",
        default="rho_rel",
        help="Coluna de densificação relativa (0–1).",
    )
    parser_segmentation.add_argument(
        "--thresholds",
        default=None,
        help="Lista de limiares (ex.: '0.55,0.70,0.90') para o modo fixed.",
    )
    parser_segmentation.add_argument(
        "--segments",
        type=int,
        default=3,
        help="Número de segmentos para o modo data-driven.",
    )
    parser_segmentation.add_argument(
        "--min-size",
        type=int,
        default=5,
        help="Número mínimo de pontos por segmento.",
    )
    parser_segmentation.set_defaults(func=cmd_segmentation)

    parser_mechanism = subparsers.add_parser(
        "mechanism", help="Detectar mudança de mecanismo via piecewise linear"
    )
    parser_mechanism.add_argument(
        "--theta",
        type=Path,
        required=True,
        help="CSV com colunas theta e densification.",
    )
    parser_mechanism.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Arquivo CSV para salvar o resumo do mecanismo.",
    )
    parser_mechanism.add_argument(
        "--group-col",
        default="sample_id",
        help="Coluna de agrupamento (opcional).",
    )
    parser_mechanism.add_argument(
        "--theta-column",
        default="theta",
        help="Nome da coluna com θ acumulado.",
    )
    parser_mechanism.add_argument(
        "--y-column",
        default="densification",
        help="Coluna de densificação associada a θ.",
    )
    parser_mechanism.add_argument(
        "--segments",
        type=int,
        default=2,
        help="Número máximo de segmentos no ajuste piecewise.",
    )
    parser_mechanism.add_argument(
        "--min-size",
        type=int,
        default=5,
        help="Número mínimo de pontos por segmento.",
    )
    parser_mechanism.add_argument(
        "--criterion",
        choices=["aic", "bic"],
        default="bic",
        help="Critério de seleção do modelo (AIC ou BIC).",
    )
    parser_mechanism.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Ganho mínimo no critério para sinalizar mudança de mecanismo.",
    )
    parser_mechanism.add_argument(
        "--slope-delta",
        type=float,
        default=0.02,
        help="Diferença mínima entre inclinações consecutivas para disparar mudança.",
    )
    parser_mechanism.set_defaults(func=cmd_mechanism)

    parser_features = subparsers.add_parser(
        "features", help="Utilitários de feature engineering"
    )
    parser_features.add_argument(
        "--input",
        type=Path,
        help="CSV longo com colunas sample_id,time_s,temp_C,rho_rel.",
    )
    parser_features.add_argument(
        "--output",
        type=Path,
        default=Path("features.csv"),
        help="Arquivo de saída para a tabela de features.",
    )
    parser_features.add_argument(
        "--ea",
        type=parse_ea_list,
        help="Lista de Ea em kJ/mol (ex.: '200,300,400').",
    )
    parser_features.add_argument(
        "--group-col",
        default="sample_id",
        help="Nome da coluna com o identificador da amostra.",
    )
    parser_features.add_argument(
        "--time-column",
        default="time_s",
        help="Nome da coluna de tempo (s).",
    )
    parser_features.add_argument(
        "--temperature-column",
        default="temp_C",
        help="Nome da coluna de temperatura (°C).",
    )
    parser_features.add_argument(
        "--y-column",
        default="rho_rel",
        help="Coluna de densificação relativa (0–1).",
    )
    parser_features.add_argument(
        "--print",
        action="store_true",
        help="Imprime a tabela resultante no stdout.",
    )
    parser_features.set_defaults(func=cmd_features, features_command="legacy")

    features_subparsers = parser_features.add_subparsers(dest="features_command")
    parser_features_build = features_subparsers.add_parser(
        "build", help="Montar feature store stage-aware"
    )
    parser_features_build.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV com colunas sample_id,time_s,temp_C,y e derivadas.",
    )
    parser_features_build.add_argument(
        "--output",
        type=Path,
        default=Path("feature_store.csv"),
        help="Arquivo CSV consolidado.",
    )
    parser_features_build.add_argument(
        "--stages",
        default=None,
        help="Faixas de densificação (ex.: '0.55-0.70,0.70-0.90').",
    )
    parser_features_build.add_argument(
        "--group-col",
        default="sample_id",
        help="Coluna de identificação da amostra.",
    )
    parser_features_build.add_argument(
        "--time-column",
        default="time_s",
        help="Coluna de tempo em segundos.",
    )
    parser_features_build.add_argument(
        "--temperature-column",
        default="temp_C",
        help="Coluna de temperatura em °C.",
    )
    parser_features_build.add_argument(
        "--y-column",
        default="y",
        help="Coluna de densificação normalizada.",
    )
    parser_features_build.add_argument(
        "--smooth",
        choices=["savgol", "moving", "none"],
        default="savgol",
        help="Método de suavização ao recalcular derivadas.",
    )
    parser_features_build.add_argument(
        "--window",
        type=int,
        default=11,
        help="Janela do filtro de suavização.",
    )
    parser_features_build.add_argument(
        "--poly",
        type=int,
        default=3,
        help="Ordem do polinômio no Savitzky-Golay.",
    )
    parser_features_build.add_argument(
        "--moving-k",
        type=int,
        default=5,
        help="Tamanho da janela da média móvel.",
    )
    parser_features_build.add_argument(
        "--theta-ea",
        help="Ea adicionais para integrar θ(Ea) (ex.: '200,300').",
    )
    parser_features_build.add_argument(
        "--print",
        action="store_true",
        help="Mostra a tabela completa no stdout.",
    )
    parser_features_build.set_defaults(func=cmd_features_build)

    parser_export = subparsers.add_parser("export", help="Exportação de artefatos")
    export_subparsers = parser_export.add_subparsers(
        dest="export_command", required=True
    )
    parser_export_xlsx = export_subparsers.add_parser(
        "xlsx", help="Gerar relatório consolidado em XLSX"
    )
    parser_export_xlsx.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Caminho do arquivo XLSX a ser criado.",
    )
    parser_export_xlsx.add_argument(
        "--msc",
        type=Path,
        help="CSV com a curva MSC consolidada.",
    )
    parser_export_xlsx.add_argument(
        "--features",
        type=Path,
        help="CSV com a tabela de features.",
    )
    parser_export_xlsx.add_argument(
        "--metrics",
        type=Path,
        help="JSON com métricas de validação e tuning.",
    )
    parser_export_xlsx.add_argument(
        "--dataset",
        help="Nome do conjunto de dados exibido na aba Summary.",
    )
    parser_export_xlsx.add_argument(
        "--notes",
        help="Observações adicionais para o relatório.",
    )
    parser_export_xlsx.add_argument(
        "--img-msc",
        dest="img_msc",
        type=Path,
        help="PNG opcional com a curva MSC.",
    )
    parser_export_xlsx.add_argument(
        "--img-cls",
        dest="img_cls",
        type=Path,
        help="PNG opcional com a matriz de confusão.",
    )
    parser_export_xlsx.add_argument(
        "--img-reg",
        dest="img_reg",
        type=Path,
        help="PNG opcional com o gráfico de regressão.",
    )
    parser_export_xlsx.set_defaults(func=cmd_export_xlsx)

    parser_export_onnx = export_subparsers.add_parser(
        "onnx", help="Exportar RandomForest treinado para ONNX"
    )
    parser_export_onnx.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Arquivo .joblib com o estimador treinado.",
    )
    parser_export_onnx.add_argument(
        "--features-json",
        dest="features_json",
        type=Path,
        required=True,
        help="JSON com a lista de colunas usadas no treino.",
    )
    parser_export_onnx.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Arquivo .onnx de saída.",
    )
    parser_export_onnx.set_defaults(func=cmd_export_onnx)

    parser_ml = subparsers.add_parser("ml", help="ML utilities")
    ml_subparsers = parser_ml.add_subparsers(dest="ml_command", required=True)

    parser_ml_features = ml_subparsers.add_parser(
        "features", help="Compute per-sample features for ML"
    )
    parser_ml_features.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV longo com colunas sample_id,time_s,temp_C,rho_rel.",
    )
    parser_ml_features.add_argument(
        "--output",
        type=Path,
        default=Path("features.csv"),
        help="Arquivo de saída para a tabela de features.",
    )
    parser_ml_features.add_argument(
        "--ea",
        type=parse_ea_list,
        required=True,
        help="Lista de Ea em kJ/mol (ex.: '200,300,400').",
    )
    parser_ml_features.add_argument(
        "--group-col",
        default="sample_id",
        help="Nome da coluna com o identificador da amostra.",
    )
    parser_ml_features.add_argument(
        "--time-column",
        default="time_s",
        help="Nome da coluna de tempo (s).",
    )
    parser_ml_features.add_argument(
        "--temperature-column",
        default="temp_C",
        help="Nome da coluna de temperatura (°C).",
    )
    parser_ml_features.add_argument(
        "--y-column",
        default="rho_rel",
        help="Coluna de densificação relativa (0–1).",
    )
    parser_ml_features.add_argument(
        "--print",
        action="store_true",
        help="Imprime a tabela resultante no stdout.",
    )
    parser_ml_features.set_defaults(func=cmd_ml_features)

    def _add_train_args(train_parser: argparse.ArgumentParser) -> None:
        train_parser.add_argument(
            "--table",
            type=Path,
            required=True,
            help="Tabela de features (CSV).",
        )
        train_parser.add_argument(
            "--target",
            required=True,
            help="Coluna alvo para o treinamento.",
        )
        train_parser.add_argument(
            "--group-col",
            required=True,
            help="Coluna com o identificador de grupo/amostra.",
        )
        train_parser.add_argument(
            "--features",
            nargs="+",
            required=True,
            help="Lista de colunas de entrada.",
        )
        train_parser.add_argument(
            "--outdir",
            type=Path,
            required=True,
            help="Diretório onde os artefatos serão salvos.",
        )

    parser_ml_train_cls = ml_subparsers.add_parser(
        "train-cls", help="Treinar classificador com GroupKFold"
    )
    _add_train_args(parser_ml_train_cls)
    parser_ml_train_cls.set_defaults(func=cmd_ml_train_classifier)

    parser_ml_train_reg = ml_subparsers.add_parser(
        "train-reg", help="Treinar regressor com GroupKFold"
    )
    _add_train_args(parser_ml_train_reg)
    parser_ml_train_reg.set_defaults(func=cmd_ml_train_regressor)

    parser_ml_tune_cls = ml_subparsers.add_parser(
        "tune-cls", help="RandomizedSearchCV para classificadores"
    )
    _add_train_args(parser_ml_tune_cls)
    parser_ml_tune_cls.add_argument(
        "--n-iter",
        type=int,
        default=40,
        help="Número de combinações amostradas na busca aleatória.",
    )
    parser_ml_tune_cls.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Número de splits do GroupKFold.",
    )
    parser_ml_tune_cls.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed para amostragem dos hiperparâmetros.",
    )
    parser_ml_tune_cls.set_defaults(func=cmd_ml_tune_classifier)

    parser_ml_tune_reg = ml_subparsers.add_parser(
        "tune-reg", help="RandomizedSearchCV para regressão"
    )
    _add_train_args(parser_ml_tune_reg)
    parser_ml_tune_reg.add_argument(
        "--n-iter",
        type=int,
        default=40,
        help="Número de combinações amostradas na busca aleatória.",
    )
    parser_ml_tune_reg.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Número de splits do GroupKFold.",
    )
    parser_ml_tune_reg.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed para amostragem dos hiperparâmetros.",
    )
    parser_ml_tune_reg.set_defaults(func=cmd_ml_tune_regressor)

    parser_ml_predict = ml_subparsers.add_parser(
        "predict", help="Gerar previsões usando artefato salvo"
    )
    parser_ml_predict.add_argument(
        "--table",
        type=Path,
        required=True,
        help="Tabela de features (CSV).",
    )
    parser_ml_predict.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Caminho para o arquivo .joblib salvo.",
    )
    parser_ml_predict.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Arquivo CSV para salvar as predições.",
    )
    parser_ml_predict.set_defaults(func=cmd_ml_predict)

    parser_ml_cluster = ml_subparsers.add_parser(
        "cluster", help="Clusterização exploratória via KMeans"
    )
    parser_ml_cluster.add_argument(
        "--table",
        type=Path,
        required=True,
        help="Tabela de features (CSV).",
    )
    parser_ml_cluster.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="Colunas numéricas usadas no KMeans.",
    )
    parser_ml_cluster.add_argument(
        "--k",
        type=int,
        default=3,
        help="Número de clusters (k).",
    )
    parser_ml_cluster.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Arquivo CSV para salvar os clusters.",
    )
    parser_ml_cluster.set_defaults(func=cmd_ml_cluster)

    parser_ml_report = ml_subparsers.add_parser(
        "report", help="Gerar relatório HTML com métricas e gráficos"
    )
    parser_ml_report.add_argument(
        "--table",
        type=Path,
        required=True,
        help="Tabela de features (CSV).",
    )
    parser_ml_report.add_argument(
        "--target",
        required=True,
        help="Coluna alvo presente na tabela.",
    )
    parser_ml_report.add_argument(
        "--group-col",
        required=True,
        help="Coluna de grupos/amostras (usada para estatísticas).",
    )
    parser_ml_report.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Artefato .joblib treinado/tunado.",
    )
    parser_ml_report.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Diretório onde os artefatos foram salvos (feature_cols, model_card).",
    )
    parser_ml_report.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Número de permutações para importância de features.",
    )
    parser_ml_report.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed para a importância por permutação.",
    )
    parser_ml_report.add_argument(
        "--notes",
        default="",
        help="Observações adicionais para o relatório.",
    )
    parser_ml_report.set_defaults(func=cmd_ml_report)

    parser_ml_bench = ml_subparsers.add_parser(
        "bench",
        help="Executar matriz de benchmarks com GroupKFold",
    )
    parser_ml_bench.add_argument(
        "--table",
        type=Path,
        required=True,
        help="CSV com as features consolidadas.",
    )
    parser_ml_bench.add_argument(
        "--task",
        choices=["cls", "reg"],
        required=True,
        help="Tipo de tarefa: cls (classificação) ou reg (regressão).",
    )
    parser_ml_bench.add_argument(
        "--targets",
        required=True,
        help="Colunas alvo (separadas por vírgula).",
    )
    parser_ml_bench.add_argument(
        "--feature-sets",
        nargs="+",
        required=True,
        help="Mapeamentos alias=col1,col2 para conjuntos de features.",
    )
    parser_ml_bench.add_argument(
        "--models",
        help="Lista de modelos (rf,lgbm,cat,xgb).",
    )
    parser_ml_bench.add_argument(
        "--group-col",
        required=True,
        help="Coluna de agrupamento (GroupKFold).",
    )
    parser_ml_bench.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Diretório base para salvar artefatos e bench_results.csv.",
    )
    parser_ml_bench.set_defaults(func=cmd_ml_bench)

    parser_ml_compare = ml_subparsers.add_parser(
        "compare",
        help="Gerar ranking e resumo de benchmarks",
    )
    parser_ml_compare.add_argument(
        "--bench-csv",
        type=Path,
        required=True,
        help="Caminho para o bench_results.csv consolidado.",
    )
    parser_ml_compare.add_argument(
        "--task",
        choices=["cls", "reg"],
        required=True,
        help="Tipo de tarefa avaliada no benchmark.",
    )
    parser_ml_compare.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Diretório onde bench_summary.csv e ranking.png serão salvos.",
    )
    parser_ml_compare.set_defaults(func=cmd_ml_compare)

    parser_theta = subparsers.add_parser("theta", help="Compute θ(Ea) table")
    parser_theta.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV file com ensaios",
    )
    parser_theta.add_argument(
        "--ea",
        type=parse_ea_list,
        required=True,
        help="Comma separated Ea values in kJ/mol (e.g. '200,300,400').",
    )
    parser_theta.add_argument(
        "--time-column",
        default="time_s",
        help="Nome da coluna de tempo (s).",
    )
    parser_theta.add_argument(
        "--temperature-column",
        default="temp_C",
        help="Nome da coluna de temperatura (°C).",
    )
    parser_theta.add_argument(
        "--outdir",
        type=Path,
        default=Path("exports"),
        help="Directory used to store the θ(Ea) table.",
    )
    parser_theta.add_argument(
        "--print",
        action="store_true",
        help="Print the resulting table to stdout.",
    )
    parser_theta.set_defaults(func=cmd_theta)

    parser_msc = subparsers.add_parser("msc", help="Generate Master Sintering Curve")
    parser_msc.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV file com ensaios",
    )
    parser_msc.add_argument(
        "--ea",
        type=parse_ea_list,
        help="Lista de Ea em kJ/mol para avaliar.",
    )
    parser_msc.add_argument(
        "--ea-range",
        type=parse_ea_range,
        help="Faixa de Ea no formato inicio:fim:passo.",
    )
    parser_msc.add_argument(
        "--group-col",
        default="sample_id",
        help="Nome da coluna com o identificador da amostra.",
    )
    parser_msc.add_argument(
        "--temperature-column",
        default="temp_C",
        help="Nome da coluna de temperatura (°C).",
    )
    parser_msc.add_argument(
        "--time-column",
        default="time_s",
        help="Column name containing time stamps in seconds.",
    )
    parser_msc.add_argument(
        "--y-column",
        default="rho_rel",
        help="Coluna de densificação relativa (0–1).",
    )
    parser_msc.add_argument(
        "--normalize-theta",
        choices=["minmax", "none"],
        default="minmax",
        help="Normalização aplicada antes do colapso MSC.",
    )
    parser_msc.add_argument(
        "--segmentation-mode",
        choices=["fixed", "max-rate"],
        default="fixed",
        help="Modo de segmentação para métricas segmentadas.",
    )
    parser_msc.add_argument(
        "--stages",
        default=None,
        help="Faixas de densificação (modo fixed).",
    )
    parser_msc.add_argument(
        "--segments",
        type=int,
        default=3,
        help="Número de segmentos ao usar o modo max-rate.",
    )
    parser_msc.add_argument(
        "--min-size",
        type=int,
        default=5,
        help="Número mínimo de pontos por segmento (max-rate).",
    )
    parser_msc.add_argument(
        "--metric",
        choices=["global", "segmented"],
        default="segmented",
        help="Métrica usada para escolher o melhor Ea.",
    )
    parser_msc.add_argument(
        "--csv",
        type=Path,
        help="Path to export the MSC table as CSV.",
    )
    parser_msc.add_argument(
        "--png",
        type=Path,
        help="Path to save the MSC figure.",
    )
    parser_msc.add_argument(
        "--auto-ea",
        action="store_true",
        help="Seleciona automaticamente o melhor Ea e gera diagnóstico.",
    )
    parser_msc.add_argument(
        "--auto-ea-plot",
        type=Path,
        help="Arquivo PNG opcional com o gráfico Ea vs erro.",
    )
    parser_msc.set_defaults(func=cmd_msc)

    parser_maps = subparsers.add_parser(
        "maps", help="Gerar heatmaps de Blaine/segmentos"
    )
    parser_maps.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV gerado por segment_feature_table ou feature store",
    )
    parser_maps.add_argument(
        "--outdir",
        type=Path,
        default=Path("maps"),
        help="Diretório para salvar os PNG dos mapas.",
    )
    parser_maps.add_argument(
        "--group-col",
        default="sample_id",
        help="Coluna com identificador das amostras.",
    )
    parser_maps.set_defaults(func=cmd_maps)

    parser_api = subparsers.add_parser("api", help="Rodar a API FastAPI")
    parser_api.add_argument("--host", default="0.0.0.0", help="Host de binding")
    parser_api.add_argument("--port", type=int, default=8000, help="Porta de escuta")
    parser_api.add_argument(
        "--reload",
        action="store_true",
        help="Habilitar auto-reload (desenvolvimento)",
    )
    parser_api.set_defaults(func=cmd_api)

    parser_ui = subparsers.add_parser("ui", help="Launch Gradio interface")
    parser_ui.set_defaults(func=cmd_ui)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
