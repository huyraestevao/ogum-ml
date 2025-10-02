"""Experiment management utilities for Ogum ML Lite."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from importlib import util as importlib_util
from pathlib import Path
from typing import Callable, Literal

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from .ml_hooks import (
    grouped_cv_scores,
    make_cat_classifier,
    make_cat_regressor,
    make_classifier,
    make_lgbm_classifier,
    make_lgbm_regressor,
    make_regressor,
    make_xgb_classifier,
    make_xgb_regressor,
)

TaskLiteral = Literal["cls", "reg"]
ModelFactory = Callable[[list[str], list[str]], Pipeline | None]


def _module_available(name: str) -> bool:
    return importlib_util.find_spec(name) is not None


def _split_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric.append(column)
        else:
            categorical.append(column)
    return numeric, categorical


def _ensure_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns: {missing_str}")


def _to_jsonable(obj: object) -> object:
    if isinstance(obj, dict):
        return {key: _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(value) for value in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def _dump_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def list_available_models(task: TaskLiteral) -> dict[str, ModelFactory]:
    """List factories for the requested task.

    Parameters
    ----------
    task
        ``"cls"`` for classification or ``"reg"`` for regression.

    Returns
    -------
    dict
        Mapping of model aliases to pipeline factories.
    """

    factories: dict[str, ModelFactory] = {}
    if task == "cls":
        factories["rf"] = lambda num, cat: make_classifier(num, cat, algo="rf")
        if _module_available("lightgbm"):
            factories["lgbm"] = make_lgbm_classifier
        if _module_available("catboost"):
            factories["cat"] = make_cat_classifier
        if _module_available("xgboost"):
            factories["xgb"] = make_xgb_classifier
    elif task == "reg":
        factories["rf"] = lambda num, cat: make_regressor(num, cat, algo="rf")
        if _module_available("lightgbm"):
            factories["lgbm"] = make_lgbm_regressor
        if _module_available("catboost"):
            factories["cat"] = make_cat_regressor
        if _module_available("xgboost"):
            factories["xgb"] = make_xgb_regressor
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported task '{task}'")
    return factories


def run_experiment(
    *,
    df_features: pd.DataFrame,
    target_col: str,
    group_col: str,
    feature_cols: list[str],
    model_key: str,
    outdir: Path,
    task: TaskLiteral,
) -> dict:
    """Train and evaluate a single experiment configuration.

    The resulting artifacts are stored in ``outdir / model_key``.
    """

    _ensure_columns(df_features, [target_col, group_col, *feature_cols])
    cleaned = df_features[[group_col, *feature_cols, target_col]].dropna(
        axis=0, how="any"
    )
    if cleaned.empty:
        raise ValueError("No samples available after dropping missing values")

    X = cleaned[feature_cols].copy()
    y = cleaned[target_col].copy()
    groups = cleaned[group_col].copy()
    num_cols, cat_cols = _split_feature_types(X)

    factories = list_available_models(task)
    factory = factories.get(model_key)
    if factory is None:
        return {"skipped": True, "reason": "model unavailable", "model": model_key}

    pipeline = factory(num_cols, cat_cols)
    if pipeline is None:
        return {"skipped": True, "reason": "missing dependency", "model": model_key}

    task_name = "classification" if task == "cls" else "regression"
    model_dir = outdir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    timestamp_start = datetime.now(timezone.utc)
    cv_scores = grouped_cv_scores(pipeline, X, y, groups, task=task_name)
    pipeline.fit(X, y)
    duration_s = time.perf_counter() - start
    timestamp_end = datetime.now(timezone.utc)

    model_path = model_dir / "model.joblib"
    joblib.dump(pipeline, model_path)

    feature_path = model_dir / "feature_cols.json"
    _dump_json(feature_path, {"features": feature_cols})

    target_path = model_dir / "target.json"
    _dump_json(target_path, {"target": target_col})

    metrics_payload: dict[str, object] = {
        "n_splits": cv_scores.get("n_splits"),
        "n_groups": cv_scores.get("n_groups"),
        "metrics": {},
    }
    for key, value in cv_scores.items():
        if key.endswith("_mean"):
            metric = key[:-5]
            metrics_payload["metrics"].setdefault(metric, {})["mean"] = value
        elif key.endswith("_std"):
            metric = key[:-4]
            metrics_payload["metrics"].setdefault(metric, {})["std"] = value

    cv_metrics_path = model_dir / "cv_metrics.json"
    _dump_json(cv_metrics_path, metrics_payload)

    estimator = pipeline.named_steps["model"]
    model_card = {
        "timestamp": timestamp_end.isoformat(),
        "task": task_name,
        "model_key": model_key,
        "estimator": estimator.__class__.__name__,
        "estimator_params": estimator.get_params(),
        "dataset": {
            "target": target_col,
            "group_col": group_col,
            "features": feature_cols,
            "n_samples": int(X.shape[0]),
            "n_features": int(len(feature_cols)),
            "n_groups": int(pd.Index(groups).nunique()),
        },
        "cv_metrics": metrics_payload,
    }
    model_card_path = model_dir / "model_card.json"
    _dump_json(model_card_path, model_card)

    training_log = {
        "task": task_name,
        "model": model_key,
        "target": target_col,
        "features": feature_cols,
        "group_col": group_col,
        "timestamp_start": timestamp_start.isoformat(),
        "timestamp_end": timestamp_end.isoformat(),
        "duration_s": duration_s,
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    training_log_path = model_dir / "training_log.json"
    _dump_json(training_log_path, training_log)

    return {
        "skipped": False,
        "model": model_key,
        "duration_s": duration_s,
        "cv_scores": cv_scores,
        "metrics_payload": metrics_payload,
        "artifacts": {
            "model": model_path,
            "feature_cols": feature_path,
            "target": target_path,
            "cv_metrics": cv_metrics_path,
            "model_card": model_card_path,
            "training_log": training_log_path,
        },
    }


def run_benchmark_matrix(
    *,
    df_features: pd.DataFrame,
    task: TaskLiteral,
    targets: list[str],
    feature_sets: dict[str, list[str]],
    models: list[str] | None,
    group_col: str,
    base_outdir: Path,
) -> pd.DataFrame:
    """Execute the cross-product of targets, feature sets and models."""

    base_outdir.mkdir(parents=True, exist_ok=True)
    available_factories = list_available_models(task)
    requested_models = models or list(available_factories.keys())
    records: list[dict[str, object]] = []

    for target in targets:
        for feature_name, cols in feature_sets.items():
            combo_outdir = base_outdir / target / feature_name
            for model_key in requested_models:
                model_outdir = combo_outdir / model_key
                if model_key not in available_factories:
                    metrics_payload = {"skipped": True, "reason": "model unavailable"}
                    records.append(
                        {
                            "task": task,
                            "target": target,
                            "feature_set": feature_name,
                            "model": model_key,
                            "metrics_json": json.dumps(metrics_payload),
                            "duration_s": 0.0,
                            "outdir": str(model_outdir),
                        }
                    )
                    continue

                result = run_experiment(
                    df_features=df_features,
                    target_col=target,
                    group_col=group_col,
                    feature_cols=list(cols),
                    model_key=model_key,
                    outdir=combo_outdir,
                    task=task,
                )
                if result.get("skipped"):
                    metrics_payload = result.get(
                        "metrics_payload",
                        {"skipped": True, "reason": "missing dependency"},
                    )
                    records.append(
                        {
                            "task": task,
                            "target": target,
                            "feature_set": feature_name,
                            "model": model_key,
                            "metrics_json": json.dumps(metrics_payload),
                            "duration_s": float(result.get("duration_s", 0.0)),
                            "outdir": str(model_outdir),
                        }
                    )
                    continue

                metrics_payload = result["metrics_payload"]
                records.append(
                    {
                        "task": task,
                        "target": target,
                        "feature_set": feature_name,
                        "model": model_key,
                        "metrics_json": json.dumps(metrics_payload),
                        "duration_s": float(result["duration_s"]),
                        "outdir": str(model_outdir),
                    }
                )

    df_results = pd.DataFrame.from_records(records)
    csv_path = base_outdir / "bench_results.csv"
    df_results.to_csv(
        csv_path,
        index=False,
        columns=[
            "task",
            "target",
            "feature_set",
            "model",
            "metrics_json",
            "duration_s",
            "outdir",
        ],
    )
    return df_results


def compare_models(bench_csv: Path, task: TaskLiteral) -> pd.DataFrame:
    """Generate a ranking summary from benchmark results."""

    df = pd.read_csv(bench_csv)
    if df.empty:
        raise ValueError("Benchmark results are empty")

    metrics_data = df["metrics_json"].fillna("{}").apply(json.loads)
    df = df.assign(metrics=metrics_data)
    df = df[df["metrics"].apply(lambda payload: not payload.get("skipped"))].copy()
    if df.empty:
        raise ValueError("No completed experiments to compare")

    key_metric: str
    alt_metric: str
    ascending: bool
    if task == "cls":
        key_metric = "accuracy"
        alt_metric = "f1_macro"
        ascending = False
    elif task == "reg":
        key_metric = "mae"
        alt_metric = "rmse"
        ascending = True
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported task '{task}'")

    def _extract(metric_name: str, payload: dict) -> float | None:
        metrics = payload.get("metrics", {})
        metric = metrics.get(metric_name)
        if metric is None:
            return None
        return metric.get("mean")

    df[f"{key_metric}_mean"] = [
        _extract(key_metric, payload) for payload in df["metrics"]
    ]
    df[f"{alt_metric}_mean"] = [
        _extract(alt_metric, payload) for payload in df["metrics"]
    ]
    df = df.dropna(subset=[f"{key_metric}_mean"]).copy()

    primary_col = f"{key_metric}_mean"
    group_cols = ["target", "feature_set"]
    best_series = (
        df.groupby(group_cols)[primary_col]
        .transform("min" if ascending else "max")
        .astype(float)
    )

    rel_delta: list[float] = []
    for value, best in zip(df[primary_col], best_series):
        if best == 0:
            rel_delta.append(0.0)
            continue
        if ascending:
            rel_delta.append(((value - best) / best) * 100.0)
        else:
            rel_delta.append(((best - value) / best) * 100.0)
    df["rel_delta_pct"] = rel_delta

    df["rank"] = df.groupby(group_cols)[primary_col].rank(
        method="min",
        ascending=ascending,
    )
    df = df.sort_values(group_cols + ["rank", "model"]).reset_index(drop=True)

    summary_path = bench_csv.parent / "bench_summary.csv"
    df.to_csv(summary_path, index=False)

    agg = df.groupby("model")[primary_col].mean().sort_values(ascending=ascending)
    fig, ax = plt.subplots(figsize=(6, 4))
    agg.plot(kind="bar", ax=ax)
    ylabel = "Higher is better" if not ascending else "Lower is better"
    ax.set_ylabel(ylabel)
    ax.set_title(f"Model ranking by {key_metric.upper()}")
    ax.set_xlabel("Model")
    fig.tight_layout()
    ranking_path = bench_csv.parent / "ranking.png"
    fig.savefig(ranking_path, dpi=150)
    plt.close(fig)

    return df
