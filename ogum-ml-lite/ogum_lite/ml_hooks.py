"""Machine learning pipelines and helpers for Ogum ML Lite."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance as skl_permutation_importance
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class _ColumnTypes:
    numeric: list[str]
    categorical: list[str]


def _infer_column_types(df: pd.DataFrame) -> _ColumnTypes:
    numeric: list[str] = []
    categorical: list[str] = []
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric.append(column)
        else:
            categorical.append(column)
    return _ColumnTypes(numeric=numeric, categorical=categorical)


def _ensure_required_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns: {missing_cols}")


def build_preprocessor(
    num_cols: Sequence[str], cat_cols: Sequence[str]
) -> ColumnTransformer:
    """Create a preprocessing transformer for numeric and categorical columns."""

    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), list(num_cols)))
    if cat_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(cat_cols),
            )
        )
    if not transformers:
        raise ValueError("At least one numeric or categorical column is required")
    return ColumnTransformer(transformers, remainder="drop")


def make_classifier(
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    *,
    algo: str = "rf",
    **kwargs,
) -> Pipeline:
    """Create a classification pipeline with preprocessing and estimator."""

    preprocessor = build_preprocessor(num_cols, cat_cols)
    if algo == "rf":
        estimator = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            **kwargs,
        )
    else:  # pragma: no cover - future algorithms
        raise ValueError(f"Unsupported classifier algorithm: {algo}")
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def make_regressor(
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    *,
    algo: str = "rf",
    **kwargs,
) -> Pipeline:
    """Create a regression pipeline with preprocessing and estimator."""

    preprocessor = build_preprocessor(num_cols, cat_cols)
    if algo == "rf":
        estimator = RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            **kwargs,
        )
    else:  # pragma: no cover - future algorithms
        raise ValueError(f"Unsupported regressor algorithm: {algo}")
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def _prepare_cv(groups: pd.Series) -> GroupKFold:
    unique_groups = pd.Index(groups).dropna().unique()
    n_unique = unique_groups.size
    if n_unique < 2:
        raise ValueError("At least two groups are required for cross-validation")
    n_splits = min(5, n_unique)
    return GroupKFold(n_splits=n_splits)


def grouped_cv_scores(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    task: str,
) -> dict[str, float]:
    """Evaluate a pipeline using grouped cross-validation.

    Parameters
    ----------
    model
        Fully specified :class:`~sklearn.pipeline.Pipeline` with preprocessing.
    X
        Feature matrix with the columns expected by ``model``.
    y
        Target values associated with each row in ``X``.
    groups
        Group identifiers used to keep samples from the same experiment
        together during validation.
    task
        Either ``"classification"`` or ``"regression"``.

    Returns
    -------
    dict
        Dictionary containing the number of splits and mean/std metrics.
    """

    groups_index = pd.Index(groups).dropna()
    cv = _prepare_cv(groups_index)
    scoring: dict[str, str]
    if task == "classification":
        scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}
    elif task == "regression":
        scoring = {
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        }
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported task: {task}")

    scores = cross_validate(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring=scoring,
        return_estimator=False,
        n_jobs=None,
    )

    results: dict[str, float] = {
        "n_splits": cv.get_n_splits(),
        "n_groups": int(groups_index.nunique()),
    }
    for key, values in scores.items():
        if not key.startswith("test_"):
            continue
        metric_name = key.replace("test_", "")
        metric_values = values
        if task == "regression":
            metric_values = -metric_values
        results[f"{metric_name}_mean"] = float(np.mean(metric_values))
        results[f"{metric_name}_std"] = float(np.std(metric_values, ddof=1))
    return results


def _clean_training_frame(
    df: pd.DataFrame,
    *,
    target_col: str,
    group_col: str,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    _ensure_required_columns(df, [target_col, group_col, *feature_cols])
    subset = df[[group_col, *feature_cols, target_col]].copy()
    subset = subset.dropna(axis=0, how="any")
    if subset.empty:
        raise ValueError("Training dataframe is empty after dropping missing values")
    return subset


def _make_group_kfold(groups: pd.Series, requested_splits: int) -> GroupKFold:
    unique_groups = pd.Index(groups).dropna().unique()
    if unique_groups.size < 2:
        raise ValueError("At least two groups are required for cross-validation")
    n_splits = min(max(requested_splits, 2), unique_groups.size)
    if n_splits < 2:
        raise ValueError("GroupKFold requires at least two splits")
    return GroupKFold(n_splits=n_splits)


def _update_feature_metadata(outdir: Path, feature_cols: Sequence[str]) -> Path:
    feature_cols_path = outdir / "feature_cols.json"
    _dump_json(feature_cols_path, {"features": list(feature_cols)})
    return feature_cols_path


def _update_model_card(
    outdir: Path,
    *,
    base_card: dict,
    best_params: dict,
    cv_metrics: dict[str, float],
    tuning_meta: dict[str, int | float | str],
) -> Path:
    model_card_path = outdir / "model_card.json"
    if model_card_path.exists():
        try:
            existing = json.loads(model_card_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:  # pragma: no cover - defensive
            existing = {}
    else:
        existing = {}

    model_card = {**existing, **base_card}
    model_card["best_params"] = best_params
    model_card["cross_validation"] = cv_metrics
    history = list(model_card.get("history", []))
    history.append(
        {
            "event": "tuning",
            "timestamp": base_card["timestamp"],
            "best_params": best_params,
            "cv_metrics": cv_metrics,
            "meta": tuning_meta,
        }
    )
    model_card["history"] = history
    _dump_json(model_card_path, model_card)
    return model_card_path


def random_search_classifier(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    feature_cols: list[str],
    outdir: Path,
    n_iter: int = 40,
    cv_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Tune a classification pipeline with RandomizedSearchCV.

    Parameters
    ----------
    df
        Feature dataframe containing ``target_col``, ``group_col`` and ``feature_cols``.
    target_col
        Name of the target column to be predicted.
    group_col
        Column used to build ``GroupKFold`` splits.
    feature_cols
        List of feature columns available for the model.
    outdir
        Directory where the tuned artifacts will be persisted.
    n_iter
        Number of sampled hyperparameter combinations.
    cv_splits
        Desired number of ``GroupKFold`` splits (capped by the number of groups).
    random_state
        Seed controlling sampling of hyperparameters.

    Returns
    -------
    dict
        Dictionary with best parameters, cross-validation summary and artifact paths.
    """

    cleaned = _clean_training_frame(
        df,
        target_col=target_col,
        group_col=group_col,
        feature_cols=feature_cols,
    )
    X = cleaned[list(feature_cols)]
    y = cleaned[target_col]
    groups = cleaned[group_col]
    column_types = _infer_column_types(X)
    pipeline = make_classifier(column_types.numeric, column_types.categorical)

    param_space = {
        "model__n_estimators": list(range(100, 601)),
        "model__max_depth": [None, *range(5, 31)],
        "model__max_features": ["sqrt", "log2", None],
        "model__min_samples_split": list(range(2, 11)),
    }

    cv = _make_group_kfold(groups, cv_splits)
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_space,
        n_iter=n_iter,
        scoring=scoring,
        refit="accuracy",
        cv=cv,
        random_state=random_state,
        n_jobs=None,
        return_train_score=False,
    )
    search.fit(X, y, groups=groups)

    cv_results_df = pd.DataFrame(search.cv_results_)
    best_idx = int(search.best_index_)
    cv_metrics: dict[str, float] = {
        "n_splits": cv.get_n_splits(),
        "n_groups": int(pd.Index(groups).nunique()),
        "accuracy_mean": float(cv_results_df.loc[best_idx, "mean_test_accuracy"]),
        "accuracy_std": float(cv_results_df.loc[best_idx, "std_test_accuracy"]),
        "f1_macro_mean": float(cv_results_df.loc[best_idx, "mean_test_f1_macro"]),
        "f1_macro_std": float(cv_results_df.loc[best_idx, "std_test_f1_macro"]),
    }

    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "classifier_tuned.joblib"
    joblib.dump(search.best_estimator_, model_path)

    feature_cols_path = _update_feature_metadata(outdir, feature_cols)
    target_path = outdir / "target.json"
    _dump_json(target_path, {"target": target_col})
    param_grid_path = outdir / "param_grid.json"
    _dump_json(
        param_grid_path,
        {
            "distributions": {
                key: (list(value) if isinstance(value, range) else value)
                for key, value in param_space.items()
            },
            "n_iter": n_iter,
            "random_state": random_state,
            "cv_splits": cv.get_n_splits(),
        },
    )

    cv_results_path = outdir / "cv_results.json"
    cv_payload = {"records": json.loads(cv_results_df.to_json(orient="records"))}
    _dump_json(cv_results_path, cv_payload)

    base_card = _build_model_card(
        task="classification",
        target_col=target_col,
        group_col=group_col,
        feature_cols=feature_cols,
        column_types=column_types,
        estimator=search.best_estimator_,
        cv_metrics=cv_metrics,
        n_samples=X.shape[0],
    )
    tuning_meta = {
        "n_iter": n_iter,
        "cv_splits": cv.get_n_splits(),
        "random_state": random_state,
    }
    model_card_path = _update_model_card(
        outdir,
        base_card=base_card,
        best_params=search.best_params_,
        cv_metrics=cv_metrics,
        tuning_meta=tuning_meta,
    )

    return {
        "best_params": search.best_params_,
        "cv": cv_metrics,
        "artifacts": {
            "model": model_path,
            "feature_cols": feature_cols_path,
            "target": target_path,
            "param_grid": param_grid_path,
            "cv_results": cv_results_path,
            "model_card": model_card_path,
        },
    }


def random_search_regressor(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    feature_cols: list[str],
    outdir: Path,
    n_iter: int = 40,
    cv_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Tune a regression pipeline with RandomizedSearchCV.

    Parameters
    ----------
    df
        Feature dataframe containing ``target_col``, ``group_col`` and ``feature_cols``.
    target_col
        Name of the target column to be predicted.
    group_col
        Column used to build ``GroupKFold`` splits.
    feature_cols
        List of feature columns available for the model.
    outdir
        Directory where the tuned artifacts will be persisted.
    n_iter
        Number of sampled hyperparameter combinations.
    cv_splits
        Desired number of ``GroupKFold`` splits (capped by the number of groups).
    random_state
        Seed controlling sampling of hyperparameters.

    Returns
    -------
    dict
        Dictionary with best parameters, cross-validation summary and artifact paths.
    """

    cleaned = _clean_training_frame(
        df,
        target_col=target_col,
        group_col=group_col,
        feature_cols=feature_cols,
    )
    X = cleaned[list(feature_cols)]
    y = cleaned[target_col]
    groups = cleaned[group_col]
    column_types = _infer_column_types(X)
    pipeline = make_regressor(column_types.numeric, column_types.categorical)

    param_space = {
        "model__n_estimators": list(range(100, 601)),
        "model__max_depth": [None, *range(5, 31)],
        "model__max_features": ["sqrt", "log2", None],
        "model__min_samples_split": list(range(2, 11)),
    }

    cv = _make_group_kfold(groups, cv_splits)
    scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_space,
        n_iter=n_iter,
        scoring=scoring,
        refit="mae",
        cv=cv,
        random_state=random_state,
        n_jobs=None,
        return_train_score=False,
    )
    search.fit(X, y, groups=groups)

    cv_results_df = pd.DataFrame(search.cv_results_)
    best_idx = int(search.best_index_)
    mae_mean = float(cv_results_df.loc[best_idx, "mean_test_mae"])
    mae_std = float(cv_results_df.loc[best_idx, "std_test_mae"])
    rmse_mean = float(cv_results_df.loc[best_idx, "mean_test_rmse"])
    rmse_std = float(cv_results_df.loc[best_idx, "std_test_rmse"])
    cv_metrics: dict[str, float] = {
        "n_splits": cv.get_n_splits(),
        "n_groups": int(pd.Index(groups).nunique()),
        "mae_mean": float(-mae_mean),
        "mae_std": float(abs(mae_std)),
        "rmse_mean": float(-rmse_mean),
        "rmse_std": float(abs(rmse_std)),
    }

    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "regressor_tuned.joblib"
    joblib.dump(search.best_estimator_, model_path)

    feature_cols_path = _update_feature_metadata(outdir, feature_cols)
    target_path = outdir / "target.json"
    _dump_json(target_path, {"target": target_col})
    param_grid_path = outdir / "param_grid.json"
    _dump_json(
        param_grid_path,
        {
            "distributions": {
                key: (list(value) if isinstance(value, range) else value)
                for key, value in param_space.items()
            },
            "n_iter": n_iter,
            "random_state": random_state,
            "cv_splits": cv.get_n_splits(),
        },
    )

    cv_results_path = outdir / "cv_results.json"
    cv_payload = {"records": json.loads(cv_results_df.to_json(orient="records"))}
    _dump_json(cv_results_path, cv_payload)

    base_card = _build_model_card(
        task="regression",
        target_col=target_col,
        group_col=group_col,
        feature_cols=feature_cols,
        column_types=column_types,
        estimator=search.best_estimator_,
        cv_metrics=cv_metrics,
        n_samples=X.shape[0],
    )
    tuning_meta = {
        "n_iter": n_iter,
        "cv_splits": cv.get_n_splits(),
        "random_state": random_state,
    }
    model_card_path = _update_model_card(
        outdir,
        base_card=base_card,
        best_params=search.best_params_,
        cv_metrics=cv_metrics,
        tuning_meta=tuning_meta,
    )

    return {
        "best_params": search.best_params_,
        "cv": cv_metrics,
        "artifacts": {
            "model": model_path,
            "feature_cols": feature_cols_path,
            "target": target_path,
            "param_grid": param_grid_path,
            "cv_results": cv_results_path,
            "model_card": model_card_path,
        },
    }


def compute_permutation_importance(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation feature importance scores.

    Parameters
    ----------
    model
        Fitted :class:`~sklearn.pipeline.Pipeline` used to evaluate permutations.
    X
        Feature matrix used for evaluation (must match the training schema).
    y
        Target values associated with ``X``.
    scoring
        Scoring function compatible with scikit-learn (higher is better).
    n_repeats
        Number of shuffles for each feature.
    random_state
        Seed controlling the permutation RNG.

    Returns
    -------
    pandas.DataFrame
        Dataframe with ``feature``, ``importance_mean`` and ``importance_std`` columns
        sorted by descending importance.
    """

    result = skl_permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=None,
    )
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
    else:  # pragma: no cover - defensive
        feature_names = [
            f"feature_{idx}" for idx in range(len(result.importances_mean))
        ]
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    importance_df = importance_df.sort_values("importance_mean", ascending=False)
    importance_df.reset_index(drop=True, inplace=True)
    return importance_df


def filter_features_by_importance(
    imp_df: pd.DataFrame,
    k_top: int | None = None,
    min_importance: float | None = None,
) -> list[str]:
    """Select features based on permutation importance scores.

    Parameters
    ----------
    imp_df
        Dataframe returned by :func:`compute_permutation_importance`.
    k_top
        Maximum number of features to keep (highest importances first).
    min_importance
        Minimum importance threshold required to keep a feature.

    Returns
    -------
    list[str]
        Sorted feature names passing the provided criteria.
    """

    if imp_df.empty:
        return []

    filtered = imp_df.copy()
    if min_importance is not None:
        filtered = filtered[filtered["importance_mean"] >= min_importance]
    filtered = filtered.sort_values("importance_mean", ascending=False)
    if k_top is not None:
        filtered = filtered.head(max(k_top, 0))
    return filtered["feature"].tolist()


def _dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_model_card(
    *,
    task: str,
    target_col: str,
    group_col: str,
    feature_cols: Sequence[str],
    column_types: _ColumnTypes,
    estimator: Pipeline,
    cv_metrics: dict[str, float],
    n_samples: int,
) -> dict:
    timestamp = datetime.now(timezone.utc).isoformat()
    model = estimator.named_steps["model"]
    return {
        "timestamp": timestamp,
        "task": task,
        "target": target_col,
        "group_column": group_col,
        "features": list(feature_cols),
        "dataset_info": {
            "n_samples": int(n_samples),
            "n_features": len(feature_cols),
            "n_groups": int(cv_metrics.get("n_groups", 0) or 0),
        },
        "preprocessing": {
            "numeric": column_types.numeric,
            "categorical": column_types.categorical,
        },
        "algorithm": {
            "name": model.__class__.__name__,
            "hyperparameters": {
                key: value for key, value in model.get_params().items()
            },
        },
        "cross_validation": cv_metrics,
    }


def train_classifier(
    df_features: pd.DataFrame,
    *,
    target_col: str,
    group_col: str,
    feature_cols: Sequence[str],
    outdir: Path,
) -> dict:
    """Train and persist a classification pipeline."""

    cleaned = _clean_training_frame(
        df_features,
        target_col=target_col,
        group_col=group_col,
        feature_cols=feature_cols,
    )
    X = cleaned[list(feature_cols)]
    y = cleaned[target_col]
    groups = cleaned[group_col]
    column_types = _infer_column_types(X)
    model = make_classifier(column_types.numeric, column_types.categorical)
    cv_metrics = grouped_cv_scores(model, X, y, groups, task="classification")
    model.fit(X, y)

    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "classifier.joblib"
    joblib.dump(model, model_path)

    feature_cols_path = outdir / "feature_cols.json"
    _dump_json(feature_cols_path, {"features": list(feature_cols)})

    target_path = outdir / "target.json"
    _dump_json(target_path, {"target": target_col})

    model_card = _build_model_card(
        task="classification",
        target_col=target_col,
        group_col=group_col,
        feature_cols=feature_cols,
        column_types=column_types,
        estimator=model,
        cv_metrics=cv_metrics,
        n_samples=X.shape[0],
    )
    model_card_path = outdir / "model_card.json"
    _dump_json(model_card_path, model_card)

    return {
        "artifacts": {
            "model": model_path,
            "feature_cols": feature_cols_path,
            "target": target_path,
            "model_card": model_card_path,
        },
        "cv": cv_metrics,
    }


def train_regressor(
    df_features: pd.DataFrame,
    *,
    target_col: str,
    group_col: str,
    feature_cols: Sequence[str],
    outdir: Path,
) -> dict:
    """Train and persist a regression pipeline."""

    cleaned = _clean_training_frame(
        df_features,
        target_col=target_col,
        group_col=group_col,
        feature_cols=feature_cols,
    )
    X = cleaned[list(feature_cols)]
    y = cleaned[target_col]
    groups = cleaned[group_col]
    column_types = _infer_column_types(X)
    model = make_regressor(column_types.numeric, column_types.categorical)
    cv_metrics = grouped_cv_scores(model, X, y, groups, task="regression")
    model.fit(X, y)

    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "regressor.joblib"
    joblib.dump(model, model_path)

    feature_cols_path = outdir / "feature_cols.json"
    _dump_json(feature_cols_path, {"features": list(feature_cols)})

    target_path = outdir / "target.json"
    _dump_json(target_path, {"target": target_col})

    model_card = _build_model_card(
        task="regression",
        target_col=target_col,
        group_col=group_col,
        feature_cols=feature_cols,
        column_types=column_types,
        estimator=model,
        cv_metrics=cv_metrics,
        n_samples=X.shape[0],
    )
    model_card_path = outdir / "model_card.json"
    _dump_json(model_card_path, model_card)

    return {
        "artifacts": {
            "model": model_path,
            "feature_cols": feature_cols_path,
            "target": target_path,
            "model_card": model_card_path,
        },
        "cv": cv_metrics,
    }


def _load_feature_columns(model_dir: Path) -> list[str]:
    feature_path = model_dir / "feature_cols.json"
    if not feature_path.exists():
        raise FileNotFoundError(f"Could not find feature_cols.json at {feature_path}")
    payload = json.loads(feature_path.read_text(encoding="utf-8"))
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError("feature_cols.json must contain a list under 'features'")
    return [str(item) for item in features]


def predict_from_artifact(model_path: Path, df_features: pd.DataFrame) -> pd.DataFrame:
    """Load a persisted model and generate predictions."""

    model = joblib.load(model_path)
    model_dir = model_path.parent
    feature_cols = _load_feature_columns(model_dir)
    _ensure_required_columns(df_features, ["sample_id", *feature_cols])
    subset = df_features[["sample_id", *feature_cols]].dropna(axis=0, how="any")
    if subset.empty:
        raise ValueError("No rows available for prediction after dropping NaNs")
    predictions = model.predict(subset[feature_cols])
    return pd.DataFrame(
        {
            "sample_id": subset["sample_id"].to_numpy(),
            "y_pred": predictions,
        }
    )


def kmeans_explore(
    df_features: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    k: int = 3,
) -> pd.DataFrame:
    """Cluster samples using KMeans on the selected features."""

    if k < 2:
        raise ValueError("k must be greater than or equal to 2")
    _ensure_required_columns(df_features, ["sample_id", *feature_cols])
    subset = df_features[["sample_id", *feature_cols]].dropna(axis=0, how="any")
    if subset.empty:
        raise ValueError("No rows available for clustering after dropping NaNs")
    X = subset[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = model.fit_predict(X_scaled)
    return pd.DataFrame(
        {"sample_id": subset["sample_id"].to_numpy(), "cluster": labels}
    )


__all__ = [
    "build_preprocessor",
    "grouped_cv_scores",
    "kmeans_explore",
    "make_classifier",
    "make_regressor",
    "predict_from_artifact",
    "train_classifier",
    "train_regressor",
]
