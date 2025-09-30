"""FastAPI service exposing Ogum Lite pipelines."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Sequence

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ogum_lite.features import build_feature_store
from ogum_lite.mechanism import detect_mechanism_change
from ogum_lite.ml_hooks import train_classifier, train_regressor
from ogum_lite.preprocess import SmoothMethod, derive_all
from ogum_lite.segmentation import segment_dataframe
from ogum_lite.stages import DEFAULT_STAGES
from ogum_lite.theta_msc import MasterCurveResult, score_activation_energies

app = FastAPI(title="Ogum ML Lite API", version="0.1.0")


def _records_to_dataframe(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    dataframe = pd.DataFrame(records or [])
    if dataframe.empty:
        raise HTTPException(
            status_code=400, detail="data must contain at least one record"
        )
    return dataframe


def _parse_stages(
    stages: Sequence[Sequence[float]] | None,
) -> list[tuple[float, float]] | None:
    if stages is None:
        return None
    parsed: list[tuple[float, float]] = []
    for item in stages:
        if len(item) != 2:
            raise HTTPException(
                status_code=400, detail="Stage definitions must contain two values"
            )
        lower, upper = float(item[0]), float(item[1])
        parsed.append((lower, upper))
    return parsed


def _encode_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _result_to_payload(result: MasterCurveResult) -> dict[str, Any]:
    segment_metrics = {
        f"{lower:.2f}-{upper:.2f}": value
        for (lower, upper), value in result.segment_mse.items()
    }
    return {
        "activation_energy": result.activation_energy,
        "mse_global": result.mse_global,
        "mse_segmented": result.mse_segmented,
        "segment_mse": segment_metrics,
        "segments": [[float(lower), float(upper)] for lower, upper in result.segments],
    }


def _load_model(encoded: str) -> Any:
    try:
        buffer = BytesIO(base64.b64decode(encoded))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=400, detail="Invalid base64 model payload"
        ) from exc
    buffer.seek(0)
    try:
        return joblib.load(buffer)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=400, detail="Unable to load serialized model"
        ) from exc


class PrepRequest(BaseModel):
    data: List[Dict[str, Any]]
    time_column: str = Field("time_s", alias="time_column")
    temperature_column: str = Field("temp_C", alias="temperature_column")
    y_column: str = Field("rho_rel", alias="y_column")
    smooth: SmoothMethod = "savgol"
    window: int = 11
    poly: int = 3
    moving_k: int = Field(5, alias="moving_k")


class FeaturesRequest(PrepRequest):
    group_col: str = Field("sample_id", alias="group_col")
    stages: Sequence[Sequence[float]] | None = None
    theta_ea: Sequence[float] | None = Field(None, alias="theta_ea")
    segment_method: str = "fixed"
    segment_thresholds: Sequence[float] | None = None
    segment_n_segments: int = 3
    segment_min_size: int = 5


class MSCRequest(BaseModel):
    data: List[Dict[str, Any]]
    ea: Sequence[float]
    group_col: str = "sample_id"
    time_column: str = "time_s"
    temperature_column: str = "temp_C"
    y_column: str = "rho_rel"
    normalize_theta: str = "minmax"
    metric: str = "segmented"
    segments: Sequence[Sequence[float]] | None = None


class SegmentationRequest(BaseModel):
    data: List[Dict[str, Any]]
    group_col: str = "sample_id"
    time_column: str = "time_s"
    y_column: str = "rho_rel"
    method: str = "fixed"
    thresholds: Sequence[float] | None = None
    n_segments: int = 3
    min_size: int = 5


class MechanismRequest(BaseModel):
    data: List[Dict[str, Any]]
    group_col: str = "sample_id"
    theta_column: str = "theta"
    y_column: str = "densification"
    max_segments: int = 2
    min_size: int = 5
    criterion: str = "bic"
    threshold: float = 2.0
    slope_delta: float = 0.02


class TrainRequest(BaseModel):
    data: List[Dict[str, Any]]
    target: str
    group_col: str
    task: str = Field("classification", pattern="^(classification|regression)$")
    feature_cols: Sequence[str] | None = None


class PredictRequest(BaseModel):
    model: str
    data: List[Dict[str, Any]]
    feature_cols: Sequence[str]
    return_proba: bool = False


@app.post("/prep")
def preprocess(request: PrepRequest) -> dict[str, Any]:
    df = _records_to_dataframe(request.data)
    derived = derive_all(
        df,
        t_col=request.time_column,
        T_col=request.temperature_column,
        y_col=request.y_column,
        smooth=request.smooth,
        window=request.window,
        poly=request.poly,
        moving_k=request.moving_k,
    )
    return {"rows": derived.to_dict(orient="records")}


@app.post("/features")
def feature_store(request: FeaturesRequest) -> dict[str, Any]:
    df = _records_to_dataframe(request.data)
    stage_ranges = _parse_stages(request.stages)
    thresholds = (
        tuple(float(value) for value in request.segment_thresholds)
        if request.segment_thresholds is not None
        else None
    )
    kwargs: dict[str, Any] = {
        "group_col": request.group_col,
        "t_col": request.time_column,
        "T_col": request.temperature_column,
        "y_col": request.y_column,
        "smooth": request.smooth,
        "window": request.window,
        "poly": request.poly,
        "moving_k": request.moving_k,
        "theta_ea_kj": request.theta_ea,
        "segment_method": request.segment_method,
        "segment_thresholds": thresholds,
        "segment_n_segments": request.segment_n_segments,
        "segment_min_size": request.segment_min_size,
    }
    if stage_ranges is not None:
        kwargs["stages"] = stage_ranges
    else:
        kwargs["stages"] = DEFAULT_STAGES
    features = build_feature_store(df, **kwargs)
    return {"rows": features.to_dict(orient="records")}


@app.post("/msc")
def msc(request: MSCRequest) -> dict[str, Any]:
    df = _records_to_dataframe(request.data)
    normalize = None if request.normalize_theta == "none" else request.normalize_theta
    stages = _parse_stages(request.segments)
    kwargs: dict[str, Any] = {
        "metric": request.metric,
        "group_col": request.group_col,
        "t_col": request.time_column,
        "temp_col": request.temperature_column,
        "y_col": request.y_column,
        "normalize_theta": normalize,
    }
    if stages is not None:
        kwargs["segments"] = stages
    summary, best_result, results = score_activation_energies(
        df,
        request.ea,
        **kwargs,
    )
    payload = {
        "summary": summary.to_dict(orient="records"),
        "best": _result_to_payload(best_result),
        "results": [_result_to_payload(item) for item in results],
    }
    return payload


@app.post("/segmentation")
def segmentation(request: SegmentationRequest) -> dict[str, Any]:
    df = _records_to_dataframe(request.data)
    segments = segment_dataframe(
        df,
        group_col=request.group_col,
        t_col=request.time_column,
        y_col=request.y_column,
        method=request.method,
        thresholds=(
            tuple(request.thresholds)
            if request.thresholds is not None
            else (0.55, 0.70, 0.90)
        ),
        n_segments=request.n_segments,
        min_size=request.min_size,
    )
    rows = [segment.to_dict() for segment in segments]
    return {"segments": rows}


@app.post("/mechanism")
def mechanism(request: MechanismRequest) -> dict[str, Any]:
    df = _records_to_dataframe(request.data)
    result = detect_mechanism_change(
        df,
        group_col=request.group_col,
        theta_col=request.theta_column,
        y_col=request.y_column,
        max_segments=request.max_segments,
        min_size=request.min_size,
        criterion=request.criterion,
        threshold=request.threshold,
        slope_delta=request.slope_delta,
    )
    return {"rows": result.to_dict(orient="records")}


@app.post("/ml/train")
def ml_train(request: TrainRequest) -> dict[str, Any]:
    df = _records_to_dataframe(request.data)
    feature_cols = (
        list(request.feature_cols)
        if request.feature_cols
        else [
            col for col in df.columns if col not in {request.target, request.group_col}
        ]
    )
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise HTTPException(
            status_code=400, detail=f"Missing feature columns: {missing_cols}"
        )

    with TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        if request.task == "classification":
            result = train_classifier(
                df,
                target_col=request.target,
                group_col=request.group_col,
                feature_cols=feature_cols,
                outdir=outdir,
            )
        else:
            result = train_regressor(
                df,
                target_col=request.target,
                group_col=request.group_col,
                feature_cols=feature_cols,
                outdir=outdir,
            )

        artifacts = result.get("artifacts", {})
        model_path = Path(artifacts.get("model", outdir / "model.joblib"))
        feature_cols_path = Path(
            artifacts.get("feature_cols", outdir / "feature_cols.json")
        )
        target_path = Path(artifacts.get("target", outdir / "target.json"))
        model_card_path = Path(artifacts.get("model_card", outdir / "model_card.json"))

        if not model_path.exists():
            raise HTTPException(status_code=500, detail="Model artifact not generated")

        payload = {
            "model": _encode_file(model_path),
            "feature_cols": _load_json(feature_cols_path).get("features", feature_cols),
            "target": _load_json(target_path).get("target", request.target),
            "cv": result.get("cv", {}),
            "model_card": _load_json(model_card_path),
        }
    return payload


@app.post("/ml/predict")
def ml_predict(request: PredictRequest) -> dict[str, Any]:
    model = _load_model(request.model)
    df = _records_to_dataframe(request.data)
    missing = [col for col in request.feature_cols if col not in df.columns]
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise HTTPException(
            status_code=400, detail=f"Missing feature columns: {missing_cols}"
        )

    X = df[list(request.feature_cols)]
    predictions = model.predict(X)
    response: dict[str, Any] = {"predictions": predictions.tolist()}

    if request.return_proba and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        response["probabilities"] = probabilities.tolist()

    return response


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok"}
