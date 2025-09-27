"""Export helpers for reports and interoperable artifacts."""

from __future__ import annotations

import importlib
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

try:  # pragma: no cover - imported lazily for images
    from openpyxl.drawing.image import Image as XLImage
except Exception:  # pragma: no cover - defensive fallback when openpyxl missing
    XLImage = None  # type: ignore[assignment]


def _flatten_context(context: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def _append(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                _append(f"{prefix}.{sub_key}" if prefix else str(sub_key), sub_value)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                _append(f"{prefix}[{index}]", item)
        else:
            rows.append({"item": prefix, "value": value})

    for key, value in context.items():
        _append(str(key), value)

    if not rows:
        return pd.DataFrame(columns=["item", "value"])
    return pd.DataFrame(rows)


def _ensure_dataframe(payload: Any) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        return payload
    if payload is None:
        return pd.DataFrame()
    if isinstance(payload, dict):
        return pd.DataFrame(sorted(payload.items()), columns=["metric", "value"])
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    return pd.DataFrame([{"value": payload}])


def export_xlsx(
    out_path: Path,
    *,
    context: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    images: dict[str, bytes] | None = None,
) -> Path:
    """Export an Excel report consolidating tables and optional figures."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = _flatten_context(context)

    def _fetch_table(*keys: str) -> pd.DataFrame | None:
        for key in keys:
            value = tables.get(key)
            if value is not None:
                return value
        return None

    msc_df = _ensure_dataframe(_fetch_table("MSC", "msc"))
    features_df = _ensure_dataframe(_fetch_table("Features", "features"))
    metrics_df = _ensure_dataframe(_fetch_table("Metrics", "metrics"))

    segments_sheet = None

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        msc_df.to_excel(writer, sheet_name="MSC", index=False)
        features_df.to_excel(writer, sheet_name="Features", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

        segments_payload = _fetch_table("Segments", "segments", "Segmentation")
        if segments_payload is not None:
            segments_df = _ensure_dataframe(segments_payload)
            segments_df.to_excel(writer, sheet_name="Segments", index=False)
            segments_sheet = "Segments"

        if images:
            workbook = writer.book
            mapping = {
                "msc.png": ("MSC", "H2"),
                "confusion.png": ("Metrics", "H2"),
                "scatter.png": ("Metrics", "H18"),
                "segments.png": ("Segments", "H2"),
            }
            for key, (sheet, anchor) in mapping.items():
                data = images.get(key)
                if not data or XLImage is None:
                    continue
                if sheet not in workbook.sheetnames:
                    if sheet == "Segments" and segments_sheet is None:
                        empty_df = pd.DataFrame()
                        empty_df.to_excel(writer, sheet_name=sheet, index=False)
                        workbook = writer.book
                    else:
                        continue
                worksheet = workbook[sheet]
                image = XLImage(BytesIO(data))
                worksheet.add_image(image, anchor)

    return out_path


def export_onnx(
    sklearn_model: Any,
    feature_names: list[str],
    out_path: Path,
) -> Path | None:
    """Export a RandomForest model to ONNX if optional deps are available."""

    estimator = sklearn_model
    if hasattr(estimator, "named_steps"):
        estimator = (
            estimator.named_steps.get("model")
            or list(estimator.named_steps.values())[-1]
        )

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if not isinstance(estimator, (RandomForestClassifier, RandomForestRegressor)):
        return None

    try:
        skl2onnx = importlib.import_module("skl2onnx")
        data_types = importlib.import_module("skl2onnx.common.data_types")
        convert_func = getattr(skl2onnx, "convert_sklearn")
        tensor_type = getattr(data_types, "FloatTensorType")
    except ImportError:
        try:
            onnxmltools = importlib.import_module("onnxmltools")
            convert_func = getattr(onnxmltools, "convert_sklearn")
            data_types = importlib.import_module(
                "onnxmltools.convert.common.data_types"
            )
            tensor_type = getattr(data_types, "FloatTensorType")
        except ImportError:
            return None

    if not feature_names:
        return None

    initial_types = [("input", tensor_type([None, len(feature_names)]))]
    onnx_model = convert_func(estimator, initial_types=initial_types)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        handle.write(onnx_model.SerializeToString())
    return out_path
