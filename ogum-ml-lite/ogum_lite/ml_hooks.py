"""Integration points for future ML pipelines."""

from __future__ import annotations

from typing import Any, Dict


def register_pipeline(name: str, config: Dict[str, Any]) -> None:
    """Register a ML pipeline definition (stub)."""

    # TODO: plug into ogumsoftware registry when available.
    raise NotImplementedError("Pipeline registry not implemented yet.")


def load_pipeline(name: str) -> Any:
    """Load a ML pipeline (stub)."""

    # TODO: integrate with joblib or MLflow models.
    raise NotImplementedError("Loading ML pipelines is not implemented yet.")


__all__ = ["load_pipeline", "register_pipeline"]
