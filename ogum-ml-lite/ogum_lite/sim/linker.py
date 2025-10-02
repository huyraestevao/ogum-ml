"""Helpers to link simulation features with experimental runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def _load_yaml_mapping(path: Path | None) -> dict[str, str]:
    if path is None or not Path(path).exists():
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):  # pragma: no cover - defensive
        raise ValueError("link.yaml must contain a mapping of sample_id to sim_id")
    return {str(key): str(value) for key, value in data.items()}


def _heuristic_match(sample_id: str, candidates: set[str]) -> str | None:
    if sample_id in candidates:
        return sample_id
    stem = Path(sample_id).stem
    if stem in candidates:
        return stem
    return None


def link_runs(
    exp_features_csv: Path,
    sim_features_csv: Path,
    link_yaml: Path | None = None,
) -> pd.DataFrame:
    """Join experimental features with simulation features."""

    exp_df = pd.read_csv(exp_features_csv)
    if "sample_id" not in exp_df.columns:
        raise ValueError("Experimental features must contain a 'sample_id' column")

    sim_df = pd.read_csv(sim_features_csv)
    if "sim_id" not in sim_df.columns:
        raise ValueError("Simulation features must contain a 'sim_id' column")

    mapping = _load_yaml_mapping(link_yaml)
    candidates = set(sim_df["sim_id"].astype(str))

    if "sim_id" in exp_df.columns:
        resolved = exp_df["sim_id"].astype(str)
        resolved = resolved.where(exp_df["sim_id"].notna() & (resolved != ""))
    else:
        resolved = pd.Series([None] * len(exp_df), index=exp_df.index, dtype="object")

    sample_ids = exp_df["sample_id"].astype(str)
    mapped = sample_ids.map(mapping)
    resolved = resolved.where(resolved.notna(), mapped)
    heuristics = sample_ids.map(lambda sid: _heuristic_match(sid, candidates))
    resolved = resolved.where(resolved.notna(), heuristics)

    if resolved.isna().any():
        missing = sorted(sample_ids[resolved.isna()].unique())
        raise ValueError(f"Missing sim_id for samples: {', '.join(missing)}")

    exp_df = exp_df.copy()
    exp_df["sim_id"] = resolved.astype(str)

    rename_map = {col: f"{col}_sim" for col in sim_df.columns if col != "sim_id"}
    sim_df = sim_df.rename(columns=rename_map)

    merged = exp_df.merge(sim_df, on="sim_id", how="left")
    sim_columns = list(rename_map.values())
    missing_features = merged[sim_columns].isna().all(axis=1)
    if missing_features.any():  # pragma: no cover - defensive
        missing_sim_ids = ", ".join(
            sorted(merged.loc[missing_features, "sim_id"].unique())
        )
        raise ValueError(
            f"Simulation features missing for sim_id(s): {missing_sim_ids}"
        )

    return merged


__all__ = ["link_runs"]
