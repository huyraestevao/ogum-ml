"""Helpers to segment densification curves into canonical sintering stages."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_STAGES: tuple[tuple[float, float], ...] = ((0.55, 0.70), (0.70, 0.90))


def stage_masks(
    y: np.ndarray, stages: Iterable[tuple[float, float]]
) -> list[np.ndarray]:
    """Build boolean masks selecting samples belonging to each stage."""

    y = np.asarray(y, dtype=float)
    masks: list[np.ndarray] = []
    for lower, upper in stages:
        mask = (y >= float(lower)) & (y < float(upper))
        masks.append(mask)
    return masks


def split_by_stages(
    df: pd.DataFrame,
    *,
    y_col: str = "y",
    group_col: str = "sample_id",
    stages: Iterable[tuple[float, float]] = DEFAULT_STAGES,
) -> dict[str, pd.DataFrame]:
    """Segment a long-format dataframe according to densification stages."""

    if y_col not in df.columns:
        raise KeyError(f"Column '{y_col}' not present in dataframe")

    grouped = df.groupby(group_col) if group_col in df.columns else [(None, df)]
    stage_dict: dict[str, list[pd.DataFrame]] = {
        f"stage_{idx+1}": [] for idx, _ in enumerate(stages)
    }

    for _, group in grouped:
        if group.empty:
            continue
        y_values = group[y_col].to_numpy(dtype=float)
        masks = stage_masks(y_values, stages)
        for idx, mask in enumerate(masks):
            if not np.any(mask):
                continue
            stage_df = group.loc[mask].copy()
            stage_df["stage_label"] = f"stage_{idx+1}"
            stage_dict[f"stage_{idx+1}"].append(stage_df)

    columns = list(df.columns)
    if "stage_label" not in columns:
        columns = [*columns, "stage_label"]

    return {
        label: (
            pd.concat(parts, ignore_index=True)
            if parts
            else pd.DataFrame(columns=columns)
        )
        for label, parts in stage_dict.items()
    }


__all__ = ["DEFAULT_STAGES", "stage_masks", "split_by_stages"]
