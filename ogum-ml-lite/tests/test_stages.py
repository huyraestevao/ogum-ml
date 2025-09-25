from __future__ import annotations

import numpy as np
import pandas as pd
from ogum_lite.stages import DEFAULT_STAGES, split_by_stages, stage_masks


def test_stage_masks_split_ranges() -> None:
    y = np.array([0.5, 0.6, 0.72, 0.88, 0.93])
    masks = stage_masks(y, DEFAULT_STAGES)
    assert len(masks) == len(DEFAULT_STAGES)
    assert masks[0].sum() == 1  # only the 0.6 value
    assert masks[1].sum() == 2


def test_split_by_stages_preserves_labels() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["A"] * 5 + ["B"] * 5,
            "time_s": np.tile(np.arange(5.0), 2),
            "y": np.concatenate([np.linspace(0.5, 0.95, 5), np.linspace(0.52, 0.9, 5)]),
        }
    )

    splits = split_by_stages(df, y_col="y", stages=DEFAULT_STAGES)
    assert set(splits) == {"stage_1", "stage_2"}
    for stage_df in splits.values():
        if not stage_df.empty:
            assert "stage_label" in stage_df.columns
            assert set(stage_df["stage_label"]) <= {"stage_1", "stage_2"}
