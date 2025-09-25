from __future__ import annotations

import pandas as pd
import pytest
from ogum_lite.io_mapping import (
    TECHNIQUE_CHOICES,
    apply_mapping,
    infer_mapping,
    read_table,
)


def test_infer_mapping_detects_units_and_aliases() -> None:
    df = pd.DataFrame(
        {
            "Sample": ["A", "B"],
            "tempo_min": [1.5, 2.0],
            "temp_K": [973.15, 983.15],
            "densidade_relativa": [0.5, 0.6],
            "composition": ["Fe", "Fe"],
            "technique": [TECHNIQUE_CHOICES[0], TECHNIQUE_CHOICES[0]],
        }
    )

    mapping = infer_mapping(df)
    assert mapping.sample_id.lower().startswith("sample")
    assert mapping.time_unit == "min"
    assert mapping.temp_unit == "K"

    mapped = apply_mapping(df, mapping)
    assert mapped["time_s"].iloc[0] == pytest.approx(90.0)
    assert mapped["temp_C"].iloc[0] == pytest.approx(700.0)
    assert "composition" in mapped.columns
    assert set(mapped["technique"]) == {TECHNIQUE_CHOICES[0]}


def test_read_table_supports_csv_and_excel(tmp_path) -> None:
    pytest.importorskip("openpyxl")
    dataframe = pd.DataFrame(
        {
            "sample_id": [1, 2],
            "time_s": [0.0, 1.0],
            "temp_C": [25.0, 30.0],
        }
    )

    io_dir = tmp_path / "io"
    io_dir.mkdir()
    csv_path = io_dir / "data.csv"
    dataframe.to_csv(csv_path, index=False)
    loaded_csv = read_table(csv_path)
    assert list(loaded_csv.columns) == list(dataframe.columns)

    xlsx_path = io_dir / "data.xlsx"
    dataframe.to_excel(xlsx_path, index=False)
    loaded_xlsx = read_table(xlsx_path)
    assert list(loaded_xlsx.columns) == list(dataframe.columns)
