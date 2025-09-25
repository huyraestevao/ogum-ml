from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ogum_lite.cli import main


def _run_cli(args: list[str]) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        main(args)
    return buffer.getvalue()


def _make_raw_dataframe() -> pd.DataFrame:
    time_min = np.linspace(0.0, 6.0, 48)
    temp_K = 900.0 + 60.0 * (time_min / time_min.max()) + 30.0
    densification = 0.50 + 0.45 * (time_min / time_min.max()) ** 1.2

    first = pd.DataFrame(
        {
            "Sample": "A",
            "tempo_min": time_min,
            "Temp_K": temp_K,
            "rho_rel": densification,
            "composition": "Fe2O3",
            "technique": "SPS",
        }
    )

    second = pd.DataFrame(
        {
            "Sample": "B",
            "tempo_min": time_min,
            "Temp_K": temp_K + 20.0,
            "rho_rel": densification * 0.98,
            "composition": "Fe2O3",
            "technique": "SPS",
        }
    )

    combined = pd.concat([first, second], ignore_index=True)
    return combined


def test_cli_pipeline_end_to_end(tmp_path: Path) -> None:
    raw_df = _make_raw_dataframe()
    raw_path = tmp_path / "raw.csv"
    raw_df.to_csv(raw_path, index=False)

    mapping_path = tmp_path / "mapping.json"
    _run_cli(["io", "map", "--input", str(raw_path), "--out", str(mapping_path)])
    mapping_data = json.loads(mapping_path.read_text())
    assert mapping_data["time_unit"] == "min"
    assert mapping_data["temp_unit"] == "K"

    derived_path = tmp_path / "derivatives.csv"
    _run_cli(
        [
            "preprocess",
            "derive",
            "--input",
            str(raw_path),
            "--map",
            str(mapping_path),
            "--smooth",
            "none",
            "--out",
            str(derived_path),
        ]
    )
    derived_df = pd.read_csv(derived_path)
    assert {"dy_dt", "dT_dt", "dp_dT"}.issubset(derived_df.columns)
    assert np.all(np.isfinite(derived_df["time_s"]))

    stages_arg = "0.55-0.70,0.70-0.90"
    arrhenius_path = tmp_path / "arrhenius.csv"
    _run_cli(
        [
            "arrhenius",
            "fit",
            "--input",
            str(derived_path),
            "--out",
            str(arrhenius_path),
            "--stages",
            stages_arg,
        ]
    )
    arrhenius_df = pd.read_csv(arrhenius_path)
    assert "Ea_arr_global_kJ" in arrhenius_df.columns
    assert arrhenius_df["Ea_arr_global_kJ"].notna().all()

    features_path = tmp_path / "feature_store.csv"
    _run_cli(
        [
            "features",
            "build",
            "--input",
            str(derived_path),
            "--output",
            str(features_path),
            "--stages",
            stages_arg,
            "--smooth",
            "none",
        ]
    )
    feature_store = pd.read_csv(features_path)
    expected_columns = {
        "heating_rate_med_C_per_s",
        "heating_rate_med_C_per_s_s1",
        "Ea_arr_global_kJ",
        "Ea_arr_55_70_kJ",
        "Ea_arr_70_90_kJ",
    }
    assert expected_columns.issubset(feature_store.columns)
    assert feature_store["Ea_arr_global_kJ"].notna().all()
