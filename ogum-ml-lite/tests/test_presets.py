from pathlib import Path

from ogum_lite.ui.presets import load_presets, merge_presets, save_presets


def test_presets_roundtrip(tmp_path: Path) -> None:
    preset = {
        "columns": {"group_col": "sample_id", "t_col": "time_s"},
        "msc": {"ea_kj": [200, 300]},
    }
    path = tmp_path / "preset.yaml"
    save_presets(preset, path)

    loaded = load_presets(path)
    assert loaded == preset

    override = {"msc": {"metric": "global"}, "columns": {"temp_col": "temp_C"}}
    merged = merge_presets(preset, override)

    assert merged["msc"]["ea_kj"] == [200, 300]
    assert merged["msc"]["metric"] == "global"
    assert merged["columns"]["temp_col"] == "temp_C"
    # Ensure original preset remains unchanged
    assert "metric" not in preset["msc"]
