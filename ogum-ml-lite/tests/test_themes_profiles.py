from pathlib import Path

from app.design.theme import get_theme
from app.services import profiles, themes_loader


def test_merge_custom_theme():
    base = themes_loader.load_theme_yaml(Path("app/config/themes/base.yaml"))
    override = themes_loader.load_theme_yaml(
        Path("app/config/themes/custom.example.yaml")
    )
    merged = themes_loader.merge_theme(base, override)
    assert merged["colors"]["primary"] == override["colors"]["primary"]
    assert merged["colors"]["background"] == base["colors"]["background"]


def test_get_theme_with_override():
    override = themes_loader.load_theme_yaml(
        Path("app/config/themes/custom.example.yaml")
    )
    themed = get_theme(dark=False, override=override)
    assert themed["colors"]["primary"] == override["colors"]["primary"]
    dark_theme = get_theme(dark=True)
    assert dark_theme["colors"]["background"] != themed["colors"]["background"]


def test_apply_profile_merges_fields():
    profile = profiles.load_profile("fs_uhs")
    base_preset = {
        "msc": {"ea_kj": [100], "metric": "segmented"},
        "features": {"include": ["global"]},
        "segmentation": {"bounds": [55, 70, 90]},
    }
    applied = profiles.apply_profile(base_preset, profile)
    assert applied["msc"]["ea_kj"] == profile["msc"]["ea_kj"]
    assert "electric" in applied["features"]["include"]
    assert applied["segmentation"]["bounds"] == profile["segmentation"]["bounds"]
