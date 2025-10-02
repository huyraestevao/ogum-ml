from collections import Counter

import pytest
from app.services import ab


def test_assign_variant_sticky():
    variants = ["wizard", "tabs"]
    session_id = "session-123"
    first = ab.assign_variant(session_id, "wizard_vs_tabs", variants)
    second = ab.assign_variant(session_id, "wizard_vs_tabs", variants)
    assert first == second


def test_assign_variant_probability_extremes():
    variants = ["top", "bottom"]
    assert ab.assign_variant("sess", "buttons", variants, p=[1.0, 0.0]) == "top"
    assert ab.assign_variant("sess", "buttons", variants, p=[0.0, 1.0]) == "bottom"


def test_variant_distribution_balanced():
    variants = ["compact", "expanded"]
    assignments = Counter(
        ab.assign_variant(f"session-{idx}", "msc_controls_layout", variants)
        for idx in range(200)
    )
    assert sum(assignments.values()) == 200
    difference = abs(assignments[variants[0]] - assignments[variants[1]])
    assert difference < 80  # tolerate skew from hashing but ensure both used


def test_ab_flag_without_streamlit(monkeypatch):
    monkeypatch.setattr(ab, "st", None)
    assert ab.ab_flag("msc_controls_layout", "compact", variants=["compact"]) is True
    with pytest.raises(ValueError):
        ab.ab_flag("msc_controls_layout", "compact")
