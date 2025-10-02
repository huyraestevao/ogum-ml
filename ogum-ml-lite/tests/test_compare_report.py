"""Report generation smoke tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from ogum_lite.compare import reporters


def test_render_html_compare(tmp_path: Path) -> None:
    html_path = reporters.render_html_compare(
        tmp_path,
        meta={"runs": ["a", "b"]},
        diffs={
            "summary": {"kpis": [{"metric": "mse", "a": 1, "b": 0.5}]},
            "presets": {"changed": {"foo": {"a": 1, "b": 2}}},
        },
    )
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")
    assert "Ogum-ML" in content


def test_export_xlsx_compare(tmp_path: Path) -> None:
    out = tmp_path / "report.xlsx"
    summary = pd.DataFrame([{"metric": "mse", "a": 1, "b": 0.5}])
    tables = {"Presets": pd.DataFrame([{"key": "foo", "a": 1, "b": 2}])}
    path = reporters.export_xlsx_compare(out, summary=summary, tables=tables)
    assert path.exists()
