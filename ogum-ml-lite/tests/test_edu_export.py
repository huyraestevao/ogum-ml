from pathlib import Path

import plotly.graph_objects as go
from app.edu import exporter


def _context() -> dict:
    return {
        "title": "Edu",
        "concepts": [{"title": "Theta", "body": "**demo**", "formula": "x"}],
        "simulations": [{"title": "Sim", "body": "text", "figure_key": "fig1"}],
        "exercises": [{"title": "Ex", "statement": "desc", "answer": {"score": 1.0}}],
    }


def test_export_html_generates_file(tmp_path: Path) -> None:
    fig = go.Figure(data=go.Scatter(x=[0, 1], y=[0, 1]))
    figures = {"fig1": exporter.capture_fig(fig)}
    target = tmp_path / "edu.html"
    path = exporter.export_html(_context(), figures, target)
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Conceitos" in content


def test_export_pdf_optional(tmp_path: Path) -> None:
    target = tmp_path / "edu.pdf"
    result = exporter.export_pdf(_context(), {}, target)
    if result is None:
        assert not target.exists()
    else:
        assert target.exists()
