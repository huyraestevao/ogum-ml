"""Export utilities for the Educational Mode."""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import Any, Mapping

import markdown2

try:  # pragma: no cover - optional dependency
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    canvas = None
    A4 = None
    cm = None
    ImageReader = None


def capture_fig(fig: Any) -> bytes:
    """Capture a Plotly figure as PNG if possible, otherwise as HTML bytes."""

    try:
        return fig.to_image(format="png")
    except Exception:  # pragma: no cover - depends on kaleido availability
        html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        return html.encode("utf-8")


def _markdown_to_html(md_text: str) -> str:
    return markdown2.markdown(md_text, extras=["fenced-code-blocks", "tables"])


def export_html(
    context: Mapping[str, Any], figures: Mapping[str, bytes], out_path: Path
) -> Path:
    """Generate a static HTML report for the educational flow."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    title = context.get("title", "Ogum-ML — Modo Educacional")
    concepts = context.get("concepts", [])
    simulations = context.get("simulations", [])
    exercises = context.get("exercises", [])

    body: list[str] = [f"<h1>{title}</h1>"]

    if concepts:
        body.append("<section><h2>Conceitos / Concepts</h2>")
        for item in concepts:
            body.append(f"<h3>{item.get('title')}</h3>")
            body.append(_markdown_to_html(item.get("body", "")))
            formula = item.get("formula")
            if formula:
                body.append(f"<p><em>{formula}</em></p>")
        body.append("</section>")

    if simulations:
        body.append("<section><h2>Simulações</h2>")
        for item in simulations:
            body.append(f"<h3>{item.get('title')}</h3>")
            body.append(_markdown_to_html(item.get("body", "")))
            fig_key = item.get("figure_key")
            if fig_key and fig_key in figures:
                body.append(_embed_figure(figures[fig_key]))
        body.append("</section>")

    if exercises:
        body.append("<section><h2>Exercícios / Exercises</h2>")
        for ex in exercises:
            body.append(f"<h3>{ex.get('title')}</h3>")
            body.append(_markdown_to_html(ex.get("statement", "")))
            answer = ex.get("answer")
            if answer:
                body.append(f"<p><strong>Resposta:</strong> {answer}</p>")
        body.append("</section>")

    html = (
        "<html><head><meta charset='utf-8'><title>{title}</title></head><body>"
        f"{''.join(body)}</body></html>"
    )
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _embed_figure(data: bytes) -> str:
    if data.startswith(b"<"):
        return data.decode("utf-8")
    encoded = base64.b64encode(data).decode("ascii")
    return (
        "<figure><img src='data:image/png;base64,"
        f"{encoded}' alt='Figura' style='max-width:100%'>"
        "<figcaption></figcaption></figure>"
    )


def export_pdf(
    context: Mapping[str, Any], figures: Mapping[str, bytes], out_path: Path
) -> Path | None:
    """Generate a lightweight PDF report if ReportLab is available."""

    if canvas is None or A4 is None:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    title = context.get("title", "Ogum-ML — Modo Educacional")
    c = canvas.Canvas(str(out_path), pagesize=A4)
    width, height = A4
    text = c.beginText(2 * cm, height - 2 * cm)
    text.setFont("Helvetica-Bold", 16)
    text.textLine(title)
    text.moveCursor(0, 12)
    text.setFont("Helvetica", 11)

    def _write_section(header: str, items: list[dict]) -> None:
        nonlocal text
        if not items:
            return
        text.setFont("Helvetica-Bold", 13)
        text.textLine(header)
        text.setFont("Helvetica", 11)
        for item in items:
            _write_wrapped(text, item.get("title", ""), bullet=True)
            statement = item.get("statement") or item.get("body") or ""
            if statement:
                _write_wrapped(text, _html_to_text(_markdown_to_html(statement)))
            answer = item.get("answer")
            if answer:
                _write_wrapped(text, f"Resposta: {answer}")
        text.moveCursor(0, 12)

    concepts = context.get("concepts", [])
    simulations = context.get("simulations", [])
    exercises = context.get("exercises", [])

    _write_section("Conceitos", concepts)
    _write_section("Simulações", simulations)
    _write_section("Exercícios", exercises)
    c.drawText(text)

    y_cursor = text.getY() - 20
    for key, data in figures.items():
        image = _to_image(data)
        if image is None:
            continue
        img_width, img_height = image.getSize()
        scale = min((width - 4 * cm) / img_width, 10 * cm / img_height)
        if y_cursor - img_height * scale < 2 * cm:
            c.showPage()
            y_cursor = height - 2 * cm
        c.drawImage(
            image,
            2 * cm,
            y_cursor - img_height * scale,
            width=img_width * scale,
            height=img_height * scale,
        )
        y_cursor -= img_height * scale + 20

    c.showPage()
    c.save()
    return out_path


def _write_wrapped(text_obj, message: str, bullet: bool = False) -> None:
    if not message:
        return
    max_width = 85
    words = message.split()
    line = "• " if bullet else ""
    while words:
        word = words.pop(0)
        if len(line + word) > max_width:
            text_obj.textLine(line.rstrip())
            line = "  " if bullet else ""
        line += word + " "
    if line.strip():
        text_obj.textLine(line.rstrip())


def _html_to_text(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html)


def _to_image(data: bytes):  # pragma: no cover - trivial glue
    if ImageReader is None:
        return None
    if data.startswith(b"<"):
        return None
    return ImageReader(io.BytesIO(data))
