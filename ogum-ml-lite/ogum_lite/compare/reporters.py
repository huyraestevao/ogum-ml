"""Report generation helpers for the comparison module."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def render_html_compare(
    outdir: Path,
    meta: dict[str, Any],
    diffs: dict[str, Any],
    images: dict[str, bytes] | None = None,
) -> Path:
    """Render a standalone HTML comparison report."""

    _ensure_dir(outdir)
    html_path = Path(outdir) / "report.html"

    runs = meta.get("runs", []) if isinstance(meta, dict) else []
    title = "Ogum-ML · Run comparison"
    if runs:
        title += " — " + " vs ".join(map(str, runs))
    sections: list[str] = [f"<h1>{title}</h1>"]
    summary_rows = diffs.get("summary", {}).get("kpis", [])
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        sections.append("<h2>Summary</h2>")
        sections.append(summary_df.to_html(index=False, escape=False))
    alerts = diffs.get("summary", {}).get("alerts", [])
    if alerts:
        sections.append("<h3>Alerts</h3>")
        items = "".join(f"<li>{alert}</li>" for alert in alerts)
        sections.append(f"<ul>{items}</ul>")

    for name in ("presets", "msc", "segments", "mechanism", "ml"):
        payload = diffs.get(name)
        sections.append(f"<h2>{name.title()}</h2>")
        if payload:
            sections.append(
                f"<pre>{json.dumps(payload, indent=2, ensure_ascii=False)}</pre>"
            )
        else:
            sections.append("<p>No data.</p>")

    if images:
        for label, content in images.items():
            b64 = base64.b64encode(content).decode("ascii")
            tag = (
                f"<h3>{label}</h3>"
                f"<img alt='{label}' src='data:image/png;base64,{b64}' />"
            )
            sections.append(tag)

    html = "\n".join(sections)
    html_path.write_text(html, encoding="utf-8")
    return html_path


def export_xlsx_compare(
    out_path: Path,
    *,
    summary: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    images: dict[str, bytes] | None = None,
) -> Path:
    """Export comparison results to an XLSX workbook."""

    out_path = Path(out_path)
    _ensure_dir(out_path.parent)

    def _write_workbook(writer: pd.ExcelWriter) -> None:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        for name, table in tables.items():
            sheet = f"Diff-{name[:25]}"
            table.to_excel(writer, sheet_name=sheet, index=False)
        if images:
            workbook = writer.book
            for idx, (label, payload) in enumerate(images.items()):
                sheet_name = f"Img{idx+1}"
                worksheet = workbook.add_worksheet(sheet_name)
                writer.sheets[sheet_name] = worksheet
                worksheet.write(0, 0, label)
                image_data = io.BytesIO(payload)
                worksheet.insert_image(1, 0, f"{label}.png", {"image_data": image_data})

    try:
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            _write_workbook(writer)
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        with pd.ExcelWriter(out_path) as writer:
            _write_workbook(writer)
    return out_path


__all__ = ["render_html_compare", "export_xlsx_compare"]
