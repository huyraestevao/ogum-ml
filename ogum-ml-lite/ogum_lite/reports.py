"""HTML reporting helpers for Ogum ML Lite."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def _figure_to_png_bytes(fig) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[Any]
) -> bytes:
    """Render a confusion matrix plot and return PNG bytes."""

    label_list = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=label_list)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    tick_labels = [str(item) for item in label_list]
    ax.set_xticks(range(len(label_list)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(label_list)))
    ax.set_yticklabels(tick_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _figure_to_png_bytes(fig)


def plot_regression_scatter(y_true: np.ndarray, y_pred: np.ndarray) -> bytes:
    """Plot predictions versus true values and return PNG bytes."""

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true, y_pred, alpha=0.7, s=30)
    limits = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(limits, limits, linestyle="--", linewidth=1.5)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs True")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _figure_to_png_bytes(fig)


def plot_feature_importance(imp_df: pd.DataFrame) -> bytes:
    """Generate a bar chart with permutation importances."""

    sorted_df = imp_df.sort_values("importance_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(
        sorted_df["feature"],
        sorted_df["importance_mean"],
        xerr=sorted_df["importance_std"],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Importance (mean Δ score)")
    ax.set_title("Permutation feature importance")
    fig.tight_layout()
    return _figure_to_png_bytes(fig)


def _dict_to_html_list(data: dict[str, Any]) -> str:
    items: list[str] = []
    for key, value in data.items():
        items.append(f"<li><strong>{key}</strong>: {value}</li>")
    return "\n".join(items)


def _table_to_html(table: Any) -> str:
    if isinstance(table, pd.DataFrame):
        return table.to_html(index=False, classes="table", border=0)
    if isinstance(table, list):
        if table and isinstance(table[0], dict):
            return pd.DataFrame(table).to_html(index=False, classes="table", border=0)
        return pd.DataFrame(table).to_html(index=False, classes="table", border=0)
    if isinstance(table, dict):
        return pd.DataFrame([table]).to_html(index=False, classes="table", border=0)
    return ""


def render_html_report(outdir: Path, context: dict, figures: dict[str, bytes]) -> Path:
    """Generate an HTML report embedding PNG figures as base64 images."""

    outdir.mkdir(parents=True, exist_ok=True)
    figure_entries: list[str] = []
    for name, data in figures.items():
        filename = outdir / f"{name}.png"
        filename.write_bytes(data)
        encoded = base64.b64encode(data).decode("ascii")
        figure_entries.append(
            f"<figure><img alt='{name}' src='data:image/png;base64,{encoded}' />"
            f"<figcaption>{name.replace('_', ' ').title()}</figcaption></figure>"
        )

    title = context.get("title", "Ogum ML Report")
    task = context.get("task", "unknown")
    metrics = context.get("metrics", {})
    cv_metrics = context.get("cv_metrics", {})
    best_params = context.get("best_params", {})
    dataset_info = context.get("dataset", {})
    features = context.get("features", [])
    timestamp = context.get("timestamp", "")
    observations = context.get("observations", "")
    segments_table = context.get("segments_table")
    segments_summary = context.get("segments_summary", {})

    segments_summary_html = ""
    if isinstance(segments_summary, dict):
        segments_summary_html = "<ul>" + _dict_to_html_list(segments_summary) + "</ul>"
    elif isinstance(segments_summary, list):
        items = "".join(
            f"<li>{item}</li>" for item in segments_summary
        )
        segments_summary_html = f"<ul>{items}</ul>"

    segments_table_html = _table_to_html(segments_table)
    segments_section = ""
    if segments_summary_html or segments_table_html:
        table_html = segments_table_html or "<p>No segmentation table provided.</p>"
        segments_section = f"""
        <section>
          <h2>Segmentation</h2>
          {segments_summary_html}
          {table_html}
        </section>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <title>{title}</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 2rem; }}
          header {{ margin-bottom: 2rem; }}
          section {{ margin-bottom: 1.5rem; }}
          figure {{ margin: 1rem 0; }}
          img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ccc;
            padding: 0.5rem;
          }}
          ul {{ list-style: square; }}
          code {{
            background-color: #f5f5f5;
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
          }}
          table {{
            border-collapse: collapse;
            margin: 1rem 0;
            width: 100%;
          }}
          th, td {{
            border: 1px solid #ddd;
            padding: 0.5rem;
            text-align: left;
          }}
        </style>
      </head>
      <body>
        <header>
          <h1>{title}</h1>
          <p>
            <strong>Task:</strong> {task} |
            <strong>Generated at:</strong> {timestamp}
          </p>
        </header>
        <section>
          <h2>Dataset</h2>
          <ul>
            {_dict_to_html_list(dataset_info)}
          </ul>
        </section>
        <section>
          <h2>Evaluation Metrics</h2>
          <ul>
            {_dict_to_html_list(metrics)}
          </ul>
        </section>
        <section>
          <h2>Cross-validation (tuning)</h2>
          <ul>
            {_dict_to_html_list(cv_metrics)}
          </ul>
        </section>
        <section>
          <h2>Best Hyperparameters</h2>
          <pre>{best_params}</pre>
        </section>
        <section>
          <h2>Features</h2>
          <p>{', '.join(features)}</p>
        </section>
        <section>
          <h2>Observations</h2>
          <p>{observations}</p>
        </section>
        {segments_section}
        <section>
          <h2>Figures</h2>
          {''.join(figure_entries)}
        </section>
      </body>
    </html>
    """

    report_path = outdir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path
