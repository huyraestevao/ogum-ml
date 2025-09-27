"""Heatmap utilities for segmentation/Blaine metrics."""

from __future__ import annotations

from io import BytesIO
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _figure_to_png(fig: plt.Figure) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def _extract_metric_columns(
    df: pd.DataFrame,
    *,
    metric: str,
    group_col: str,
) -> pd.DataFrame:
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not present in dataframe")

    suffix = f"_{metric}"
    columns: list[str] = []
    for column in df.columns:
        if column.endswith(suffix):
            prefix = column[: -len(suffix)]
            columns.append(prefix)
    if not columns:
        return pd.DataFrame(index=df[group_col])

    working = df.set_index(group_col)
    data = {}
    for prefix in sorted(columns):
        source_column = f"{prefix}{suffix}"
        data[prefix] = pd.to_numeric(working[source_column], errors="coerce")
    matrix = pd.DataFrame(data)
    matrix.index.name = group_col
    return matrix


def prepare_segment_heatmap(
    df: pd.DataFrame,
    *,
    metric: str,
    group_col: str = "sample_id",
) -> pd.DataFrame:
    """Return a matrix (samples × segments) for the requested metric."""

    matrix = _extract_metric_columns(df, metric=metric, group_col=group_col)
    if matrix.empty:
        return matrix

    matrix = matrix.sort_index()
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    return matrix


def render_segment_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    cmap: str = "viridis",
    fmt: str = ".2f",
) -> bytes:
    """Render a heatmap from the provided matrix and return PNG bytes."""

    if matrix.empty:
        raise ValueError("matrix must contain at least one column")

    values = matrix.to_numpy(dtype=float)
    n_rows, n_cols = values.shape
    fig_width = max(4.0, n_cols * 1.2)
    fig_height = max(3.0, n_rows * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Segment")
    ax.set_ylabel(matrix.index.name or "sample")
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(matrix.index.astype(str))

    vmin = float(np.nanmin(values)) if not np.all(np.isnan(values)) else float("nan")
    vmax = float(np.nanmax(values)) if not np.all(np.isnan(values)) else float("nan")
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        im.set_clim(vmin=0.0, vmax=1.0)
    else:
        if vmin == vmax:
            span = 1e-6 if vmin == 0 else abs(vmin) * 1e-6
            im.set_clim(vmin=vmin - span, vmax=vmax + span)
        else:
            im.set_clim(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(metric_label_from_title(title))

    denom = vmax - vmin if np.isfinite(vmin) and np.isfinite(vmax) else None
    for i, j in product(range(n_rows), range(n_cols)):
        value = values[i, j]
        if np.isnan(value):
            continue
        if denom and denom != 0:
            normalized = (value - vmin) / denom
            color = "white" if normalized > 0.5 else "black"
        else:
            color = "black"
        ax.text(j, i, format(value, fmt), ha="center", va="center", color=color)

    fig.tight_layout()
    return _figure_to_png(fig)


def metric_label_from_title(title: str) -> str:
    lowered = title.lower()
    if "mse" in lowered:
        return "MSE"
    if "blaine" in lowered:
        return "Blaine n"
    if "n " in lowered or lowered.endswith(" n"):
        return "n"
    return "Value"


def save_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
    cmap: str = "viridis",
    fmt: str = ".2f",
) -> Path:
    """Persist a heatmap to disk and return the file path."""

    png_bytes = render_segment_heatmap(matrix, title=title, cmap=cmap, fmt=fmt)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(png_bytes)
    return out_path


__all__ = [
    "metric_label_from_title",
    "prepare_segment_heatmap",
    "render_segment_heatmap",
    "save_heatmap",
]
