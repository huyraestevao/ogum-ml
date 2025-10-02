"""Utilities for comparing Ogum-ML runs."""

from .diff_core import (
    compose_diff_summary,
    diff_mechanism,
    diff_ml,
    diff_msc,
    diff_presets,
    diff_segments,
)
from .loaders import find_run_root, scan_run
from .reporters import export_xlsx_compare, render_html_compare

__all__ = [
    "find_run_root",
    "scan_run",
    "diff_presets",
    "diff_msc",
    "diff_segments",
    "diff_mechanism",
    "diff_ml",
    "compose_diff_summary",
    "render_html_compare",
    "export_xlsx_compare",
]
