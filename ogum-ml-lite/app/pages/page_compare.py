"""Streamlit page for comparing Ogum-ML runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from ogum_lite.compare.cli_compare import cmd_compare_matrix, cmd_compare_runs

from ..services import state


def _render_download(path: Path, label: str) -> None:
    if path.exists():
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=path.name,
        )


def render(_) -> None:
    """Render the comparison utilities page."""

    st.subheader("Compare Runs")
    workspace = state.get_workspace()
    default_dir = Path(workspace.path) if workspace else Path.cwd()

    mode = st.selectbox("Mode", ("Runs", "Matrix"))

    if mode == "Runs":
        col_a, col_b = st.columns(2)
        with col_a:
            path_a = st.text_input("Run A", value=str(default_dir))
        with col_b:
            path_b = st.text_input("Run B", value=str(default_dir))
        outdir = Path(
            st.text_input("Output directory", value=str(default_dir / "compare_runs"))
        )
        if st.button("Compare"):
            with st.spinner("Comparing runs..."):
                cmd_compare_runs(
                    argparse.Namespace(a=path_a, b=path_b, outdir=str(outdir))
                )
            st.success("Comparison finished")
        summary_path = outdir / "compare_summary.json"
        if summary_path.exists():
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_rows = data.get("diffs", {}).get("summary", {}).get("kpis", [])
            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows))
            html_path = Path(data.get("html", outdir / "report.html"))
            xlsx_path = Path(data.get("xlsx", outdir / "report.xlsx"))
            _render_download(html_path, f"HTML · {html_path.name}")
            _render_download(xlsx_path, f"XLSX · {xlsx_path.name}")
            if html_path.exists():
                components.v1.html(
                    html_path.read_text(encoding="utf-8"),
                    height=400,
                    scrolling=True,
                )
    else:
        path_ref = st.text_input("Reference run", value=str(default_dir))
        candidates_raw = st.text_area(
            "Candidates (one per line)", value=str(default_dir)
        )
        candidates = [
            line.strip() for line in candidates_raw.splitlines() if line.strip()
        ]
        outdir = Path(
            st.text_input("Output directory", value=str(default_dir / "compare_matrix"))
        )
        if st.button("Build matrix"):
            with st.spinner("Building matrix..."):
                cmd_compare_matrix(
                    argparse.Namespace(
                        ref=path_ref, candidates=candidates, outdir=str(outdir)
                    )
                )
            st.success("Matrix ready")
        ranking_path = outdir / "ranking.csv"
        if ranking_path.exists():
            ranking = pd.read_csv(ranking_path)
            st.dataframe(ranking)
            _render_download(outdir / "matrix.html", "HTML · matrix.html")


__all__ = ["render"]
