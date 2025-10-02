"""Reusable UI components for the Educational Mode."""

from __future__ import annotations

from typing import Literal

import streamlit as st


def card_conceito(title: str, md_text: str, formula_latex: str | None = None) -> None:
    """Render a concept card with Markdown explanation and optional formula.

    Parameters
    ----------
    title:
        Card title displayed in bold.
    md_text:
        Markdown string with the conceptual explanation in pt/en.
    formula_latex:
        Optional LaTeX expression rendered below the body using Streamlit's MathJax.
    """

    with st.container(border=True):
        st.markdown(f"### {title}")
        st.markdown(md_text, unsafe_allow_html=False)
        if formula_latex:
            st.markdown("---")
            st.latex(formula_latex)


def callout(kind: Literal["info", "warn", "tip"], text: str) -> None:
    """Render a lightweight callout for contextual hints.

    Parameters
    ----------
    kind:
        Type of callout. ``"info"`` uses an informational tone, ``"warn"`` warns the
        learner and ``"tip"`` highlights practical suggestions.
    text:
        Markdown message rendered in the callout body.
    """

    renderers = {
        "info": st.info,
        "warn": st.warning,
        "tip": st.success,
    }
    renderer = renderers.get(kind, st.info)
    renderer(text)


def figure_plotly(fig, caption: str) -> None:
    """Display a Plotly figure with a caption.

    Parameters
    ----------
    fig:
        Plotly ``Figure`` instance already configured by the simulator.
    caption:
        Text explaining what the learner should observe in the chart.
    """

    st.plotly_chart(fig, use_container_width=True)
    st.caption(caption)


def formula_block(latex: str) -> None:
    """Render a LaTeX formula using Streamlit's MathJax support.

    Parameters
    ----------
    latex:
        LaTeX expression displayed as a standalone block.
    """

    st.latex(latex)
