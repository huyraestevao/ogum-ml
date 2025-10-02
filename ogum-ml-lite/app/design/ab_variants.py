"""Definitions for A/B experiments exposed to the UI."""

from __future__ import annotations

EXPERIMENTS: dict[str, list[str]] = {
    "msc_controls_layout": ["compact", "expanded"],
    "wizard_vs_tabs": ["wizard", "tabs"],
    "run_buttons_position": ["top", "bottom"],
}


__all__ = ["EXPERIMENTS"]
