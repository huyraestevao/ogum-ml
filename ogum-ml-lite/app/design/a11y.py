"""Accessibility helpers for the Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FocusHint:
    """Structured representation of a focus hint message."""

    message: str

    def __str__(self) -> str:  # pragma: no cover - simple delegation
        return self.message


def focus_hint(message: str | None = None) -> FocusHint:
    """Return a keyboard navigation hint for captions/tooltips."""

    base = "Use Tab/Shift+Tab to navigate."
    text = message.strip() if message else base
    return FocusHint(text)


def aria_label(text: str) -> dict[str, str]:
    """Return kwargs that attach an accessible description to components."""

    if not text:
        return {"help": ""}
    return {"help": text}


def describe_chart(summary: str, *, prefix: str = "📈") -> str:
    """Return a concise textual description for charts."""

    text = summary.strip()
    if not text:
        text = "Chart summary unavailable"
    return f"{prefix} {text}".strip()
