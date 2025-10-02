"""A/B experiment assignment helpers."""

from __future__ import annotations

import hashlib
from typing import Sequence

try:  # pragma: no cover - optional dependency for runtime
    import streamlit as st
except Exception:  # pragma: no cover - tests may run without streamlit
    st = None  # type: ignore[assignment]

from .telemetry import ensure_session_id

ASSIGNMENTS_KEY = "ab_assignments"


def _normalise_variants(variants: Sequence[str]) -> list[str]:
    options = [str(option) for option in variants if option]
    if not options:
        raise ValueError("Variants list must not be empty")
    return options


def _hash_to_float(seed: str) -> float:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    as_int = int(digest[:16], 16)
    return as_int / float(16**16)


def assign_variant(
    session_id: str,
    experiment: str,
    variants: Sequence[str],
    p: Sequence[float] | None = None,
) -> str:
    """Deterministically assign a variant based on session hash."""

    options = _normalise_variants(variants)
    if p is not None:
        if len(p) != len(options):
            raise ValueError("Probability vector length must match variants")
        total = float(sum(p))
        if total <= 0:
            raise ValueError("Probability vector must have positive sum")
        cumulative: list[float] = []
        acc = 0.0
        for value in p:
            acc += float(value) / total
            cumulative.append(acc)
        bucket = _hash_to_float(f"{experiment}:{session_id}")
        for idx, threshold in enumerate(cumulative):
            if bucket <= threshold:
                return options[idx]
        return options[-1]
    digest = hashlib.sha256(f"{experiment}:{session_id}".encode("utf-8")).hexdigest()
    index = int(digest, 16)
    return options[index % len(options)]


def _session_assignments() -> dict[str, str]:  # pragma: no cover - helper
    if st is None:
        return {}
    assignments = st.session_state.setdefault(ASSIGNMENTS_KEY, {})
    if not isinstance(assignments, dict):
        assignments = {}
        st.session_state[ASSIGNMENTS_KEY] = assignments
    return assignments


def current_variant(
    experiment: str,
    variants: Sequence[str],
    *,
    probabilities: Sequence[float] | None = None,
    session_id: str | None = None,
) -> str:
    """Return the sticky variant for the current session."""

    sid = session_id or ensure_session_id()
    choice = assign_variant(sid, experiment, variants, probabilities)
    if st is not None:
        assignments = _session_assignments()
        assignments.setdefault(experiment, choice)
        st.session_state[ASSIGNMENTS_KEY] = assignments
    return choice


def ab_flag(
    experiment: str,
    variant: str,
    *,
    variants: Sequence[str] | None = None,
) -> bool:
    """Return ``True`` if the active variant for ``experiment`` matches ``variant``."""

    if variants is None:
        if st is not None:
            assignments = _session_assignments()
            active = assignments.get(experiment)
            if active is None:
                return False
            return active == variant
        raise ValueError("Variants must be provided when Streamlit is unavailable")
    choice = current_variant(experiment, variants)
    return choice == variant


def export_assignments() -> dict[str, str]:
    """Return current session assignments for debugging or telemetry."""

    if st is None:
        return {}
    return dict(_session_assignments())


__all__ = [
    "ab_flag",
    "assign_variant",
    "current_variant",
    "export_assignments",
]
