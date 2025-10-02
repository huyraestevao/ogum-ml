"""Telemetry helpers with optional opt-in instrumentation."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping
from uuid import uuid4

try:  # pragma: no cover - streamlit not installed in some environments
    import streamlit as st
except Exception:  # pragma: no cover - fallback for tests
    st = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from ogum_lite.ui.workspace import Workspace
except Exception:  # pragma: no cover - tests without package import
    Workspace = Any  # type: ignore[misc]


_LAST_EVENT: ContextVar[dict[str, Any] | None] = ContextVar(
    "telemetry_last", default=None
)
SESSION_KEY = "telemetry_session_id"
ENABLED_KEY = "telemetry_enabled"
USER_HASH_KEY = "telemetry_user_hash"
LOG_FILENAME = "telemetry.jsonl"


def _env_opt_in() -> bool | None:
    value = os.getenv("OGUML_TELEMETRY")
    if value is None:
        return None
    return value not in {"0", "false", "False", "no", "off"}


def _session_state() -> MutableMapping[str, Any]:  # pragma: no cover - helper
    if st is None:
        raise RuntimeError("Streamlit session state not available")
    return st.session_state


def ensure_session_id() -> str:
    """Return a sticky UUID4 used to bucket telemetry sessions."""

    if st is None:
        return os.getenv("OGUML_SESSION_ID", str(uuid4()))
    state = _session_state()
    session_id = state.setdefault(SESSION_KEY, str(uuid4()))
    return str(session_id)


def set_user_hash(hash_value: str | None) -> None:
    """Persist the optional user hash in session state."""

    if st is None:
        return
    state = _session_state()
    if hash_value is None:
        state.pop(USER_HASH_KEY, None)
    else:
        state[USER_HASH_KEY] = hash_value


def _session_opt_in() -> bool | None:
    if st is None:
        return None
    state = _session_state()
    if ENABLED_KEY not in state:
        state[ENABLED_KEY] = False
    return bool(state.get(ENABLED_KEY))


def set_opt_in(enabled: bool) -> None:
    """Update opt-in flag for the current session."""

    if st is None:
        return
    _session_state()[ENABLED_KEY] = bool(enabled)


def is_enabled() -> bool:
    """Return whether telemetry should be recorded."""

    env_toggle = _env_opt_in()
    if env_toggle is not None:
        return env_toggle
    session_toggle = _session_opt_in()
    if session_toggle is not None:
        return session_toggle
    return False


def _resolve_workspace(workspace: Workspace | Path | str | None) -> Path:
    if isinstance(workspace, Workspace):  # pragma: no branch - runtime type check
        return workspace.resolve(LOG_FILENAME)
    if workspace is not None:
        path = Path(workspace)
        if path.is_dir():
            return (path / LOG_FILENAME).resolve()
        return path.resolve()
    if st is not None:
        try:  # pragma: no cover - defensive import
            from . import state

            return state.get_workspace().resolve(LOG_FILENAME)
        except Exception:  # pragma: no cover - fallback on failure
            pass
    default_dir = Path.cwd() / "workspace"
    default_dir.mkdir(parents=True, exist_ok=True)
    return (default_dir / LOG_FILENAME).resolve()


def _prepare_event(
    event: str, props: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session": ensure_session_id(),
        "event": event,
        "props": dict(props or {}),
    }
    if st is not None:
        state = _session_state()
        if USER_HASH_KEY in state:
            payload["user_hash"] = state[USER_HASH_KEY]
    return payload


def _write_event(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def log_event(
    event: str,
    props: Mapping[str, Any] | None = None,
    *,
    workspace: Workspace | Path | str | None = None,
) -> None:
    """Persist a telemetry event when enabled."""

    if not is_enabled():
        _LAST_EVENT.set(None)
        return

    payload = _prepare_event(event, props)
    path = _resolve_workspace(workspace)
    _write_event(path, payload)
    _LAST_EVENT.set(dict(payload))


def last_event() -> dict[str, Any] | None:
    """Return the last event recorded during this context."""

    return _LAST_EVENT.get()


def iter_events(path: Path | str) -> Iterable[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:  # pragma: no cover - defensive
                continue
    return events


def aggregate(path: Path | str) -> dict[str, Any]:
    """Aggregate counts and average duration per event/variant."""

    events = iter_events(path)
    summary: dict[str, Any] = {"events": {}, "experiments": {}}
    for event in events:
        name = event.get("event", "unknown")
        props = event.get("props", {}) or {}
        event_bucket = summary["events"].setdefault(
            name,
            {
                "count": 0,
                "total_duration_ms": 0.0,
                "duration_samples": 0,
                "variants": {},
            },
        )
        event_bucket["count"] += 1
        duration = props.get("duration_ms")
        if isinstance(duration, (int, float)):
            event_bucket["total_duration_ms"] += float(duration)
            event_bucket["duration_samples"] += 1
        variant = props.get("variant")
        experiment = props.get("experiment")
        if isinstance(variant, str):
            variant_bucket = event_bucket["variants"]
            variant_bucket[variant] = variant_bucket.get(variant, 0) + 1
        if isinstance(experiment, str) and isinstance(variant, str):
            exp_bucket = summary["experiments"].setdefault(experiment, {})
            exp_bucket[variant] = exp_bucket.get(variant, 0) + 1

    for bucket in summary["events"].values():
        samples = bucket.pop("duration_samples", 0) or 0
        total = bucket.pop("total_duration_ms", 0.0)
        bucket["avg_duration_ms"] = (total / samples) if samples else None
    return summary


@contextmanager
def telemetry_session(enabled: bool = True) -> Iterable[None]:
    """Context manager to temporarily override telemetry toggle (tests)."""

    previous_env = os.getenv("OGUML_TELEMETRY")
    os.environ["OGUML_TELEMETRY"] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous_env is None:
            os.environ.pop("OGUML_TELEMETRY", None)
        else:
            os.environ["OGUML_TELEMETRY"] = previous_env
        _LAST_EVENT.set(None)


__all__ = [
    "aggregate",
    "ensure_session_id",
    "is_enabled",
    "iter_events",
    "last_event",
    "log_event",
    "set_opt_in",
    "set_user_hash",
    "telemetry_session",
]
