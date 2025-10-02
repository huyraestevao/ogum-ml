"""Light-weight profiling helpers to record runtime metrics."""

from __future__ import annotations

import os
import time
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Mapping

try:  # pragma: no cover - optional dependency
    import psutil
except Exception:  # pragma: no cover - fallback when psutil unavailable
    psutil = None  # type: ignore[assignment]

try:  # pragma: no cover - workspace optional during tests
    from ogum_lite.ui.workspace import Workspace
except Exception:  # pragma: no cover - fallback for isolated tests
    Workspace = Any  # type: ignore[misc]

from . import telemetry

_LAST_PROFILE: ContextVar[dict[str, Any] | None] = ContextVar(
    "profile_last", default=None
)


def _memory_usage_mb() -> float | None:
    if psutil is None:
        return None
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def _find_workspace(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Workspace | None:
    for value in kwargs.values():
        if hasattr(value, "resolve") and hasattr(value, "log_event"):
            return value  # type: ignore[return-value]
    for value in args:
        if hasattr(value, "resolve") and hasattr(value, "log_event"):
            return value  # type: ignore[return-value]
    return None


def _record(workspace: Workspace | None, name: str, payload: Mapping[str, Any]) -> None:
    if workspace is not None:
        try:
            workspace.log_event(f"profile.{name}", dict(payload))
        except Exception:  # pragma: no cover - logging best effort
            pass
    telemetry.log_event(f"profile.{name}", payload, workspace=workspace)


def set_last_profile(data: dict[str, Any] | None) -> None:
    _LAST_PROFILE.set(data)


def get_last_profile() -> dict[str, Any] | None:
    return _LAST_PROFILE.get()


def profile_step(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator capturing elapsed time and memory deltas."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            workspace = _find_workspace(args, kwargs)
            start_time = time.perf_counter()
            start_mem = _memory_usage_mb()
            try:
                return func(*args, **kwargs)
            finally:
                end_time = time.perf_counter()
                end_mem = _memory_usage_mb()
                duration_ms = (end_time - start_time) * 1000
                memory_delta = None
                if start_mem is not None and end_mem is not None:
                    memory_delta = end_mem - start_mem
                payload = {
                    "step": name,
                    "duration_ms": duration_ms,
                    "memory_mb": memory_delta,
                }
                _record(workspace, name, payload)
                set_last_profile(payload)

        return wrapper

    return decorator


__all__ = ["get_last_profile", "profile_step", "set_last_profile"]
