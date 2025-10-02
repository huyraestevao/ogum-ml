"""File-system backed caching helpers for expensive operations."""

from __future__ import annotations

import hashlib
import json
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

try:  # pragma: no cover - optional dependency
    from ogum_lite.ui.workspace import Workspace
except Exception:  # pragma: no cover - fallback for isolated tests
    Workspace = Any  # type: ignore[misc]

from . import profiling

_LAST_HIT: ContextVar[bool | None] = ContextVar("cache_last_hit", default=None)


def last_cache_hit() -> bool | None:
    return _LAST_HIT.get()


def reset_last_cache() -> None:
    _LAST_HIT.set(None)


def _resolve_workspace(workspace: Workspace | Path | str) -> Path:
    if isinstance(workspace, Workspace):  # pragma: no branch - runtime type check
        return workspace.resolve(".cache")
    return Path(workspace).expanduser().resolve()


def _hash_file(path: Path) -> str:
    resolved = Path(path)
    if not resolved.exists():
        return hashlib.sha256(f"missing:{resolved}".encode("utf-8")).hexdigest()
    sha = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _hash_inputs(inputs: Iterable[Any]) -> str:
    sha = hashlib.sha256()
    for item in inputs:
        if isinstance(item, Path):
            sha.update(b"PATH")
            sha.update(_hash_file(item).encode("utf-8"))
        else:
            sha.update(b"ARG")
            sha.update(str(item).encode("utf-8"))
    return sha.hexdigest()


def _hash_params(params: Mapping[str, Any] | None) -> str:
    payload = json.dumps(params or {}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _evaluate(arg: Any, *args: Any, **kwargs: Any) -> Any:
    return arg(*args, **kwargs) if callable(arg) else arg


def _serialise(result: Any) -> Any:
    if isinstance(result, Path):
        return str(result)
    if isinstance(result, Mapping):
        return {key: _serialise(value) for key, value in result.items()}
    if isinstance(result, (list, tuple, set)):
        return [_serialise(value) for value in result]
    return result


def _cache_entry(
    directory: Path,
    task_name: str,
    inputs_hash: str,
    params_hash: str,
) -> Path:
    return directory / task_name / f"{inputs_hash}-{params_hash}.json"


def cache_result(
    task_name: str,
    inputs: Iterable[Any] | Callable[..., Iterable[Any]],
    params: Mapping[str, Any] | Callable[..., Mapping[str, Any]] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that caches the callable output on disk."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            reset_last_cache()
            workspace = kwargs.get("workspace")
            if workspace is None:
                for value in args:
                    if isinstance(value, Workspace):
                        workspace = value
                        break
            if workspace is None:
                raise ValueError("cache_result requires a workspace argument")

            cache_dir = _resolve_workspace(workspace)
            cache_dir.mkdir(parents=True, exist_ok=True)

            resolved_inputs = list(_evaluate(inputs, *args, **kwargs))
            resolved_params = _evaluate(params, *args, **kwargs) if params else {}
            inputs_hash = _hash_inputs(resolved_inputs)
            params_hash = _hash_params(resolved_params)
            entry = _cache_entry(cache_dir, task_name, inputs_hash, params_hash)
            entry.parent.mkdir(parents=True, exist_ok=True)

            if entry.exists():
                data = json.loads(entry.read_text(encoding="utf-8"))
                _LAST_HIT.set(True)
                profiling.set_last_profile(data.get("profile"))
                data["hits"] = int(data.get("hits", 0)) + 1
                data["last_used"] = datetime.now(timezone.utc).isoformat()
                serialized = json.dumps(data, ensure_ascii=False, indent=2)
                entry.write_text(serialized, encoding="utf-8")
                return data.get("result")

            _LAST_HIT.set(False)
            result = func(*args, **kwargs)
            profile = profiling.get_last_profile()
            payload = {
                "result": _serialise(result),
                "hits": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_used": datetime.now(timezone.utc).isoformat(),
            }
            if profile is not None:
                payload["profile"] = profile
            serialized = json.dumps(payload, ensure_ascii=False, indent=2)
            entry.write_text(serialized, encoding="utf-8")
            return result

        return wrapper

    return decorator


def cache_stats(directory: Path | str) -> dict[str, Any]:
    directory = Path(directory)
    summary = {
        "entries": 0,
        "size_bytes": 0,
        "tasks": {},
    }
    if not directory.exists():
        return summary
    for path in directory.rglob("*.json"):
        if not path.is_file():
            continue
        summary["entries"] += 1
        summary["size_bytes"] += path.stat().st_size
        task = path.parent.name
        payload = json.loads(path.read_text(encoding="utf-8"))
        bucket = summary["tasks"].setdefault(
            task,
            {"count": 0, "hits": 0},
        )
        bucket["count"] += 1
        bucket["hits"] += int(payload.get("hits", 0))
    return summary


def cache_purge(directory: Path | str) -> None:
    directory = Path(directory)
    if not directory.exists():
        return
    for path in sorted(directory.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:  # pragma: no cover - directory not empty
                continue


__all__ = [
    "cache_purge",
    "cache_result",
    "cache_stats",
    "last_cache_hit",
    "reset_last_cache",
]
