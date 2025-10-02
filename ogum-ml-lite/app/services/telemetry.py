"""Telemetry helpers for dashboard interactions."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Mapping

from ogum_lite.ui.workspace import Workspace


def _enabled() -> bool:
    return os.getenv("OGUML_TELEMETRY", "1") not in {"0", "false", "False"}


def log_event(
    workspace: Workspace, event: str, payload: Mapping[str, Any] | None = None
) -> None:
    """Append a telemetry event to ``run_log.jsonl`` when enabled."""

    if not _enabled():
        return

    data = {
        "event": event,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    if payload:
        data.update(dict(payload))
    log_path = workspace.resolve("run_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")
