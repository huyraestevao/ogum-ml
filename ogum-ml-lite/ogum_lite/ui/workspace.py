"""Workspace utilities used by the interactive applications."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass
class Workspace:
    """Represent a sandboxed directory tree for a UI session."""

    root: Path
    log_name: str = "run_log.jsonl"
    _root: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        self._root = root

    @property
    def path(self) -> Path:
        """Return the root path for the workspace."""

        return self._root

    def resolve(self, path_like: str | Path) -> Path:
        """Resolve ``path_like`` within the workspace boundary."""

        candidate = (self._root / Path(path_like)).expanduser().resolve()
        if not candidate.is_relative_to(self._root):  # pragma: no cover - safety
            raise ValueError(f"Path {candidate} escapes workspace root {self._root}")
        return candidate

    def ensure_dirs(self, *paths: str | Path) -> None:
        """Ensure that the provided directories exist."""

        for item in paths:
            directory = self.resolve(item)
            directory.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: str, payload: dict | None = None) -> None:
        """Append an event to the JSONL provenance log."""

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload or {},
        }
        log_path = self.resolve(self.log_name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def zip_outputs(self, outdir: Path, zip_path: Path) -> Path:
        """Zip the contents of ``outdir`` into ``zip_path``."""

        source = self.resolve(outdir)
        target = self.resolve(zip_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(source.rglob("*")):
                if file.is_file():
                    zf.write(file, arcname=file.relative_to(source))
        return target

    def iter_artifacts(self, patterns: Iterable[str] | None = None) -> list[Path]:
        """Return a list of artifacts stored in the workspace.

        Parameters
        ----------
        patterns
            Optional glob patterns to filter the artifacts.
        """

        patterns = list(patterns or ["**/*"])
        results: list[Path] = []
        for pattern in patterns:
            results.extend(sorted(self._root.glob(pattern)))
        return [path for path in results if path.is_file()]
