"""Simple link checker for local markdown references."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILES = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "DESIGN_SPEC_UX.md",
)
_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _resolve(base: Path, target: str) -> Path:
    clean = target.split("#", 1)[0].split("?", 1)[0]
    if not clean:
        return base
    if clean.startswith("/"):
        return (REPO_ROOT / clean.lstrip("/")).resolve()
    path = Path(clean)
    return (base.parent / path).resolve()


def find_broken_links(files: Iterable[Path]) -> list[tuple[Path, str]]:
    missing: list[tuple[Path, str]] = []
    for file_path in files:
        if not file_path.exists():
            continue
        text = file_path.read_text(encoding="utf-8")
        for match in _LINK_RE.finditer(text):
            target = match.group(1)
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            resolved = _resolve(file_path, target)
            if not resolved.exists():
                missing.append((file_path, target))
    return missing


def main(paths: Sequence[str] | None = None) -> int:
    files = [Path(p) for p in paths] if paths else list(DEFAULT_FILES)
    broken = find_broken_links(files)
    if not broken:
        print("links: ok")
        return 0
    for origin, target in broken:
        print(f"{origin.relative_to(REPO_ROOT)} -> {target} (missing)")
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main(sys.argv[1:]))
