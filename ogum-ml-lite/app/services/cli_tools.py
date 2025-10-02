"""Utility command line interface for cache, telemetry, and A/B helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import cache, telemetry


def _write_output(path: Path | None, payload: Any) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    if path is None:
        print(data)
    else:
        Path(path).write_text(data, encoding="utf-8")


def _telemetry_aggregate(args: argparse.Namespace) -> None:
    summary = telemetry.aggregate(args.file)
    _write_output(Path(args.out) if args.out else None, summary)


def _telemetry_clean(args: argparse.Namespace) -> None:
    target = Path(args.file)
    if target.exists():
        target.unlink()


def _cache_stats(args: argparse.Namespace) -> None:
    summary = cache.cache_stats(args.dir)
    _write_output(Path(args.out) if args.out else None, summary)


def _cache_purge(args: argparse.Namespace) -> None:
    cache.cache_purge(args.dir)


def _ab_export(args: argparse.Namespace) -> None:
    summary = telemetry.aggregate(args.file)
    payload = summary.get("experiments", {})
    _write_output(Path(args.out) if args.out else None, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ogum-ML service utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    telemetry_parser = subparsers.add_parser("telemetry", help="Telemetry utilities")
    telemetry_sub = telemetry_parser.add_subparsers(dest="telemetry_cmd", required=True)

    agg = telemetry_sub.add_parser("aggregate", help="Aggregate telemetry JSONL")
    agg.add_argument("--file", required=True, help="Telemetry JSONL path")
    agg.add_argument("--out", help="Optional output JSON path")
    agg.set_defaults(func=_telemetry_aggregate)

    clean = telemetry_sub.add_parser("clean", help="Remove telemetry log")
    clean.add_argument("--file", required=True, help="Telemetry JSONL path")
    clean.set_defaults(func=_telemetry_clean)

    cache_parser = subparsers.add_parser("cache", help="Cache utilities")
    cache_sub = cache_parser.add_subparsers(dest="cache_cmd", required=True)

    stats = cache_sub.add_parser("stats", help="Inspect cache usage")
    stats.add_argument("--dir", required=True, help="Cache directory")
    stats.add_argument("--out", help="Optional output JSON path")
    stats.set_defaults(func=_cache_stats)

    purge = cache_sub.add_parser("purge", help="Purge cache contents")
    purge.add_argument("--dir", required=True, help="Cache directory")
    purge.set_defaults(func=_cache_purge)

    ab_parser = subparsers.add_parser("ab", help="A/B experiment helpers")
    ab_sub = ab_parser.add_subparsers(dest="ab_cmd", required=True)

    export = ab_sub.add_parser("export", help="Export experiment tallies")
    export.add_argument("--file", required=True, help="Telemetry JSONL path")
    export.add_argument("--out", help="Optional output JSON path")
    export.set_defaults(func=_ab_export)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "func", None)
    if handler is None:
        parser.error("Missing command handler")
    handler(args)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()
