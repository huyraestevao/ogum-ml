"""Utility to compare translation catalogues and spot missing keys."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping

LOCALES_DIR = Path(__file__).resolve().parents[1] / "i18n" / "locales"
DEFAULT_LOCALES = ("pt", "en")


def _load_locale(locale: str) -> Mapping[str, object]:
    path = LOCALES_DIR / f"{locale}.json"
    if not path.exists():
        raise FileNotFoundError(f"Locale file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten(mapping: Mapping[str, object], prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for key, value in mapping.items():
        compound = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, Mapping):
            keys.update(_flatten(value, compound))
        else:
            keys.add(compound)
    return keys


def lint(
    locales: Iterable[str] = DEFAULT_LOCALES,
) -> dict[str, MutableMapping[str, set[str]]]:
    """Return missing/extra keys for each locale."""

    locales = tuple(locales)
    if not locales:
        raise ValueError("At least one locale must be provided")

    keysets = {locale: _flatten(_load_locale(locale)) for locale in locales}
    union: set[str] = set().union(*keysets.values())
    report: dict[str, MutableMapping[str, set[str]]] = {}
    for locale, keys in keysets.items():
        missing = union - keys
        extra = keys - union
        report[locale] = {"missing": missing, "extra": extra}
    return report


def _format_report(report: Mapping[str, Mapping[str, set[str]]]) -> str:
    lines: list[str] = []
    for locale, payload in sorted(report.items()):
        missing = sorted(payload.get("missing", set()))
        extra = sorted(payload.get("extra", set()))
        if not missing and not extra:
            lines.append(f"{locale}: ok")
            continue
        if missing:
            lines.append(f"{locale}: missing {', '.join(missing)}")
        if extra:
            lines.append(f"{locale}: extra {', '.join(extra)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    locales = tuple(argv or DEFAULT_LOCALES)
    report = lint(locales)
    issues = any(payload["missing"] or payload["extra"] for payload in report.values())
    print(_format_report(report))
    return 1 if issues else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main(sys.argv[1:]))
