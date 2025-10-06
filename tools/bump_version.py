#!/usr/bin/env python3
"""Atualiza a versão no pyproject.toml.

Uso:
  python tools/bump_version.py --file pyproject.toml --version 1.0.0
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

try:
    import tomli
    import tomli_w
except Exception as exc:
    print(f"[err] tomli/tomli-w não disponível: {exc}", file=sys.stderr)
    sys.exit(2)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True)
    p.add_argument("--version", required=True)
    args = p.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"[err] arquivo não encontrado: {path}", file=sys.stderr)
        return 2

    data = tomli.loads(path.read_text(encoding="utf-8"))
    # suporta tanto [project] quanto tool.poetry
    if "project" in data and isinstance(data["project"], dict):
        data["project"]["version"] = args.version
    elif "tool" in data and "poetry" in data["tool"]:
        data["tool"]["poetry"]["version"] = args.version
    else:
        print("[err] não achei chave de versão (project.version ou tool.poetry.version)", file=sys.stderr)
        return 3

    path.write_text(tomli_w.dumps(data), encoding="utf-8")
    print(f"[ok] versão atualizada para {args.version} em {path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
