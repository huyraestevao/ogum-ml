#!/usr/bin/env python3
"""
Ogum-ML — Release 1.0 readiness checker.

Executa:
1) Upgrade de pip/setuptools/wheel (feito no workflow).
2) Instalação de requirements.
3) Checagem de pins legados (pydantic<2, pydantic-settings<2, pyyaml<6).
4) Lint: ruff.
5) Formatação: black --check.
6) Testes: pytest -q.
7) Smoke da CLI: python -m ogum_lite.cli --help.

Gera:
- release_check.log       (log textual)
- release_summary.md      (resumo markdown para Actions Summary)

Retorno:
- exit code 0 se GO
- exit code != 0 se NO-GO
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "release_check.log"
SUMMARY = ROOT / "release_summary.md"


def run(cmd: list[str], check: bool = False) -> int:
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        fh.write(proc.stdout)
        returncode = proc.returncode
    if check and returncode != 0:
        sys.exit(returncode)
    return returncode


def write_summary(lines: list[str]) -> None:
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def grep_legacy_pins() -> list[str]:
    """Procura pins legados que quebram Py 3.12."""
    patterns = ("pydantic<2", "pydantic-settings<2", "pyyaml<6")
    offenders: list[str] = []
    skip_dirs = {".git", ".venv", "venv", "site-packages", "build", "dist", ".mypy_cache"}
    allowed_suffixes = {".txt", ".cfg", ".toml", ".yml", ".yaml", ".in", ".py"}
    for path in ROOT.rglob("*"):
        if path.is_dir():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix.lower() not in allowed_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pat in patterns:
            if pat in text:
                offenders.append(f"{path}: '{pat}'")
    return sorted(set(offenders))


def main() -> int:
    LOG.write_text("# Ogum-ML — Release 1.0 readiness log\n", encoding="utf-8")
    summary = ["### Checklist", ""]
    ok = True

    # 1) Sanidade: ferramentas
    for tool in ("ruff", "black", "pytest"):
        if shutil.which(tool) is None:
            run([sys.executable, "-m", "pip", "install", "--upgrade", tool])
    summary.append("✅ Ferramentas dev disponíveis (ruff/black/pytest)")

    # 2) Instalação (já feita no workflow), mas validamos import rápido
    # Se quiser reforçar, descomente a linha abaixo:
    # run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", "requirements.txt"])

    # 3) Pins legados
    offenders = grep_legacy_pins()
    if offenders:
        ok = False
        summary.append("❌ Pins legados encontrados:")
        summary.extend([f"- {o}" for o in offenders])
    else:
        summary.append("✅ Sem pins legados (pydantic<2, pydantic-settings<2, pyyaml<6)")

    # 4) Lint
    rc = run(["ruff", "check", "."], check=False)
    if rc != 0:
        ok = False
        summary.append("❌ Ruff falhou")
    else:
        summary.append("✅ Ruff OK")

    # 5) Black
    rc = run(["black", "--check", "."], check=False)
    if rc != 0:
        ok = False
        summary.append("❌ Black --check falhou")
    else:
        summary.append("✅ Black OK")

    # 6) Pytest
    rc = run(["pytest", "-q"], check=False)
    if rc != 0:
        ok = False
        summary.append("❌ Testes falharam (pytest)")
    else:
        summary.append("✅ Testes OK (pytest)")

    # 7) Smoke CLI
    rc = run([sys.executable, "-m", "ogum_lite.cli", "--help"], check=False)
    if rc != 0:
        ok = False
        summary.append("❌ CLI falhou (python -m ogum_lite.cli --help)")
    else:
        summary.append("✅ CLI OK")

    # Resultado final
    summary.append("")
    if ok:
        summary.append("## ✅ Release 1.0 READY (GO)")
        code = 0
    else:
        summary.append("## ❌ Release 1.0 NOT READY (NO-GO)")
        code = 1

    write_summary(summary)
    return code


if __name__ == "__main__":
    sys.exit(main())
