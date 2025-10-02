"""Artifact gathering and packaging helpers for publication bundles."""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Mapping
from pathlib import Path

_ARTIFACT_CANDIDATES: Mapping[str, tuple[str, ...]] = {
    "report.html": ("report.html",),
    "report.xlsx": ("report.xlsx",),
    "msc.csv": ("msc.csv",),
    "msc.png": ("msc.png",),
    "segments.json": ("segments.json",),
    "n_segments.csv": ("n_segments.csv",),
    "mech_report.csv": ("mech_report.csv",),
    "model_card.json": ("model_card.json",),
    "cv_metrics.json": ("cv_metrics.json",),
    "theta.zip": ("theta.zip", "theta_curves.zip", "theta_csv.zip"),
    "features.csv": ("features.csv",),
    "presets.yaml": ("presets.yaml",),
    "run_log.jsonl": ("run_log.jsonl",),
    "telemetry.jsonl": ("telemetry.jsonl",),
}


def _first_existing(base: Path, candidates: tuple[str, ...]) -> Path | None:
    for candidate in candidates:
        path = base / candidate
        if path.exists():
            return path
    return None


def gather_run_artifacts(run_dir: Path) -> dict[str, Path]:
    """Collect known artifact files from a run directory.

    Parameters
    ----------
    run_dir
        Path containing generated run artifacts.

    Returns
    -------
    dict
        Mapping of artifact file names to their paths.
    """

    if not run_dir.exists():
        msg = f"Run directory '{run_dir}' does not exist."
        raise FileNotFoundError(msg)

    artifacts: dict[str, Path] = {}
    for arcname, candidates in _ARTIFACT_CANDIDATES.items():
        path = _first_existing(run_dir, candidates)
        if path:
            artifacts[path.name] = path
    return artifacts


def pack_zip(artifacts: dict[str, Path], out_zip: Path) -> Path:
    """Create a deterministic ZIP archive of collected artifacts.

    Parameters
    ----------
    artifacts
        Mapping of archive names to source paths.
    out_zip
        Destination path for the generated archive.

    Returns
    -------
    Path
        Path to the generated ZIP archive.
    """

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for arcname in sorted(artifacts):
            source = artifacts[arcname]
            info = zipfile.ZipInfo(arcname)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.external_attr = 0o644 << 16
            with source.open("rb") as handle:
                data = handle.read()
            bundle.writestr(info, data)
    digest = hashlib.sha256(out_zip.read_bytes()).hexdigest()
    checksum_path = out_zip.with_suffix(out_zip.suffix + ".sha256")
    with checksum_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{digest}  {out_zip.name}\n")
    return out_zip


def make_publish_bundle(
    run_dir: Path, extra_files: list[Path] | None, out_dir: Path
) -> dict[str, object]:
    """Build the publish bundle for a run directory.

    Parameters
    ----------
    run_dir
        Directory containing the run artefacts.
    extra_files
        Optional additional files to be uploaded alongside the bundle.
    out_dir
        Output directory for generated files.

    Returns
    -------
    dict
        A dictionary containing paths to the zip archive, checksum file and the
        manifest data.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = gather_run_artifacts(run_dir)
    zip_path = out_dir / "bundle.zip"
    pack_zip(artifacts, zip_path)
    checksum_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    manifest: dict[str, object] = {
        "run_dir": str(run_dir),
        "artifacts": {name: str(path) for name, path in artifacts.items()},
    }
    if extra_files:
        manifest["extra_files"] = [
            str(path.relative_to(out_dir)) for path in extra_files
        ]
    manifest["zip"] = str(zip_path.relative_to(out_dir))
    manifest["checksum"] = checksum_path.read_text(encoding="utf-8").strip()
    return {"zip": zip_path, "checksum": checksum_path, "manifest": manifest}
