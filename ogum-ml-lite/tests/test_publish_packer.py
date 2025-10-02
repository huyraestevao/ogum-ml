from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from ogum_lite.publish.packer import gather_run_artifacts, make_publish_bundle


def _create_file(path: Path, content: str = "data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_gather_and_bundle_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _create_file(run_dir / "report.html")
    _create_file(run_dir / "model_card.json", "{}")
    _create_file(run_dir / "cv_metrics.json", "{}")

    artifacts = gather_run_artifacts(run_dir)
    assert "report.html" in artifacts
    assert "model_card.json" in artifacts
    assert artifacts["report.html"].name == "report.html"

    out_dir = tmp_path / "bundle"
    extra_dir = out_dir / "extra_files"
    _create_file(extra_dir / "readme.txt", "info")

    bundle = make_publish_bundle(run_dir, [extra_dir / "readme.txt"], out_dir)
    zip_path = bundle["zip"]
    checksum_path = bundle["checksum"]
    manifest = bundle["manifest"]

    assert zip_path.exists()
    assert checksum_path.exists()
    assert manifest["zip"] == "bundle.zip"
    assert manifest["extra_files"] == ["extra_files/readme.txt"]
    assert set(manifest["artifacts"]) == {
        "report.html",
        "model_card.json",
        "cv_metrics.json",
    }

    with zipfile.ZipFile(zip_path) as archive:
        assert set(archive.namelist()) == {
            "cv_metrics.json",
            "model_card.json",
            "report.html",
        }

    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    recorded_digest = checksum_path.read_text(encoding="utf-8").split()[0]
    assert digest == recorded_digest
