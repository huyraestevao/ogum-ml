from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from ogum_lite.publish import metadata
from ogum_lite.publish.workflow import prepare_run_for_publish, publish_to_zenodo


class FakeZenodoClient:
    def __init__(self, token: str, base_url: str) -> None:
        self.token = token
        self.base_url = base_url
        self.uploaded: list[Path] = []
        self.received_meta: metadata.PublicationMeta | None = None

    def create_deposition(self, meta: metadata.PublicationMeta) -> dict[str, Any]:
        self.received_meta = meta
        return {"id": 42}

    def upload_file(self, deposition_id: int, path: Path) -> dict[str, Any]:
        self.uploaded.append(path)
        return {"filename": path.name, "id": deposition_id}

    def publish(self, deposition_id: int) -> dict[str, Any]:
        return {"status": "published", "id": deposition_id}

    def get_record(self, deposition_id: int) -> dict[str, Any]:
        return {
            "doi": "10.5281/zenodo.9999",
            "links": {"html": "https://zenodo.org/record/9999"},
        }


@pytest.fixture()
def sample_run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "report.html").write_text("report", encoding="utf-8")
    (run_dir / "model_card.json").write_text("{}", encoding="utf-8")

    meta_yaml = tmp_path / "meta.yaml"
    meta = metadata.PublicationMeta(
        title="Ogum Run",
        description="Results ready for sharing.",
        version="v0.1.0",
        authors=[metadata.PublicationAuthor(name="Alice")],
        keywords=["ogum"],
        license="MIT",
    )
    metadata.to_yaml(meta, meta_yaml)
    return run_dir, meta_yaml


def test_publish_to_zenodo_creates_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_run: tuple[Path, Path],
) -> None:
    run_dir, meta_yaml = sample_run
    bundle_dir = tmp_path / "bundle"
    prepare_run_for_publish(run_dir, meta_yaml, bundle_dir)

    created_clients: list[FakeZenodoClient] = []

    def _factory(token: str, base_url: str) -> FakeZenodoClient:
        client = FakeZenodoClient(token, base_url)
        created_clients.append(client)
        return client

    monkeypatch.setattr("ogum_lite.publish.workflow.ZenodoClient", _factory)

    receipt = publish_to_zenodo(
        bundle_dir,
        env={"ZENODO_TOKEN": "token", "ZENODO_BASE": "https://sandbox.zenodo.org/api"},
    )

    assert receipt["doi"] == "10.5281/zenodo.9999"
    assert receipt["url"] == "https://zenodo.org/record/9999"

    receipt_path = bundle_dir / "zenodo_receipt.json"
    assert receipt_path.exists()
    saved = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert saved["doi"] == "10.5281/zenodo.9999"

    assert created_clients
    client = created_clients[0]
    assert client.received_meta is not None
    assert client.uploaded
    assert client.uploaded[0].name == "bundle.zip"


def test_publish_to_zenodo_requires_token(
    tmp_path: Path, sample_run: tuple[Path, Path]
) -> None:
    run_dir, meta_yaml = sample_run
    bundle_dir = tmp_path / "bundle"
    prepare_run_for_publish(run_dir, meta_yaml, bundle_dir)

    with pytest.raises(RuntimeError):
        publish_to_zenodo(bundle_dir, env={})
