from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from ogum_lite.publish import metadata
from ogum_lite.publish.workflow import prepare_run_for_publish, publish_to_figshare


class FakeFigshareClient:
    def __init__(self, token: str, base_url: str) -> None:
        self.token = token
        self.base_url = base_url
        self.uploaded: list[Path] = []

    def create_article(self, meta: metadata.PublicationMeta) -> dict[str, Any]:
        return {"id": 77}

    def initiate_upload(self, article_id: int, path: Path) -> dict[str, Any]:
        self.uploaded.append(path)
        return {"id": article_id, "name": path.name}

    def publish(self, article_id: int) -> dict[str, Any]:
        return {"status": "ok", "id": article_id}

    def get_article(self, article_id: int) -> dict[str, Any]:
        return {
            "doi": "10.6084/m9.figshare.777",
            "url_public_html": "https://figshare.com/articles/777",
        }


@pytest.fixture()
def sample_bundle(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "report.html").write_text("report", encoding="utf-8")
    (run_dir / "model_card.json").write_text("{}", encoding="utf-8")

    meta_yaml = tmp_path / "meta.yaml"
    meta = metadata.PublicationMeta(
        title="Ogum Figshare",
        description="Bundle for figshare tests.",
        version="v0.1.1",
        authors=[metadata.PublicationAuthor(name="Alice")],
        keywords=["ogum"],
        license="MIT",
    )
    metadata.to_yaml(meta, meta_yaml)

    bundle_dir = tmp_path / "bundle"
    prepare_run_for_publish(run_dir, meta_yaml, bundle_dir)
    return bundle_dir


def test_publish_to_figshare_with_receipt(
    monkeypatch: pytest.MonkeyPatch, sample_bundle: Path
) -> None:
    created_clients: list[FakeFigshareClient] = []

    def _factory(token: str, base_url: str) -> FakeFigshareClient:
        client = FakeFigshareClient(token, base_url)
        created_clients.append(client)
        return client

    monkeypatch.setattr("ogum_lite.publish.workflow.FigshareClient", _factory)

    receipt = publish_to_figshare(
        sample_bundle,
        env={
            "FIGSHARE_TOKEN": "secret",
            "FIGSHARE_BASE": "https://api.figshare.com/v2",
        },
    )

    assert receipt["doi"] == "10.6084/m9.figshare.777"
    assert receipt["url"] == "https://figshare.com/articles/777"

    receipt_path = sample_bundle / "figshare_receipt.json"
    saved = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert saved["doi"] == "10.6084/m9.figshare.777"

    assert created_clients
    assert created_clients[0].uploaded
    assert created_clients[0].uploaded[0].name == "bundle.zip"


def test_publish_to_figshare_requires_token(sample_bundle: Path) -> None:
    with pytest.raises(RuntimeError):
        publish_to_figshare(sample_bundle, env={})
