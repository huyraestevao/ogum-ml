"""High-level workflows orchestrating publication preparation and uploads."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Mapping

from .figshare_client import FigshareClient
from .metadata import PublicationMeta, from_yaml, to_yaml, validate_meta
from .packer import make_publish_bundle
from .zenodo_client import ZenodoClient


def _metadata_to_dict(meta: PublicationMeta) -> dict:
    authors = []
    for author in meta.authors:
        authors.append(
            {
                "name": author.name,
                "affiliation": author.affiliation,
                "orcid": author.orcid,
            }
        )
    return {
        "title": meta.title,
        "description": meta.description,
        "version": meta.version,
        "authors": authors,
        "keywords": list(meta.keywords),
        "license": meta.license,
        "funding": list(meta.funding) if meta.funding is not None else None,
        "related_identifiers": (
            list(meta.related_identifiers)
            if meta.related_identifiers is not None
            else None
        ),
        "upload_files": (
            [str(path) for path in meta.upload_files]
            if meta.upload_files is not None
            else None
        ),
        "community": meta.community,
        "category": meta.category,
    }


def prepare_run_for_publish(run_dir: Path, meta_yaml: Path, out_dir: Path) -> dict:
    """Prepare a run directory for publication.

    The function validates metadata, copies auxiliary files and creates the
    publish bundle.
    """

    meta = from_yaml(meta_yaml)
    validation = validate_meta(meta)
    if not validation["ok"]:
        issues = "; ".join(validation["issues"])
        msg = f"Invalid publication metadata: {issues}"
        raise ValueError(msg)

    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_copy = out_dir / "metadata.yaml"
    to_yaml(meta, metadata_copy)

    extra_files: list[Path] = []
    if meta.upload_files:
        extras_dir = out_dir / "extra_files"
        extras_dir.mkdir(parents=True, exist_ok=True)
        for path in meta.upload_files:
            if not path.exists():
                msg = f"Upload file '{path}' does not exist."
                raise FileNotFoundError(msg)
            destination = extras_dir / path.name
            shutil.copy2(path, destination)
            extra_files.append(destination)

    bundle = make_publish_bundle(run_dir, extra_files, out_dir)

    manifest_data = bundle["manifest"].copy()
    manifest_data["metadata"] = _metadata_to_dict(meta)
    manifest_data["metadata_path"] = str(metadata_copy)
    manifest_path = out_dir / "publish_manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    return {
        "manifest_path": manifest_path,
        "metadata_path": metadata_copy,
        "bundle_zip": bundle["zip"],
        "checksum_path": bundle["checksum"],
    }


def _load_manifest(bundle_dir: Path) -> dict:
    manifest_path = bundle_dir / "publish_manifest.json"
    if not manifest_path.exists():
        msg = f"Bundle manifest not found at {manifest_path}."
        raise FileNotFoundError(msg)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_metadata(bundle_dir: Path) -> PublicationMeta:
    metadata_path = bundle_dir / "metadata.yaml"
    if not metadata_path.exists():
        msg = f"Metadata YAML not found at {metadata_path}."
        raise FileNotFoundError(msg)
    return from_yaml(metadata_path)


def _resolve_path(bundle_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = bundle_dir / path
    return path


def publish_to_zenodo(bundle_dir: Path, env: Mapping[str, str]) -> dict:
    """Publish a previously prepared bundle to Zenodo."""

    manifest = _load_manifest(bundle_dir)
    meta = _load_metadata(bundle_dir)
    token = env.get("ZENODO_TOKEN", "")
    if not token:
        msg = "ZENODO_TOKEN is not configured."
        raise RuntimeError(msg)
    base_url = env.get("ZENODO_BASE", "https://zenodo.org/api")

    client = ZenodoClient(token=token, base_url=base_url)
    deposition = client.create_deposition(meta)
    deposition_id = deposition.get("id")
    if deposition_id is None:
        msg = "Zenodo deposition response missing 'id'."
        raise RuntimeError(msg)

    bundle_path = _resolve_path(bundle_dir, manifest["zip"])
    client.upload_file(int(deposition_id), bundle_path)

    for extra in manifest.get("extra_files", []):
        client.upload_file(int(deposition_id), _resolve_path(bundle_dir, extra))

    publish_response = client.publish(int(deposition_id))
    record = client.get_record(int(deposition_id))
    receipt = {
        "platform": "zenodo",
        "deposition_id": deposition_id,
        "doi": record.get("doi") or record.get("metadata", {}).get("doi"),
        "conceptdoi": record.get("conceptdoi")
        or record.get("metadata", {}).get("conceptdoi"),
        "url": record.get("links", {}).get("html"),
        "publish_response": publish_response,
    }

    receipt_path = bundle_dir / "zenodo_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def publish_to_figshare(bundle_dir: Path, env: Mapping[str, str]) -> dict:
    """Publish a previously prepared bundle to Figshare."""

    manifest = _load_manifest(bundle_dir)
    meta = _load_metadata(bundle_dir)
    token = env.get("FIGSHARE_TOKEN", "")
    if not token:
        msg = "FIGSHARE_TOKEN is not configured."
        raise RuntimeError(msg)
    base_url = env.get("FIGSHARE_BASE", "https://api.figshare.com/v2")

    client = FigshareClient(token=token, base_url=base_url)
    article = client.create_article(meta)
    article_id = article.get("id") or article.get("article_id")
    if article_id is None:
        msg = "Figshare response missing article identifier."
        raise RuntimeError(msg)

    bundle_path = _resolve_path(bundle_dir, manifest["zip"])
    client.initiate_upload(int(article_id), bundle_path)
    for extra in manifest.get("extra_files", []):
        client.initiate_upload(int(article_id), _resolve_path(bundle_dir, extra))

    publish_response = client.publish(int(article_id))
    record = client.get_article(int(article_id))
    receipt = {
        "platform": "figshare",
        "article_id": article_id,
        "doi": record.get("doi") or record.get("doi_url"),
        "url": record.get("url_public_html") or record.get("url"),
        "publish_response": publish_response,
    }

    receipt_path = bundle_dir / "figshare_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def publish_status(bundle_dir: Path) -> dict:
    """Summarise publication receipts for the given bundle directory."""

    status: dict[str, dict] = {}
    manifest = _load_manifest(bundle_dir)
    status["manifest"] = manifest

    zenodo_receipt = bundle_dir / "zenodo_receipt.json"
    if zenodo_receipt.exists():
        status["zenodo"] = json.loads(zenodo_receipt.read_text(encoding="utf-8"))

    figshare_receipt = bundle_dir / "figshare_receipt.json"
    if figshare_receipt.exists():
        status["figshare"] = json.loads(figshare_receipt.read_text(encoding="utf-8"))

    return status
