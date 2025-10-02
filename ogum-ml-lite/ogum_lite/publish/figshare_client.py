"""Minimal Figshare API client for publication workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from .metadata import PublicationMeta


class FigshareClient:
    """Client wrapper for Figshare article management endpoints."""

    def __init__(self, token: str, base_url: str) -> None:
        if not token:
            msg = (
                "Figshare token is required. "
                "Configure FIGSHARE_TOKEN in the environment."
            )
            raise ValueError(msg)
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"token {token}"})

    def _request(self, method: str, endpoint: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    def create_article(self, meta: PublicationMeta) -> dict[str, Any]:
        """Create a Figshare article using the supplied metadata."""

        payload = self._serialize_metadata(meta)
        response = self._request("POST", "/account/articles", json=payload)
        return response.json()

    def initiate_upload(self, article_id: int, path: Path) -> dict[str, Any]:
        """Upload a file to the specified article."""

        with path.open("rb") as handle:
            files = {"filedata": (path.name, handle)}
            response = self._request(
                "POST", f"/account/articles/{article_id}/files", files=files
            )
        return response.json()

    def publish(self, article_id: int) -> dict[str, Any]:
        """Publish an article and expose it publicly."""

        response = self._request("POST", f"/account/articles/{article_id}/publish")
        return response.json()

    def get_article(self, article_id: int) -> dict[str, Any]:
        """Fetch article metadata from Figshare."""

        response = self._request("GET", f"/account/articles/{article_id}")
        return response.json()

    def _serialize_metadata(self, meta: PublicationMeta) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "title": meta.title,
            "description": meta.description,
            "version": meta.version,
            "authors": [author.name for author in meta.authors],
            "keywords": meta.keywords,
            "license": meta.license,
        }
        if meta.category:
            payload["defined_type"] = meta.category
        if meta.funding:
            payload["funding"] = list(meta.funding)
        if meta.related_identifiers:
            payload["references"] = [
                item.get("identifier")
                for item in meta.related_identifiers
                if item.get("identifier")
            ]
        return payload
