"""Minimal REST client for interacting with the Zenodo deposition API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from .metadata import PublicationMeta


class ZenodoClient:
    """Client providing a thin wrapper around the Zenodo deposition API."""

    def __init__(self, token: str, base_url: str) -> None:
        if not token:
            msg = "Zenodo token is required. Configure ZENODO_TOKEN in the environment."
            raise ValueError(msg)
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}"})

    def _request(self, method: str, endpoint: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    def create_deposition(self, meta: PublicationMeta) -> dict[str, Any]:
        """Create a new Zenodo deposition for the provided metadata.

        Parameters
        ----------
        meta
            Publication metadata.

        Returns
        -------
        dict
            JSON response describing the deposition, including the identifier.
        """

        payload = {"metadata": self._serialize_metadata(meta)}
        response = self._request("POST", "/deposit/depositions", json=payload)
        return response.json()

    def upload_file(self, deposition_id: int, path: Path) -> dict[str, Any]:
        """Upload a file to an existing deposition."""

        with path.open("rb") as handle:
            files = {"file": (path.name, handle)}
            response = self._request(
                "POST", f"/deposit/depositions/{deposition_id}/files", files=files
            )
        return response.json()

    def publish(self, deposition_id: int) -> dict[str, Any]:
        """Publish a deposition, minting the DOI."""

        response = self._request(
            "POST", f"/deposit/depositions/{deposition_id}/actions/publish"
        )
        return response.json()

    def get_record(self, deposition_id: int) -> dict[str, Any]:
        """Retrieve the deposition record."""

        response = self._request("GET", f"/deposit/depositions/{deposition_id}")
        return response.json()

    def _serialize_metadata(self, meta: PublicationMeta) -> dict[str, Any]:
        creators = []
        for author in meta.authors:
            creator = {"name": author.name}
            if author.affiliation:
                creator["affiliation"] = author.affiliation
            if author.orcid:
                creator["orcid"] = author.orcid
            creators.append(creator)
        metadata: dict[str, Any] = {
            "title": meta.title,
            "upload_type": "dataset",
            "description": meta.description,
            "version": meta.version,
            "creators": creators,
            "keywords": meta.keywords,
            "license": meta.license,
        }
        if meta.funding:
            metadata["funding"] = list(meta.funding)
        if meta.related_identifiers:
            metadata["related_identifiers"] = [
                dict(item) for item in meta.related_identifiers
            ]
        if meta.community:
            metadata["communities"] = [{"identifier": meta.community}]
        return metadata
