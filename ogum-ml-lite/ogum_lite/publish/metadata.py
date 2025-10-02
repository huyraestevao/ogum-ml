"""Metadata models and utilities for scientific publication workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(slots=True)
class PublicationAuthor:
    """Author information for a publication record.

    Parameters
    ----------
    name
        Full name of the author as it should appear in the record.
    affiliation
        Affiliation of the author (laboratory, institution or company). Optional.
    orcid
        ORCID identifier for the author, if available.
    """

    name: str
    affiliation: str | None = None
    orcid: str | None = None


@dataclass(slots=True)
class PublicationMeta:
    """Canonical metadata describing a publication package.

    Parameters
    ----------
    title
        Title of the deposition or article.
    description
        Rich description of the content. Markdown or HTML is accepted by most
        repositories.
    version
        Semantic version or label identifying the release.
    authors
        List of authors contributing to the deposition.
    keywords
        Keywords that summarise the content.
    license
        Short license identifier (e.g. ``"CC-BY-4.0"`` or ``"MIT"``).
    funding
        Optional list describing funding acknowledgements.
    related_identifiers
        Optional list of mappings describing related works. Each mapping should
        at least contain ``identifier`` and ``relation`` keys.
    upload_files
        Optional list of extra files to upload besides the generated bundle.
    community
        Optional Zenodo community slug.
    category
        Optional Figshare article category.
    """

    title: str
    description: str
    version: str
    authors: list[PublicationAuthor] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    license: str = ""
    funding: list[str] | None = None
    related_identifiers: list[dict[str, Any]] | None = None
    upload_files: list[Path] | None = None
    community: str | None = None
    category: str | None = None


def _require(condition: bool, issues: list[str], message: str) -> None:
    if not condition:
        issues.append(message)


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _validate_authors(authors: Iterable[PublicationAuthor], issues: list[str]) -> None:
    authors_list = list(authors)
    _require(bool(authors_list), issues, "At least one author is required.")
    for index, author in enumerate(authors_list):
        if not _is_non_empty_str(author.name):
            issues.append(f"Author #{index + 1} name must be a non-empty string.")
        if author.orcid and not _validate_orcid(author.orcid):
            msg = (
                f"Author #{index + 1} ORCID '{author.orcid}' is not valid "
                "(format: 0000-0000-0000-0000)."
            )
            issues.append(msg)


def _validate_orcid(orcid: str) -> bool:
    if not isinstance(orcid, str):
        return False
    parts = orcid.split("-")
    if len(parts) != 4:
        return False
    if any(len(part) != 4 or not part.isdigit() for part in parts[:-1]):
        return False
    last_part = parts[-1]
    return len(last_part) == 4 and all(
        ch.isdigit() or ch.upper() == "X" for ch in last_part
    )


def _validate_keywords(keywords: Iterable[str], issues: list[str]) -> None:
    provided = list(keywords)
    kw_list = [kw for kw in provided if _is_non_empty_str(kw)]
    _require(bool(kw_list), issues, "At least one keyword is required.")
    if len(kw_list) != len(provided):
        issues.append("Keywords must be non-empty strings.")


def _validate_related_identifiers(
    related_identifiers: Iterable[dict[str, Any]] | None, issues: list[str]
) -> None:
    if related_identifiers is None:
        return
    for index, entry in enumerate(related_identifiers):
        if not isinstance(entry, dict):
            issues.append(
                "Related identifiers must be dictionaries with identifier metadata."
            )
            continue
        if not _is_non_empty_str(entry.get("identifier")):
            msg = (
                f"Related identifier #{index + 1} must include a non-empty "
                "'identifier'."
            )
            issues.append(msg)
        if not _is_non_empty_str(entry.get("relation")):
            issues.append(
                f"Related identifier #{index + 1} must include a non-empty 'relation'."
            )


def _validate_upload_files(
    upload_files: Iterable[Path] | None, issues: list[str]
) -> None:
    if upload_files is None:
        return
    for path in upload_files:
        if not isinstance(path, Path):
            issues.append("Upload files must be provided as pathlib.Path objects.")


def validate_meta(meta: PublicationMeta) -> dict[str, Any]:
    """Validate metadata values for completeness.

    Parameters
    ----------
    meta
        Metadata instance to validate.

    Returns
    -------
    dict
        Mapping containing ``ok`` boolean flag and a list of textual issues.
    """

    issues: list[str] = []
    _require(_is_non_empty_str(meta.title), issues, "Title is required.")
    _require(_is_non_empty_str(meta.description), issues, "Description is required.")
    _require(_is_non_empty_str(meta.version), issues, "Version is required.")
    _validate_authors(meta.authors, issues)
    _validate_keywords(meta.keywords, issues)
    _require(_is_non_empty_str(meta.license), issues, "License is required.")
    _validate_related_identifiers(meta.related_identifiers, issues)
    _validate_upload_files(meta.upload_files, issues)
    return {"ok": not issues, "issues": issues}


def from_yaml(path: Path) -> PublicationMeta:
    """Load publication metadata from a YAML file.

    Parameters
    ----------
    path
        Path to a YAML document describing :class:`PublicationMeta` fields.

    Returns
    -------
    PublicationMeta
        Metadata object populated with the file contents.
    """

    with path.open("r", encoding="utf-8") as handle:
        raw_data = yaml.safe_load(handle) or {}
    authors = [
        PublicationAuthor(**author_dict) for author_dict in raw_data.get("authors", [])
    ]
    upload_files = raw_data.get("upload_files")
    if upload_files is not None:
        resolved_files: list[Path] = []
        for item in upload_files:
            path_obj = Path(item)
            if not path_obj.is_absolute():
                path_obj = (path.parent / path_obj).resolve()
            resolved_files.append(path_obj)
        upload_files = resolved_files
    return PublicationMeta(
        title=raw_data.get("title", ""),
        description=raw_data.get("description", ""),
        version=raw_data.get("version", ""),
        authors=authors,
        keywords=list(raw_data.get("keywords", [])),
        license=raw_data.get("license", ""),
        funding=(
            list(raw_data.get("funding", []))
            if raw_data.get("funding") is not None
            else None
        ),
        related_identifiers=(
            list(raw_data.get("related_identifiers", []))
            if raw_data.get("related_identifiers") is not None
            else None
        ),
        upload_files=upload_files,
        community=raw_data.get("community"),
        category=raw_data.get("category"),
    )


def _serialize_author(author: PublicationAuthor) -> dict[str, Any]:
    return {key: value for key, value in asdict(author).items() if value is not None}


def to_yaml(meta: PublicationMeta, path: Path) -> None:
    """Write metadata information to a YAML file.

    Parameters
    ----------
    meta
        Metadata instance to serialise.
    path
        Output YAML file path.
    """

    serialisable = asdict(meta)
    serialisable["authors"] = [_serialize_author(author) for author in meta.authors]
    if meta.upload_files is not None:
        serialisable["upload_files"] = [str(item) for item in meta.upload_files]
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serialisable, handle, sort_keys=False, allow_unicode=True)
