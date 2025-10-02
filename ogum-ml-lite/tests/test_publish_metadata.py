from __future__ import annotations

from pathlib import Path

from ogum_lite.publish.metadata import (
    PublicationAuthor,
    PublicationMeta,
    from_yaml,
    to_yaml,
    validate_meta,
)


def test_validate_meta_success() -> None:
    meta = PublicationMeta(
        title="Ogum ML Results",
        description="Comprehensive run outputs.",
        version="v0.3.0",
        authors=[PublicationAuthor(name="Alice Doe", affiliation="Ogum Lab")],
        keywords=["sintering", "ogum"],
        license="MIT",
        funding=["CNPq-123"],
        related_identifiers=[
            {"identifier": "10.5281/zenodo.123", "relation": "isSupplementTo"}
        ],
    )
    result = validate_meta(meta)
    assert result["ok"] is True
    assert result["issues"] == []


def test_validate_meta_reports_missing_fields() -> None:
    meta = PublicationMeta(title="", description="", version="", license="")
    validation = validate_meta(meta)
    assert validation["ok"] is False
    assert "Title is required." in validation["issues"]
    assert "Description is required." in validation["issues"]
    assert "Version is required." in validation["issues"]
    assert "At least one author is required." in validation["issues"]
    assert "At least one keyword is required." in validation["issues"]
    assert "License is required." in validation["issues"]


def test_metadata_yaml_roundtrip(tmp_path: Path) -> None:
    extra_file = tmp_path / "notes.txt"
    extra_file.write_text("important notes", encoding="utf-8")

    meta = PublicationMeta(
        title="Ogum Run",
        description="Important scientific artefacts.",
        version="v1.0.0",
        authors=[
            PublicationAuthor(name="Alice Doe"),
            PublicationAuthor(name="Bob Ray"),
        ],
        keywords=["ogum", "ml"],
        license="CC-BY-4.0",
        upload_files=[extra_file],
    )

    yaml_path = tmp_path / "meta.yaml"
    to_yaml(meta, yaml_path)

    loaded = from_yaml(yaml_path)
    assert loaded.title == meta.title
    assert loaded.description == meta.description
    assert loaded.version == meta.version
    assert [author.name for author in loaded.authors] == [
        "Alice Doe",
        "Bob Ray",
    ]
    assert loaded.upload_files == [extra_file.resolve()]

    validation = validate_meta(loaded)
    assert validation["ok"] is True
