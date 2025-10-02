"""Utilities for preparing and publishing Ogum-ML results."""

from .metadata import (
    PublicationAuthor,
    PublicationMeta,
    from_yaml,
    to_yaml,
    validate_meta,
)
from .packer import gather_run_artifacts, make_publish_bundle, pack_zip
from .workflow import (
    prepare_run_for_publish,
    publish_status,
    publish_to_figshare,
    publish_to_zenodo,
)

__all__ = [
    "PublicationAuthor",
    "PublicationMeta",
    "from_yaml",
    "to_yaml",
    "validate_meta",
    "gather_run_artifacts",
    "pack_zip",
    "make_publish_bundle",
    "prepare_run_for_publish",
    "publish_to_zenodo",
    "publish_to_figshare",
    "publish_status",
]
