"""Session state helpers for the Streamlit dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import streamlit as st
import yaml
from ogum_lite.ui.presets import load_presets, merge_presets
from ogum_lite.ui.workspace import Workspace

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRESET_PATH = APP_ROOT / "presets.yaml"
DEFAULT_WORKSPACE = Path.cwd() / "artifacts" / "ui-session"


@dataclass(slots=True)
class Artifact:
    """Metadata for generated artifacts."""

    key: str
    path: Path
    description: str | None = None


def ensure_session(locale: str = "pt") -> None:
    """Initialise Streamlit session state with sane defaults."""

    st.session_state.setdefault("locale", locale)
    st.session_state.setdefault("dark_mode", False)
    st.session_state.setdefault("workspace_root", str(DEFAULT_WORKSPACE))
    st.session_state.setdefault("artifacts", {})
    st.session_state.setdefault(
        "preset_yaml", DEFAULT_PRESET_PATH.read_text(encoding="utf-8")
    )


def get_workspace() -> Workspace:
    """Return the active workspace object."""

    root = Path(st.session_state.get("workspace_root", str(DEFAULT_WORKSPACE)))
    workspace = st.session_state.get("workspace")
    if isinstance(workspace, Workspace) and workspace.path == root:
        return workspace
    workspace = Workspace(root)
    st.session_state["workspace"] = workspace
    return workspace


def set_workspace(root: Path) -> Workspace:
    """Persist a new workspace path and rebuild the object."""

    st.session_state["workspace_root"] = str(root)
    workspace = Workspace(root)
    st.session_state["workspace"] = workspace
    return workspace


def _load_preset_from_state() -> dict[str, Any]:
    yaml_text = st.session_state.get("preset_yaml")
    if not yaml_text:
        return load_presets(DEFAULT_PRESET_PATH)
    try:
        parsed = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError:
        return load_presets(DEFAULT_PRESET_PATH)
    base = load_presets(DEFAULT_PRESET_PATH)
    return merge_presets(base, parsed)


def get_preset() -> dict[str, Any]:
    """Return the active preset merging overrides when necessary."""

    preset = st.session_state.get("preset")
    if isinstance(preset, dict):
        return preset
    preset = _load_preset_from_state()
    st.session_state["preset"] = preset
    return preset


def set_preset_yaml(text: str) -> None:
    """Update the YAML preset override in session state."""

    st.session_state["preset_yaml"] = text
    st.session_state.pop("preset", None)


def get_preset_yaml() -> str:
    """Return the current YAML override for the preset."""

    return st.session_state.get(
        "preset_yaml", DEFAULT_PRESET_PATH.read_text(encoding="utf-8")
    )


def load_preset_file(path: Path) -> dict[str, Any]:
    """Load a preset from disk and update the override."""

    preset = load_presets(path)
    base = _load_preset_from_state()
    merged = merge_presets(base, preset)
    st.session_state["preset_yaml"] = yaml.safe_dump(
        merged, sort_keys=False, allow_unicode=True
    )
    st.session_state["preset"] = merged
    return merged


def reset_preset() -> dict[str, Any]:
    """Restore the default preset bundled with the application."""

    text = DEFAULT_PRESET_PATH.read_text(encoding="utf-8")
    st.session_state["preset_yaml"] = text
    st.session_state.pop("preset", None)
    return get_preset()


def persist_upload(upload, *, subdir: str = "uploads") -> Path:
    """Persist an uploaded file inside the workspace."""

    workspace = get_workspace()
    uploads_dir = workspace.resolve(subdir)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    target = uploads_dir / upload.name
    with target.open("wb") as handle:
        handle.write(upload.getbuffer())
    register_artifact(upload.name, target, description="upload")
    return target


def register_artifact(key: str, path: Path, description: str | None = None) -> None:
    """Register an artifact path in session state."""

    artifacts: Dict[str, Any] = st.session_state.setdefault("artifacts", {})
    artifacts[key] = {"path": str(path), "description": description}
    st.session_state["artifacts"] = artifacts


def get_artifact(key: str) -> Path | None:
    """Retrieve a previously registered artifact."""

    artifacts: Dict[str, Any] = st.session_state.get("artifacts", {})
    info = artifacts.get(key)
    if not info:
        return None
    return Path(info["path"])


def list_artifacts() -> list[Artifact]:
    """Return a list of registered artifacts."""

    artifacts: Dict[str, Any] = st.session_state.get("artifacts", {})
    return [
        Artifact(
            key=key,
            path=Path(payload["path"]),
            description=payload.get("description"),
        )
        for key, payload in artifacts.items()
    ]


def workspace_log_tail(lines: int = 8) -> list[str]:
    """Return the tail of the workspace log file."""

    workspace = get_workspace()
    log_path = workspace.resolve(workspace.log_name)
    if not log_path.exists():
        return []
    tail = log_path.read_text(encoding="utf-8").splitlines()[-lines:]
    return tail
