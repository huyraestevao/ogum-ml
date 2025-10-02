"""Smoke tests for the guided wizard state machine."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

import pytest
from app.pages import page_wizard


class DummyState:
    def __init__(self, resolver: Callable[[str], object | None]) -> None:
        self._resolver = resolver

    def get_artifact(self, key: str) -> object | None:
        return self._resolver(key)


class DummyStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}

    def columns(self, sizes):  # pragma: no cover - not used in tests
        return [
            SimpleNamespace(
                button=lambda *_, **__: False, caption=lambda *_, **__: None
            )
            for _ in sizes
        ]

    def caption(self, *_args, **_kwargs) -> None:  # pragma: no cover - noop
        return None

    def subheader(self, *_args, **_kwargs) -> None:  # pragma: no cover - noop
        return None

    def markdown(self, *_args, **_kwargs) -> None:  # pragma: no cover - noop
        return None

    def divider(self) -> None:  # pragma: no cover - noop
        return None


@pytest.fixture(autouse=True)
def _reset_session(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_st = DummyStreamlit()
    monkeypatch.setattr(page_wizard, "st", dummy_st)
    monkeypatch.setattr(page_wizard, "state", DummyState(lambda _key: None))


def _make_path(exists: bool) -> SimpleNamespace:
    return SimpleNamespace(exists=lambda: exists)


def test_sync_flags_detects_existing_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts = {
        "prep_csv": _make_path(True),
        "features_csv": _make_path(True),
        "theta_table": _make_path(True),
        "segments": _make_path(True),
        "mechanism": _make_path(True),
        "ml_cls": _make_path(True),
        "export_report": _make_path(True),
        "session_zip": _make_path(True),
    }
    monkeypatch.setattr(
        page_wizard, "state", DummyState(lambda key: artifacts.get(key))
    )
    page_wizard._sync_flags()
    flags = page_wizard.st.session_state[page_wizard.FLAGS_KEY]
    assert all(flags.get(step.key) for step in page_wizard.STEPS)


def test_flags_do_not_mark_missing_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts = {"prep_csv": _make_path(True), "features_csv": None}
    monkeypatch.setattr(
        page_wizard, "state", DummyState(lambda key: artifacts.get(key))
    )
    page_wizard._sync_flags()
    flags = page_wizard.st.session_state[page_wizard.FLAGS_KEY]
    assert flags.get("data") is True
    assert flags.get("features") is not True


def test_mark_complete_updates_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    page_wizard._mark_complete("data")
    assert page_wizard._is_complete("data") is True
    assert page_wizard._is_complete("features") is False


def test_step_order_matches_spec() -> None:
    assert [step.key for step in page_wizard.STEPS] == [
        "data",
        "features",
        "msc",
        "segments",
        "mechanism",
        "ml",
        "export",
    ]
