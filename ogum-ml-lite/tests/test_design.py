"""Design system smoke tests."""

from __future__ import annotations

from app.design.theme import get_theme
from streamlit.testing.v1 import AppTest


def test_theme_variants() -> None:
    light = get_theme(False)
    dark = get_theme(True)
    assert light["colors"]["background"] != dark["colors"]["background"]
    assert light["colors"]["primary"]


def test_components_smoke() -> None:
    def app() -> None:
        from app.design.components import alert, card, toolbar

        card("Title", "Body")
        alert("info", "Testing")
        toolbar([("Action", "act")])

    test = AppTest.from_function(app)
    result = test.run()
    assert not result.exception
