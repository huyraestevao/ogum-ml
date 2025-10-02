from app.edu import exercises


def _get(key: str):
    return next(ex for ex in exercises.EXERCISES if ex.key == key)


def test_choose_ea_scoring():
    ex = _get("choose_ea")
    good = ex.evaluate({"Ea": 320.0})
    poor = ex.evaluate({"Ea": 100.0})
    assert good["score"] >= poor["score"]
    assert "MSE" in good["feedback"]


def test_segments_tolerance():
    ex = _get("segments")
    refs = exercises.get_references()
    answers = {
        "seg_55": refs["seg_55"],
        "seg_70": refs["seg_70"],
        "seg_90": refs["seg_90"],
    }
    result = ex.evaluate(answers)
    assert result["score"] == 1.0
    off = ex.evaluate({"seg_55": 0.0, "seg_70": 0.0, "seg_90": 0.0})
    assert off["score"] < 1.0


def test_blaine_error_metric():
    ex = _get("blaine_n")
    refs = exercises.get_references()
    good = ex.evaluate({"n": refs["n"]})
    bad = ex.evaluate({"n": refs["n"] + 1.0})
    assert good["score"] > bad["score"]
    assert "Referência" in good["feedback"]
