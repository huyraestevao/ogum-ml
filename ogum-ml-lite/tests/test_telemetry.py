from app.services import telemetry


def test_log_event_respects_opt_out(tmp_path, monkeypatch):
    log_dir = tmp_path / "telemetry"
    log_dir.mkdir()
    log_file = log_dir / "telemetry.jsonl"
    with telemetry.telemetry_session(enabled=False):
        telemetry.log_event("prep", {"duration_ms": 10}, workspace=log_dir)
    assert not log_file.exists()


def test_aggregate_collects_durations(tmp_path):
    log_dir = tmp_path / "telemetry"
    log_dir.mkdir()
    log_file = log_dir / "telemetry.jsonl"
    with telemetry.telemetry_session(enabled=True):
        telemetry.log_event(
            "prep",
            {"duration_ms": 100, "experiment": "wizard_vs_tabs", "variant": "wizard"},
            workspace=log_dir,
        )
        telemetry.log_event(
            "prep",
            {"duration_ms": 200, "experiment": "wizard_vs_tabs", "variant": "tabs"},
            workspace=log_dir,
        )
    assert log_file.exists()
    summary = telemetry.aggregate(log_file)
    prep_data = summary["events"]["prep"]
    assert prep_data["count"] == 2
    assert prep_data["avg_duration_ms"] == 150.0
    experiments = summary["experiments"]["wizard_vs_tabs"]
    assert experiments == {"wizard": 1, "tabs": 1}
