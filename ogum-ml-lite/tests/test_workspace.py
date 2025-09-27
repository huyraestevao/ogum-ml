import json
import zipfile
from pathlib import Path

from ogum_lite.ui.workspace import Workspace


def test_workspace_zip_outputs(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "session")
    data_dir = ws.resolve("outputs")
    data_dir.mkdir(parents=True, exist_ok=True)
    file_a = data_dir / "a.txt"
    file_a.write_text("alpha", encoding="utf-8")
    file_b = data_dir / "nested" / "b.txt"
    file_b.parent.mkdir(parents=True, exist_ok=True)
    file_b.write_text("bravo", encoding="utf-8")

    log_payload = {"step": "test", "ok": True}
    ws.log_event("unit-test", log_payload)
    log_path = ws.resolve(ws.log_name)
    assert log_path.exists()
    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert entry["event"] == "unit-test"

    zip_path = ws.zip_outputs(data_dir, ws.resolve("exports/session.zip"))
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        assert "a.txt" in names
        assert "nested/b.txt" in names
