from __future__ import annotations

import io

from fastapi.testclient import TestClient
from server import api_main, auth, storage


def _auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_api_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OGUM_STORAGE_PATH", str(tmp_path / "storage"))
    auth.SECRET_KEY = "api-test-secret"

    storage.ensure_demo_run("demo")
    auth.upsert_user("admin", "123", role="admin")

    client = TestClient(api_main.app)
    with client:
        login_response = client.post(
            "/auth/login", json={"username": "admin", "password": "123"}
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        runs_response = client.get("/runs/list", headers=_auth_header(token))
        assert runs_response.status_code == 200
        runs = runs_response.json()
        assert isinstance(runs, list)

        upload_response = client.post(
            "/runs/upload",
            headers=_auth_header(token),
            files={"file": ("new-run.json", io.BytesIO(b"{}"), "application/json")},
            data={"description": "Test run"},
        )
        assert upload_response.status_code == 200

        jobs_response = client.get("/jobs/list", headers=_auth_header(token))
        assert jobs_response.status_code == 200

        compare_response = client.post(
            "/compare/run",
            json={
                "base_run": runs[0]["name"],
                "candidate_run": runs[0]["name"],
                "compare_type": "summary",
            },
            headers=_auth_header(token),
        )
        assert compare_response.status_code == 200
        assert compare_response.json()["status"] == "queued"

        sim_response = client.post(
            "/sim/import",
            json={"name": "sim-1", "description": "bundle"},
            headers=_auth_header(token),
        )
        assert sim_response.status_code == 200
        assert "artifact_path" in sim_response.json()
