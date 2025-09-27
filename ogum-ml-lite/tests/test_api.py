from fastapi.testclient import TestClient

from ogum_lite.api.main import app


client = TestClient(app)


def test_prep_endpoint_returns_derivatives() -> None:
    payload = {
        "data": [
            {"time_s": 0.0, "temp_C": 25.0, "rho_rel": 0.50},
            {"time_s": 10.0, "temp_C": 30.0, "rho_rel": 0.55},
            {"time_s": 20.0, "temp_C": 35.0, "rho_rel": 0.62},
        ]
    }
    response = client.post("/prep", json=payload)
    assert response.status_code == 200
    rows = response.json()["rows"]
    assert len(rows) == 3
    assert any("dy_dt" in row for row in rows)


def test_root_endpoint_reports_status() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
