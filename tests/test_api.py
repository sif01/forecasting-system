"""FastAPI liveness, readiness, prediction, and metrics endpoint tests."""

from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from aqforecast.api.main import app
from aqforecast.api.model_loader import model_loader


def test_health_and_readiness(monkeypatch):
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    monkeypatch.setattr(model_loader, "_model", None)
    assert client.get("/ready").status_code == 503


def test_predict_with_stubbed_model_and_live_covariates(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(model_loader, "_model", object())
    monkeypatch.setattr(model_loader, "_version", "test")
    monkeypatch.setattr(model_loader, "predict", lambda features: 12.5)
    with patch("requests.get") as get:
        now = __import__("pandas").Timestamp.now(tz="UTC").floor("h")
        times = [
            (now - __import__("pandas").Timedelta(hours=i)).isoformat()
            for i in range(168, -1, -1)
        ]
        aq = {
            "hourly": {
                "time": times,
                "pm2_5": [10.0] * 169,
                "pm10": [20.0] * 169,
                "nitrogen_dioxide": [1.0] * 169,
                "ozone": [2.0] * 169,
            }
        }
        wxcols = [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "surface_pressure",
            "precipitation",
        ]
        wx = {"hourly": {"time": times, **{col: [1.0] * 169 for col in wxcols}}}
        responses = [Mock(), Mock()]
        responses[0].json.return_value = wx
        responses[1].json.return_value = aq
        for response in responses:
            response.raise_for_status.return_value = None
        get.side_effect = responses
        result = client.post("/predict", json={"recent_observations": [10.0] * 169})
    assert result.status_code == 200
    assert result.json()["forecast_pm2_5_24h"] == 12.5
    assert result.json()["model_version"] == "test"


def test_metrics_and_invalid_observations(monkeypatch):
    client = TestClient(app)
    assert client.get("/metrics").status_code == 200
    monkeypatch.setattr(model_loader, "_model", object())
    assert (
        client.post("/predict", json={"recent_observations": [-1] * 169}).status_code
        == 422
    )


def test_model_info_and_authenticated_reload(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(model_loader, "_version", "3")
    monkeypatch.setattr(model_loader, "_model", object())
    assert client.get("/model-info").json()["version"] == "3"

    monkeypatch.setenv("AQ_ADMIN_TOKEN", "test-secret")
    monkeypatch.setattr(model_loader, "load", Mock())
    assert client.post("/admin/reload").status_code == 401
    response = client.post("/admin/reload", headers={"X-Admin-Token": "test-secret"})
    assert response.status_code == 200
    assert response.json()["status"] == "reloaded"


def test_reload_reports_model_load_failure(monkeypatch):
    client = TestClient(app)
    monkeypatch.setenv("AQ_ADMIN_TOKEN", "test-secret")
    monkeypatch.setattr(model_loader, "load", Mock(side_effect=RuntimeError("offline")))

    response = client.post("/admin/reload", headers={"X-Admin-Token": "test-secret"})

    assert response.status_code == 503
    assert "offline" in response.json()["detail"]


def test_predict_requires_loaded_model_and_maps_source_errors(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(model_loader, "_model", None)
    assert (
        client.post("/predict", json={"recent_observations": [10.0] * 169}).status_code
        == 503
    )

    monkeypatch.setattr(model_loader, "_model", object())
    monkeypatch.setattr(
        "requests.get", Mock(side_effect=RuntimeError("source offline"))
    )
    response = client.post("/predict", json={})
    assert response.status_code == 502
    assert "source offline" in response.json()["detail"]


def test_metrics_loads_live_monitoring_gauges(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "live_metrics.json").write_text(
        '{"live_mae": 4.2, "drift_share": 0.5}', encoding="utf-8"
    )

    response = TestClient(app).get("/metrics")

    assert response.status_code == 200
    assert "aq_live_mae 4.2" in response.text
    assert "aq_drift_share 0.5" in response.text
