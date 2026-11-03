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
