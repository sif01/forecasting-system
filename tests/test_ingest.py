"""Open-Meteo response parsing and retry behavior."""

from unittest.mock import Mock, patch

import pytest
import requests

from aqforecast.ingest import _fetch, fetch_range


def test_fetch_range_joins_hourly_series():
    times = ["2025-01-01T00:00", "2025-01-01T01:00"]
    aq = {
        "hourly": {
            "time": times,
            "pm2_5": [1, 2],
            "pm10": [2, 3],
            "nitrogen_dioxide": [1, 1],
            "ozone": [2, 2],
        }
    }
    wx = {
        "hourly": {
            "time": times,
            "temperature_2m": [20, 21],
            "relative_humidity_2m": [40, 41],
            "wind_speed_10m": [3, 4],
            "wind_direction_10m": [180, 180],
            "surface_pressure": [1000, 1000],
            "precipitation": [0, 0],
        }
    }
    responses = [
        Mock(status_code=200, json=lambda: aq),
        Mock(status_code=200, json=lambda: wx),
    ]
    for response in responses:
        response.raise_for_status.return_value = None
    with patch("aqforecast.ingest.requests.get", side_effect=responses):
        frame = fetch_range("2025-01-01", "2025-01-01")
    assert len(frame) == 2
    assert str(frame.timestamp.dtype) == "datetime64[ns, UTC]"
    assert frame.pm2_5.tolist() == [1, 2]


def test_fetch_retries_transient_failure():
    failure = requests.ConnectionError("temporary")
    good = Mock()
    good.raise_for_status.return_value = None
    good.json.return_value = {"ok": True}
    with (
        patch("aqforecast.ingest.requests.get", side_effect=[failure, good]),
        patch("aqforecast.ingest.time.sleep"),
    ):
        assert _fetch("https://example.test", {}, 1, 2) == {"ok": True}


def test_malformed_payload_fails_clearly():
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"hourly": {"pm2_5": [1]}}
    with (
        patch("aqforecast.ingest.requests.get", return_value=response),
        pytest.raises(ValueError, match="hourly.time"),
    ):
        fetch_range("2025-01-01", "2025-01-01")
