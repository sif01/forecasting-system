"""Open-Meteo response parsing and retry behavior."""

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from aqforecast.ingest import _fetch, fetch_range, ingest


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


def test_incremental_ingest_backfills_when_raw_data_is_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {
        "ingestion": {
            "backfill_days": 180,
            "incremental_days": 3,
            "archive_lag_days": 5,
        }
    }
    fetch = Mock(return_value=pd.DataFrame({"timestamp": ["2025-11-25"]}))
    monkeypatch.setattr("aqforecast.ingest.fetch_range", fetch)

    ingest("incremental", config=config, end=datetime(2025, 12, 1, tzinfo=UTC))

    start, finish, _ = fetch.call_args.args
    elapsed_days = (
        datetime.fromisoformat(finish) - datetime.fromisoformat(start)
    ).days
    assert elapsed_days == 180
