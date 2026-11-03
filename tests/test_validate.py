"""Schema and gap validation tests."""

import pandas as pd
import pandera as pa
import pytest

from aqforecast.validate import validate_raw


def valid_frame():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC"),
            "pm2_5": [1.0, 2.0, 3.0, 4.0, 5.0],
            "pm10": [2.0] * 5,
            "nitrogen_dioxide": [1.0] * 5,
            "ozone": [1.0] * 5,
            "temperature_2m": [20.0] * 5,
            "relative_humidity_2m": [40.0] * 5,
            "wind_speed_10m": [3.0] * 5,
            "wind_direction_10m": [180.0] * 5,
            "surface_pressure": [1000.0] * 5,
            "precipitation": [0.0] * 5,
        }
    )


def test_valid_data_passes():
    validate_raw(valid_frame())


def test_out_of_range_target_fails():
    frame = valid_frame()
    frame.loc[0, "pm2_5"] = 2000
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw(frame)


def test_duplicate_timestamp_fails():
    frame = valid_frame()
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw(frame)


def test_large_gap_fails():
    frame = valid_frame()
    frame.loc[3:, "timestamp"] += pd.Timedelta(hours=5)
    with pytest.raises(ValueError, match="gap"):
        validate_raw(frame)
