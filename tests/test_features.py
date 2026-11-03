"""Feature engineering tests including a direct leakage check."""

import pandas as pd

from aqforecast.features import build_features


def test_target_is_exactly_24_hours_ahead_and_features_are_past_only():
    source = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=240, freq="h", tz="UTC"),
            "pm2_5": range(240),
        }
    )
    for col in (
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "surface_pressure",
        "precipitation",
    ):
        source[col] = 20.0
    output = build_features(source)
    row = output.loc[output.timestamp == source.timestamp.iloc[180]].iloc[0]
    assert row.target_pm2_5_24h == 204
    assert row.pm2_5_lag_1 == 179
    assert row.pm2_5_current == 180
    changed = source.copy()
    changed.loc[changed.index > 180, "pm2_5"] = -999
    changed_output = build_features(changed)
    changed_row = changed_output.loc[
        changed_output.timestamp == source.timestamp.iloc[180]
    ].iloc[0]
    assert row.pm2_5_lag_1 == changed_row.pm2_5_lag_1
