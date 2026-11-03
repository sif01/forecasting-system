"""Leakage-safe lag, rolling, cyclical, and future-target construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aqforecast.config import load_config


def build_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Create a row at time t using observations through t and label at t+24."""
    cfg = config or load_config()
    data = df.sort_values("timestamp").copy()
    ts = pd.to_datetime(data["timestamp"], utc=True)
    local = ts.dt.tz_convert(cfg["location"]["timezone"])
    features = pd.DataFrame(index=data.index)
    features["pm2_5_current"] = data["pm2_5"]
    # The baseline uses last week's same-hour observation; keep it available for evaluation.
    features["pm2_5_lag_144"] = data["pm2_5"].shift(144)
    for lag in cfg["features"]["lags"]:
        features[f"pm2_5_lag_{lag}"] = data["pm2_5"].shift(lag)
    for window in cfg["features"]["rolling_windows"]:
        past = data["pm2_5"].shift(1).rolling(window, min_periods=window)
        features[f"pm2_5_mean_{window}"] = past.mean()
        features[f"pm2_5_std_{window}"] = past.std()
    hour = local.dt.hour + local.dt.minute / 60
    dow = local.dt.dayofweek
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    features["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    for col in [*cfg["features"]["weather_cols"], *cfg["features"]["aq_cols"]]:
        if col in data:
            features[col] = data[col]
    horizon = cfg["horizon"]
    features["target_pm2_5_24h"] = data["pm2_5"].shift(-horizon)
    features["timestamp"] = ts
    features = features.dropna().reset_index(drop=True)
    if features.empty:
        raise ValueError(
            "Not enough observations to build configured lag and horizon features"
        )
    return features


def main() -> None:
    source = Path("data/processed/validated/observations.parquet")
    df = build_features(pd.read_parquet(source))
    dest = Path("data/processed/features")
    dest.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest / "features.parquet", index=False)
    print(f"Built {len(df)} feature rows")


if __name__ == "__main__":
    main()
