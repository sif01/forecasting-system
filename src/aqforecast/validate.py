"""Validate ingested hourly observations before they enter training."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

from aqforecast.config import load_config


def schema(config: dict | None = None) -> DataFrameSchema:
    cfg = config or load_config()
    maximum = cfg["validation"]["pm2_5_max"]
    null_max = cfg["ingestion"]["max_null_fraction"]
    return DataFrameSchema(
        {
            "timestamp": Column(pa.DateTime, nullable=False, unique=True),
            "pm2_5": Column(
                float,
                Check.in_range(cfg["validation"]["pm2_5_min"], maximum),
                nullable=True,
            ),
            "pm10": Column(float, nullable=True),
            "nitrogen_dioxide": Column(float, nullable=True),
            "ozone": Column(float, nullable=True),
            "temperature_2m": Column(float, nullable=True),
            "relative_humidity_2m": Column(float, nullable=True),
            "wind_speed_10m": Column(float, nullable=True),
            "wind_direction_10m": Column(float, nullable=True),
            "surface_pressure": Column(float, nullable=True),
            "precipitation": Column(float, nullable=True),
        },
        checks=Check(
            lambda df: df.isna().mean().max() <= null_max,
            error="null fraction exceeds configured threshold",
        ),
        strict=True,
        coerce=True,
    )


def validate_raw(df: pd.DataFrame, config: dict | None = None) -> None:
    """Raise a Pandera error with all schema violations and reject excessive gaps."""
    cfg = config or load_config()
    validated = schema(cfg).validate(df, lazy=True)
    timestamps = validated["timestamp"].sort_values()
    gap_limit = pd.Timedelta(hours=cfg["ingestion"]["max_gap_hours"])
    if not timestamps.empty:
        previous = timestamps.iloc[0]
        for current in timestamps.iloc[1:]:
            if current - previous > gap_limit:
                raise ValueError(
                    f"timestamp gap exceeds configured maximum of {gap_limit}"
                )
            previous = current


def main() -> None:
    raw = sorted(Path("data/raw").glob("*.parquet"))
    if not raw:
        raise FileNotFoundError("No raw Parquet data found in data/raw")
    df = pd.concat([pd.read_parquet(path) for path in raw], ignore_index=True)
    validate_raw(df)
    dest = Path("data/processed/validated")
    dest.mkdir(parents=True, exist_ok=True)
    df.drop_duplicates("timestamp", keep="last").sort_values("timestamp").to_parquet(
        dest / "observations.parquet", index=False
    )
    print(f"Validated {len(df)} observations")


if __name__ == "__main__":
    main()
