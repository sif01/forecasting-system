"""Download and persist hourly Bangalore air quality and weather observations."""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from aqforecast.config import load_config

AIR_FIELDS = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone"]
WEATHER_FIELDS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
]


def _fetch(url: str, params: dict, timeout: int, retries: int) -> dict:
    """Retry transient transport and server errors with bounded exponential delay."""
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable retry state")


def fetch_range(
    start_date: str, end_date: str, config: dict | None = None
) -> pd.DataFrame:
    """Fetch the two Open-Meteo hourly series and join them on UTC timestamp."""
    cfg = config or load_config()
    api, loc = cfg["api"], cfg["location"]
    common = {
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "timezone": "UTC",
    }

    def frame(payload: dict) -> pd.DataFrame:
        hourly = payload.get("hourly")
        if not isinstance(hourly, dict) or "time" not in hourly:
            raise ValueError("Open-Meteo response is missing hourly.time")
        out = pd.DataFrame(hourly)
        out["timestamp"] = pd.to_datetime(out.pop("time"), utc=True, errors="raise")
        return out

    # Air quality supports at most 92 past days per call; chunk longer backfills.
    start, finish = (
        datetime.fromisoformat(start_date).date(),
        datetime.fromisoformat(end_date).date(),
    )
    air_parts, weather_parts = [], []
    cursor = start
    while cursor <= finish:
        chunk_end = min(cursor + timedelta(days=90), finish)
        interval = {
            **common,
            "start_date": cursor.isoformat(),
            "end_date": chunk_end.isoformat(),
        }
        aq = _fetch(
            api["air_quality_url"],
            {**interval, "hourly": ",".join(AIR_FIELDS)},
            api["request_timeout"],
            api["max_retries"],
        )
        wx = _fetch(
            api["weather_archive_url"],
            {**interval, "hourly": ",".join(WEATHER_FIELDS)},
            api["request_timeout"],
            api["max_retries"],
        )
        air_parts.append(frame(aq))
        weather_parts.append(frame(wx))
        cursor = chunk_end + timedelta(days=1)
    result = pd.concat(air_parts, ignore_index=True).merge(
        pd.concat(weather_parts, ignore_index=True),
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )
    if result.empty:
        raise ValueError("Open-Meteo returned no overlapping hourly observations")
    return result.sort_values("timestamp").reset_index(drop=True)


def ingest(
    mode: str = "backfill", config: dict | None = None, end: datetime | None = None
) -> Path:
    """Download backfill or recent incremental range and deduplicate persisted rows."""
    cfg = config or load_config()
    now = end or datetime.now(UTC)
    days = (
        cfg["ingestion"]["backfill_days"]
        if mode == "backfill"
        else cfg["ingestion"]["incremental_days"]
    )
    # ERA5 archive delivery has a configured lag; older weather data is stable.
    finish_date = (now - timedelta(days=cfg["ingestion"]["archive_lag_days"])).date()
    start = (finish_date - timedelta(days=days)).isoformat()
    finish = finish_date.isoformat()
    fresh = fetch_range(start, finish, cfg)
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"aq_weather_{start}_{finish}.parquet"
    if mode == "incremental" and path.exists():
        fresh = pd.concat([pd.read_parquet(path), fresh]).drop_duplicates(
            "timestamp", keep="last"
        )
    fresh.to_parquet(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["backfill", "incremental"], default="backfill"
    )
    args = parser.parse_args()
    print(ingest(args.mode))


if __name__ == "__main__":
    main()
