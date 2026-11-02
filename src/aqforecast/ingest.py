"""
ingest.py – Fetch hourly air quality and weather data from Open-Meteo.

Responsibilities:
  - Backfill mode: fetch the last N days of history and save to data/raw/.
  - Incremental mode: append the last M days, de-duplicating by timestamp.
  - Retry with exponential back-off on transient HTTP errors.
  - All timestamps stored in UTC; display conversion happens downstream.
  - Output: Parquet file at data/raw/aq_weather_<start>_<end>.parquet

Implemented in Phase 1.
"""


def main() -> None:
    """Entry point: parse CLI args and dispatch to backfill or incremental mode."""
    raise NotImplementedError("ingest.main() will be implemented in Phase 1.")


if __name__ == "__main__":
    main()
