"""
validate.py – Pandera-based data validation for raw ingested data.

Checks performed:
  - Expected columns present with correct dtypes.
  - pm2_5 values within configured [min, max] range.
  - No duplicate timestamps.
  - No gaps larger than max_gap_hours between consecutive hourly readings.
  - Null fraction per column below max_null_fraction threshold.

Fails loudly with a human-readable SchemaError report so pipeline
failures are easy to diagnose.

Implemented in Phase 1.
"""


def validate_raw(df) -> None:  # type: ignore[no-untyped-def]
    """Validate a raw ingested DataFrame against the Pandera schema.

    Args:
        df: pandas DataFrame produced by ingest.py.

    Raises:
        pandera.errors.SchemaError: if any check fails.
    """
    raise NotImplementedError("validate_raw() will be implemented in Phase 1.")


def main() -> None:
    """Entry point: load the latest raw Parquet and validate it."""
    raise NotImplementedError("validate.main() will be implemented in Phase 1.")


if __name__ == "__main__":
    main()

