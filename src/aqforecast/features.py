"""
features.py – Feature engineering for the PM2.5 forecast model.

Features computed (all using only information available at forecast time
to prevent data leakage):
  - Lag features: pm2_5 at t-1, t-2, t-3, t-6, t-12, t-24, t-48.
  - Rolling statistics: mean and std over 3h, 6h, 24h windows.
  - Cyclical time encodings: hour-of-day and day-of-week (sin/cos pairs).
  - Weather features: as configured in configs/config.yaml.
  - Target: pm2_5 at t+24 (24-hour-ahead forecast).

Implemented in Phase 2.
"""


def build_features(df):  # type: ignore[no-untyped-def]
    """Transform a validated raw DataFrame into a model-ready feature matrix.

    Args:
        df: Validated pandas DataFrame from validate.py.

    Returns:
        Feature DataFrame with target column 'target_pm2_5_24h'.

    Note:
        All lag/rolling operations use closed='left' or shift() to ensure
        no future information leaks into the features.
    """
    raise NotImplementedError("build_features() will be implemented in Phase 2.")


def main() -> None:
    """Entry point: load validated data, build features, save to processed/."""
    raise NotImplementedError("features.main() will be implemented in Phase 2.")


if __name__ == "__main__":
    main()
