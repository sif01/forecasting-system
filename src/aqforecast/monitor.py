"""
monitor.py – Data and prediction drift monitoring with Evidently.

Generates two reports on a configurable schedule (typically after each
daily retrain):
  1. Data drift report: reference = training window, current = last 7 days.
     Saved as reports/data_drift.html and reports/data_drift.json.
  2. Target/prediction drift report: compares live predictions vs actuals
     once actuals are available (24h lag).
     Saved as reports/prediction_drift.html and reports/prediction_drift.json.

Also maintains a rolling live MAE Prometheus gauge so Grafana can display
model performance over time.

Implemented in Phase 5.
"""


def generate_drift_report(reference_df, current_df) -> dict:  # type: ignore[type-arg]
    """Generate an Evidently data drift report.

    Args:
        reference_df: Training-window DataFrame (reference distribution).
        current_df: Recent production-window DataFrame (current distribution).

    Returns:
        dict with summary statistics including drift_share.
    """
    raise NotImplementedError("generate_drift_report() will be implemented in Phase 5.")


def main() -> None:
    """Entry point: load data, generate reports, update Prometheus gauge."""
    raise NotImplementedError("monitor.main() will be implemented in Phase 5.")


if __name__ == "__main__":
    main()
