"""
evaluate.py – Model evaluation and baseline comparison.

Computes two mandatory baselines on the same test window:
  (a) Persistence: predict pm2_5(t+24) = pm2_5(t)
  (b) Seasonal-naive: predict pm2_5(t+24) = pm2_5(t - 7*24)  [same hour, 7 days ago]

Every trained model must beat both baselines to be worth considering
for promotion. Results are logged to MLflow and saved to
reports/evaluation.json.

Implemented in Phase 2.
"""


def compute_baselines(df):  # type: ignore[no-untyped-def]
    """Compute persistence and seasonal-naive baseline metrics.

    Args:
        df: Feature DataFrame with target column.

    Returns:
        dict with keys 'persistence' and 'seasonal_naive', each containing
        {'mae': float, 'rmse': float, 'smape': float}.
    """
    raise NotImplementedError("compute_baselines() will be implemented in Phase 2.")


def evaluate_model(model, df):  # type: ignore[no-untyped-def]
    """Score a trained model on the test split.

    Args:
        model: Fitted XGBoost model (or any sklearn-compatible estimator).
        df: Feature DataFrame with target column.

    Returns:
        dict with keys 'mae', 'rmse', 'smape'.
    """
    raise NotImplementedError("evaluate_model() will be implemented in Phase 2.")


def main() -> None:
    """Entry point: load model and data, evaluate, log results."""
    raise NotImplementedError("evaluate.main() will be implemented in Phase 2.")


if __name__ == "__main__":
    main()

