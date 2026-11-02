"""
promote.py – Champion / challenger promotion gate.

Logic:
  1. Load the current 'champion' alias model from the MLflow registry
     (if no champion exists, auto-promote the new model after it beats baselines).
  2. Load the new 'challenger' model.
  3. Evaluate both on the same most-recent window of data (configurable days).
  4. Promote challenger to 'champion' only if it improves MAE by at least
     the configured minimum improvement percentage.
  5. Log the decision (promoted / rejected, with metric values) to MLflow
     and write it to reports/promotion_decision.json.

Also provides a rollback() function to repoint the 'champion' alias to
a previous registered model version – documented in docs/RUNBOOK.md.

Implemented in Phase 3.
"""


def promote(challenger_run_id: str) -> dict:  # type: ignore[type-arg]
    """Run the promotion gate for a challenger model.

    Args:
        challenger_run_id: MLflow run ID that produced the challenger model.

    Returns:
        dict with keys 'promoted' (bool), 'champion_mae', 'challenger_mae',
        'decision_reason'.
    """
    raise NotImplementedError("promote() will be implemented in Phase 3.")


def rollback(version: int) -> None:
    """Repoint the 'champion' alias to the given registered model version.

    Args:
        version: Integer version number in the MLflow Model Registry to
                 roll back to.
    """
    raise NotImplementedError("rollback() will be implemented in Phase 3.")


def main() -> None:
    """Entry point: parse CLI args (--run-id, --rollback-version) and dispatch."""
    raise NotImplementedError("promote.main() will be implemented in Phase 3.")


if __name__ == "__main__":
    main()
