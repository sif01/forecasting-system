"""
train.py – XGBoost model training with MLflow experiment tracking.

Responsibilities:
  - Load feature-engineered data from data/processed/features/.
  - Perform a chronological train/val/test split (never random).
  - Train XGBoost with hyperparameters from params.yaml.
  - Log params, metrics (MAE, RMSE, sMAPE), feature importance plot,
    DVC data hash, and git commit SHA to MLflow.
  - Register the trained model in the MLflow Model Registry as a new
    version of the configured registered model name.

Implemented in Phase 2.
"""


def train() -> None:
    """Train the XGBoost model and log everything to MLflow."""
    raise NotImplementedError("train() will be implemented in Phase 2.")


def main() -> None:
    """Entry point: load data, train, log, register."""
    raise NotImplementedError("train.main() will be implemented in Phase 2.")


if __name__ == "__main__":
    main()
