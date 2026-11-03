"""Compare held-out model predictions with required naive baselines."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from aqforecast.config import load_params
from aqforecast.train import smape, split_data


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "smape": smape(y, pred),
    }


def compute_baselines(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Score persistence and weekly seasonal-naive on the chronological test block."""
    _, _, test = split_data(df)
    target = test["target_pm2_5_24h"].to_numpy()
    return {
        "persistence": _metrics(target, test["pm2_5_current"].to_numpy()),
        "seasonal_naive": _metrics(target, test["pm2_5_lag_144"].to_numpy()),
    }


def evaluate_model(model, df: pd.DataFrame) -> dict[str, float]:
    """Score an sklearn-compatible model on the same held-out test block."""
    _, _, test = split_data(df)
    columns = [c for c in df.columns if c not in {"target_pm2_5_24h", "timestamp"}]
    return _metrics(test["target_pm2_5_24h"].to_numpy(), model.predict(test[columns]))


def main() -> None:
    df = pd.read_parquet("data/processed/features/features.parquet")
    params = load_params()
    model_name = params["mlflow"]["registered_model_name"]
    versions = mlflow.search_model_versions(filter_string=f"name='{model_name}'")
    if not versions:
        raise RuntimeError("Training did not register a model version")
    latest = max(versions, key=lambda v: int(v.version))
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{latest.version}")
    model_metrics = evaluate_model(model, df)
    baseline_metrics = compute_baselines(df)
    result = {"model": model_metrics, "baselines": baseline_metrics}
    run_id = Path("reports/challenger_run_id.txt").read_text(encoding="utf-8").strip()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"test_{k}": v for k, v in model_metrics.items()})
        mlflow.log_metrics(
            {
                f"baseline_{name}_{k}": v
                for name, values in baseline_metrics.items()
                for k, v in values.items()
            }
        )
    Path("reports").mkdir(exist_ok=True)
    Path("reports/evaluation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
