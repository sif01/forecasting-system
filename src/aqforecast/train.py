"""Chronological XGBoost training with MLflow tracking and registration."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from aqforecast.config import load_params


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    return float(
        np.mean(
            np.divide(
                2 * np.abs(y_pred - y_true),
                denom,
                out=np.zeros_like(denom, dtype=float),
                where=denom != 0,
            )
        )
        * 100
    )


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows chronologically into train, validation, and test blocks."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    p = load_params()["split"]
    a, b = int(len(df) * p["train_frac"]), int(
        len(df) * (p["train_frac"] + p["val_frac"])
    )
    if a == 0 or b <= a or b >= len(df):
        raise ValueError("Insufficient feature rows for configured chronological split")
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def train() -> str:
    """Fit, evaluate on held-out validation data, and register a challenger."""
    data = pd.read_parquet("data/processed/features/features.parquet")
    train_df, val_df, _ = split_data(data)
    target = "target_pm2_5_24h"
    excluded = {target, "timestamp"}
    columns = [c for c in data.columns if c not in excluded]
    model = XGBRegressor(**load_params()["xgboost"])
    model.fit(
        train_df[columns],
        train_df[target],
        eval_set=[(val_df[columns], val_df[target])],
        verbose=False,
    )
    pred = model.predict(val_df[columns])
    metrics = {
        "validation_mae": float(mean_absolute_error(val_df[target], pred)),
        "validation_rmse": float(np.sqrt(mean_squared_error(val_df[target], pred))),
        "validation_smape": smape(val_df[target].to_numpy(), pred),
    }
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "unknown"
    mlflow.set_experiment(load_params()["mlflow"]["experiment_name"])
    with mlflow.start_run() as run:
        mlflow.log_params(load_params()["xgboost"])
        mlflow.log_metrics(metrics)
        mlflow.set_tag("git_commit", sha)
        data_path = Path("data/processed/features/features.parquet")
        mlflow.set_tag(
            "data_sha256", hashlib.sha256(data_path.read_bytes()).hexdigest()
        )
        mlflow.log_param("feature_columns", json.dumps(columns))
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=load_params()["mlflow"]["registered_model_name"],
        )
        run_id = run.info.run_id
        versions = mlflow.search_model_versions(
            filter_string=f"name='{load_params()['mlflow']['registered_model_name']}' and run_id='{run_id}'"
        )
        if versions:
            mlflow.MlflowClient().set_registered_model_alias(
                load_params()["mlflow"]["registered_model_name"],
                "challenger",
                max(versions, key=lambda v: int(v.version)).version,
            )
        fig, axis = plt.subplots(figsize=(9, 6))
        importance = (
            pd.Series(model.feature_importances_, index=columns).sort_values().tail(20)
        )
        importance.plot.barh(ax=axis)
        axis.set_title("XGBoost feature importance")
        fig.tight_layout()
        plot_path = Path("reports/feature_importance.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path)
        plt.close(fig)
        mlflow.log_artifact(str(plot_path))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    Path("reports/challenger_run_id.txt").write_text(run_id, encoding="utf-8")
    return run_id


def main() -> None:
    print(f"Registered challenger run {train()}")


if __name__ == "__main__":
    main()
