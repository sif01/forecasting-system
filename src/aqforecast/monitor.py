"""Generate Evidently drift reports and update rolling live forecast error."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow.pyfunc
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from aqforecast.config import load_config, load_params


def generate_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    """Compare matching numeric feature distributions and save HTML plus JSON."""
    columns = [
        c
        for c in reference_df.select_dtypes(include="number").columns
        if c in current_df
    ]
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df[columns], current_data=current_df[columns])
    result = report.as_dict()
    summary = result.get("metrics", [{}])[0].get("result", {})
    payload = {
        "drift_share": summary.get("drift_share"),
        "number_of_drifted_columns": summary.get("number_of_drifted_columns"),
        "number_of_columns": summary.get("number_of_columns"),
        "drift_detected": summary.get("dataset_drift"),
    }
    Path("reports").mkdir(exist_ok=True)
    report.save_html("reports/data_drift.html")
    Path("reports/data_drift.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    features = pd.read_parquet("data/processed/features/features.parquet")
    window = load_config()["monitoring"]["live_mae_window_days"]
    latest = pd.to_datetime(features.timestamp.max(), utc=True)
    current = features[features.timestamp >= latest - pd.Timedelta(days=7)]
    reference = features[features.timestamp < latest - pd.Timedelta(days=7)]
    if reference.empty or current.empty:
        raise ValueError(
            "Need both a reference and recent data window for drift monitoring"
        )
    summary = generate_drift_report(reference, current)
    name = load_params()["mlflow"]["registered_model_name"]
    model = mlflow.pyfunc.load_model(f"models:/{name}@champion")
    feature_cols = [
        c for c in current.columns if c not in {"target_pm2_5_24h", "timestamp"}
    ]
    reference = reference.dropna(subset=["target_pm2_5_24h"]).copy()
    current = current.dropna(subset=["target_pm2_5_24h"]).copy()
    reference["prediction_pm2_5"] = model.predict(reference[feature_cols])
    current["prediction_pm2_5"] = model.predict(current[feature_cols])
    target_report = Report(metrics=[DataDriftPreset()])
    target_report.run(
        reference_data=reference[["target_pm2_5_24h", "prediction_pm2_5"]],
        current_data=current[["target_pm2_5_24h", "prediction_pm2_5"]],
    )
    target_report.save_html("reports/prediction_drift.html")
    prediction_summary = (
        target_report.as_dict().get("metrics", [{}])[0].get("result", {})
    )
    Path("reports/prediction_drift.json").write_text(
        json.dumps(
            {
                "drift_share": prediction_summary.get("drift_share"),
                "dataset_drift": prediction_summary.get("dataset_drift"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary["triggered"] = bool(
        summary["drift_share"] is not None
        and summary["drift_share"]
        > load_config()["monitoring"]["drift_share_threshold"]
    )
    Path("reports/data_drift.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    live = features[features.timestamp >= latest - pd.Timedelta(days=window)].dropna(
        subset=["target_pm2_5_24h"]
    )
    live_predictions = model.predict(live[feature_cols]) if not live.empty else []
    summary["live_mae"] = (
        float(abs(live["target_pm2_5_24h"].to_numpy() - live_predictions).mean())
        if not live.empty
        else None
    )
    Path("reports/live_metrics.json").write_text(
        json.dumps(
            {
                "live_mae": summary["live_mae"],
                "drift_share": summary["drift_share"],
                "drift_triggered": summary["triggered"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    Path("reports/data_drift.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
