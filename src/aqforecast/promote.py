"""Promote a registered challenger only when it clears the MAE gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from aqforecast.config import load_config, load_params
from aqforecast.evaluate import _metrics


def promotion_decision(
    challenger_mae: float,
    champion_mae: float | None,
    baseline_mae: float,
    min_improvement_pct: float,
) -> tuple[bool, str]:
    """Apply the first-model baseline gate or subsequent champion improvement gate."""
    if champion_mae is None:
        accepted = challenger_mae < baseline_mae
        return accepted, (
            "first model beats both baselines"
            if accepted
            else "first model did not beat both baselines"
        )
    accepted = challenger_mae <= champion_mae * (1 - min_improvement_pct)
    return accepted, (
        "challenger meets configured MAE improvement"
        if accepted
        else "challenger failed configured MAE improvement"
    )


def champion_status_message(
    champion_version: str | None,
    challenger_version: str,
    promoted: bool,
    reason: str,
) -> str:
    """Explain which model remains available after the promotion decision."""
    if promoted:
        return f"Champion model version {challenger_version} is available."
    if champion_version is not None:
        return f"Champion model version {champion_version} remains active."
    return (
        "No champion yet: the first challenger was not promoted because "
        f"{reason}. The retrain completed, but serving predictions needs a champion."
    )


def promote(challenger_run_id: str) -> dict:
    """Compare champion and challenger on the same recent feature rows."""
    params = load_params()
    name = params["mlflow"]["registered_model_name"]
    client = mlflow.tracking.MlflowClient()
    challenger_versions = client.search_model_versions(
        filter_string=f"name='{name}' and run_id='{challenger_run_id}'"
    )
    if not challenger_versions:
        raise ValueError(f"No registered model found for run {challenger_run_id}")
    challenger_version = max(challenger_versions, key=lambda v: int(v.version)).version
    df = pd.read_parquet("data/processed/features/features.parquet")
    days = load_config()["promotion"]["eval_window_days"]
    recent = df[
        df.timestamp
        >= pd.to_datetime(df.timestamp.max(), utc=True) - pd.Timedelta(days=days)
    ]
    if len(recent) < 2:
        raise ValueError("Insufficient observations in the configured promotion window")
    challenger = mlflow.sklearn.load_model(f"models:/{name}/{challenger_version}")
    columns = [c for c in recent.columns if c not in {"target_pm2_5_24h", "timestamp"}]
    actual = recent["target_pm2_5_24h"].to_numpy()
    challenger_mae = _metrics(actual, challenger.predict(recent[columns]))
    champion_mae = None
    champion_version = None
    for version in client.search_model_versions(filter_string=f"name='{name}'"):
        if "champion" in (version.aliases or []):
            champion_version = version.version
            break
    if champion_version:
        champion = mlflow.sklearn.load_model(f"models:/{name}/{champion_version}")
        champion_mae = _metrics(actual, champion.predict(recent[columns]))["mae"]
    baseline_metrics = {
        col: _metrics(actual, recent[col].to_numpy())["mae"]
        for col in ("pm2_5_current", "pm2_5_lag_144")
    }
    baseline_mae = min(baseline_metrics.values())
    accepted, reason = promotion_decision(
        challenger_mae["mae"],
        champion_mae,
        baseline_mae,
        load_config()["promotion"]["min_improvement_pct"],
    )
    if accepted:
        client.set_registered_model_alias(name, "champion", challenger_version)
    decision = {
        "promoted": accepted,
        "champion_available": champion_version is not None or accepted,
        "status_message": champion_status_message(
            champion_version, challenger_version, accepted, reason
        ),
        "champion_mae": champion_mae,
        "challenger_mae": challenger_mae["mae"],
        "baseline_mae": baseline_mae,
        "challenger_version": challenger_version,
        "decision_reason": reason,
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/promotion_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    with mlflow.start_run():
        mlflow.log_params(
            {
                "challenger_version": challenger_version,
                "decision": "promoted" if accepted else "rejected",
            }
        )
        mlflow.log_metrics(
            {
                "challenger_mae": challenger_mae["mae"],
                "baseline_mae": baseline_mae,
                **({"champion_mae": champion_mae} if champion_mae is not None else {}),
            }
        )
    return decision


def rollback(version: int) -> None:
    """Point the model's champion alias at a previous registered version."""
    name = load_params()["mlflow"]["registered_model_name"]
    mlflow.MlflowClient().set_registered_model_alias(name, "champion", str(version))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run-id")
    group.add_argument("--rollback-version", type=int)
    args = parser.parse_args()
    if args.run_id:
        run_id = args.run_id
    elif args.rollback_version is None:
        run_id = (
            Path("reports/challenger_run_id.txt").read_text(encoding="utf-8").strip()
        )
    else:
        run_id = ""
    if run_id:
        print(json.dumps(promote(run_id), indent=2))
    else:
        rollback(args.rollback_version)


if __name__ == "__main__":
    main()
