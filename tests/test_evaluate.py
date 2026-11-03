"""Evaluation helpers score models and baselines on one chronological test block."""

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

from aqforecast import evaluate as evaluate_module
from aqforecast.evaluate import compute_baselines, evaluate_model


def _features() -> pd.DataFrame:
    values = np.arange(100, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=100, freq="h", tz="UTC"),
            "pm2_5_current": values,
            "pm2_5_lag_144": values - 2,
            "weather": values / 10,
            "target_pm2_5_24h": values + 1,
        }
    )


def test_baselines_share_chronological_test_block():
    metrics = compute_baselines(_features())

    assert set(metrics) == {"persistence", "seasonal_naive"}
    assert metrics["persistence"]["mae"] == 1
    assert metrics["seasonal_naive"]["mae"] == 3


def test_evaluate_model_excludes_target_and_timestamp():
    model = Mock()
    model.predict.return_value = np.arange(85, 100, dtype=float) + 1

    metrics = evaluate_model(model, _features())

    assert metrics["mae"] == 0
    passed_features = model.predict.call_args.args[0]
    assert "timestamp" not in passed_features
    assert "target_pm2_5_24h" not in passed_features


def test_main_logs_model_and_baselines(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "challenger_run_id.txt").write_text("run-1")
    frame = _features()
    model = Mock()
    model.predict.return_value = frame.iloc[85:]["target_pm2_5_24h"].to_numpy()
    client_run = SimpleNamespace(version="2")
    monkeypatch.setattr(evaluate_module.pd, "read_parquet", lambda _: frame)
    monkeypatch.setattr(
        evaluate_module,
        "load_params",
        lambda: {
            "split": {"train_frac": 0.7, "val_frac": 0.15},
            "mlflow": {"registered_model_name": "aqforecast"},
        },
    )
    monkeypatch.setattr(
        evaluate_module.mlflow, "search_model_versions", lambda _: [client_run]
    )
    monkeypatch.setattr(evaluate_module.mlflow.sklearn, "load_model", lambda _: model)
    monkeypatch.setattr(evaluate_module.mlflow, "start_run", lambda **_: nullcontext())
    monkeypatch.setattr(evaluate_module.mlflow, "log_metrics", Mock())

    evaluate_module.main()

    result = json.loads((tmp_path / "reports" / "evaluation.json").read_text())
    assert result["model"]["mae"] == 0
    assert set(result["baselines"]) == {"persistence", "seasonal_naive"}
