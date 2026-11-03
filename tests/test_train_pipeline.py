"""Training orchestration logs and registers a reproducible challenger run."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

from aqforecast import train as train_module


def test_train_logs_metrics_metadata_and_registered_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    feature_path = tmp_path / "data" / "processed" / "features" / "features.parquet"
    feature_path.parent.mkdir(parents=True)
    feature_path.write_bytes(b"fixture")
    values = np.arange(100, dtype=float)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=100, freq="h", tz="UTC"),
            "pm2_5_current": values,
            "weather": values / 10,
            "target_pm2_5_24h": values + 1,
        }
    )
    model = Mock()
    model.predict.return_value = np.zeros(15)
    model.feature_importances_ = np.array([0.6, 0.4])
    run = SimpleNamespace(info=SimpleNamespace(run_id="run-1"))
    client = Mock()
    monkeypatch.setattr(train_module.pd, "read_parquet", lambda _: frame)
    monkeypatch.setattr(train_module, "XGBRegressor", lambda **_: model)
    monkeypatch.setattr(
        train_module.subprocess, "check_output", lambda *_, **__: "abc123"
    )
    monkeypatch.setattr(train_module.mlflow, "set_experiment", Mock())
    monkeypatch.setattr(train_module.mlflow, "start_run", lambda: nullcontext(run))
    monkeypatch.setattr(train_module.mlflow, "log_params", Mock())
    monkeypatch.setattr(train_module.mlflow, "log_metrics", Mock())
    monkeypatch.setattr(train_module.mlflow, "set_tag", Mock())
    monkeypatch.setattr(train_module.mlflow, "log_param", Mock())
    monkeypatch.setattr(train_module.mlflow, "log_artifact", Mock())
    monkeypatch.setattr(train_module.mlflow.sklearn, "log_model", Mock())
    monkeypatch.setattr(
        train_module.mlflow,
        "search_model_versions",
        lambda **_: [SimpleNamespace(version="2")],
    )
    monkeypatch.setattr(train_module.mlflow, "MlflowClient", lambda: client)

    run_id = train_module.train()

    assert run_id == "run-1"
    assert client.set_registered_model_alias.called
    assert (tmp_path / "reports" / "metrics.json").is_file()
    assert (tmp_path / "reports" / "challenger_run_id.txt").read_text() == "run-1"
    assert train_module.mlflow.set_tag.call_args_list[0].args == (
        "git_commit",
        "abc123",
    )
