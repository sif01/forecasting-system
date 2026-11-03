"""Model registry loading and local inference holder behavior."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from aqforecast.api.model_loader import ModelLoader


def test_load_local_model_and_predict(monkeypatch):
    model = Mock()
    model.predict.return_value = np.array([12.5])
    loader = ModelLoader()
    monkeypatch.setenv("AQ_MODEL_URI", "models:/local/1")
    monkeypatch.setattr(
        "aqforecast.api.model_loader.mlflow.pyfunc.load_model", lambda uri: model
    )

    loader.load()

    assert loader.is_ready
    assert loader.version == "local"
    assert loader.predict([1, 2]) == 12.5
    assert loader.metadata["loaded"] is True


def test_load_champion_metadata(monkeypatch):
    loader = ModelLoader()
    monkeypatch.delenv("AQ_MODEL_URI", raising=False)
    version = SimpleNamespace(version="7", run_id="run-7")
    run = SimpleNamespace(
        info=SimpleNamespace(start_time=1_767_000_000_000),
        data=SimpleNamespace(metrics={"validation_mae": 4}),
    )
    client = Mock()
    client.get_model_version_by_alias.return_value = version
    client.get_run.return_value = run
    monkeypatch.setattr(
        "aqforecast.api.model_loader.mlflow.tracking.MlflowClient", lambda: client
    )
    monkeypatch.setattr(
        "aqforecast.api.model_loader.mlflow.pyfunc.load_model", lambda uri: Mock()
    )

    loader.load()

    assert loader.version == "7"
    assert loader.metadata["metrics"] == {"validation_mae": 4.0}
    assert loader.metadata["training_date"] is not None


def test_predict_without_a_loaded_model_fails_clearly():
    with pytest.raises(RuntimeError, match="No champion model"):
        ModelLoader().predict([])
