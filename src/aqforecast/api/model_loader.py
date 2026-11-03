"""Load the current champion model and expose stable inference metadata."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import mlflow
import mlflow.pyfunc

from aqforecast.config import load_params


class ModelLoader:
    """Small in-memory holder for the currently loaded registry model."""

    def __init__(self) -> None:
        self._model = None
        self._version: str | None = None
        self._run_id: str | None = None
        self._training_date: str | None = None
        self._metrics: dict[str, float] = {}

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def version(self) -> str | None:
        return self._version

    @property
    def metadata(self) -> dict:
        return {
            "version": self._version,
            "training_date": self._training_date,
            "metrics": self._metrics,
            "loaded": self.is_ready,
        }

    def load(self) -> None:
        """Load champion alias, supporting local model paths for offline use."""
        uri = os.getenv("AQ_MODEL_URI")
        if uri:
            self._model = mlflow.pyfunc.load_model(uri)
            self._version = "local"
            return
        name = load_params()["mlflow"]["registered_model_name"]
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_model_version_by_alias(name, "champion")
        self._model = mlflow.pyfunc.load_model(f"models:/{name}@champion")
        self._version, self._run_id = model_version.version, model_version.run_id
        run = client.get_run(model_version.run_id)
        self._training_date = datetime.fromtimestamp(
            run.info.start_time / 1000, UTC
        ).isoformat()
        self._metrics = {key: float(value) for key, value in run.data.metrics.items()}

    def predict(self, features: Any) -> float:
        model = self._model
        if model is None:
            raise RuntimeError("No champion model is loaded")
        return float(model.predict(features)[0])


model_loader = ModelLoader()
