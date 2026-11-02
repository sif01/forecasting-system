"""
model_loader.py – Load and cache the champion model from the MLflow registry.

Responsibilities:
  - On startup, fetch the model tagged with alias 'champion' from the
    configured MLflow Model Registry.
  - Cache the loaded model in memory so predictions are fast.
  - Expose a reload() function for the /admin/reload endpoint.
  - Expose model metadata (version, training date, metrics) for /model-info.

Why a separate module: keeps model lifecycle concerns isolated from
the FastAPI routing layer, and makes the loader easy to stub in tests.

Implemented in Phase 4.
"""


class ModelLoader:
    """Singleton-style loader for the MLflow champion model."""

    def __init__(self) -> None:
        self._model = None
        self._version: str | None = None

    @property
    def is_ready(self) -> bool:
        """True if a model has been loaded successfully."""
        return self._model is not None

    def load(self) -> None:
        """Load (or reload) the champion model from the MLflow registry."""
        raise NotImplementedError("ModelLoader.load() will be implemented in Phase 4.")

    def predict(self, features):  # type: ignore[no-untyped-def]
        """Run inference with the loaded model.

        Args:
            features: numpy array or pandas DataFrame of shape (1, n_features).

        Returns:
            float: predicted PM2.5 value.
        """
        raise NotImplementedError(
            "ModelLoader.predict() will be implemented in Phase 4."
        )


# Module-level singleton used by the FastAPI app
model_loader = ModelLoader()
