"""
schemas.py – Pydantic request/response models for the forecast API.

All schemas use strict types and include field descriptions so the
auto-generated OpenAPI docs (FastAPI /docs) are self-explanatory.

Implemented in Phase 4.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = Field(..., examples=["ok"])


class PredictRequest(BaseModel):
    """Request body for POST /predict.

    Either supply recent_observations (last 48h of hourly pm2_5 readings)
    or leave it empty and the API will fetch the latest data from Open-Meteo.
    """

    recent_observations: list[Annotated[float, Field(ge=0, le=1000)]] | None = Field(
        default=None,
        min_length=169,
        description="Optional list of recent hourly PM2.5 readings (most recent last).",
    )


class PredictResponse(BaseModel):
    """Response for POST /predict."""

    forecast_pm2_5_24h: float = Field(
        ..., description="Predicted PM2.5 (µg/m³) 24 hours from now."
    )
    model_version: str = Field(..., description="MLflow registered model version used.")
    forecast_timestamp_utc: str = Field(
        ..., description="ISO-8601 UTC timestamp of the forecasted hour."
    )
