"""
main.py – FastAPI application entry point.

Endpoints (implemented in Phase 4):
  GET  /health       – liveness probe (always 200 if process is up)
  GET  /ready        – readiness probe (200 only if champion model is loaded)
  GET  /model-info   – champion version, training date, test-set metrics
  POST /predict      – 24h-ahead PM2.5 forecast; accepts recent observations
                       or fetches latest data from Open-Meteo itself
  GET  /metrics      – Prometheus metrics (request count, latency, prediction
                       value histogram, model version gauge)
  POST /admin/reload – reload champion model from registry (requires secret)

Implemented in Phase 4.
"""

from fastapi import FastAPI

app = FastAPI(
    title="AQ Forecast API",
    description="24-hour PM2.5 forecast for Bangalore.",
    version="0.1.0",
)


@app.get("/health", tags=["ops"])
async def health() -> dict:  # type: ignore[type-arg]
    """Liveness probe – returns 200 if the process is running."""
    return {"status": "ok"}

