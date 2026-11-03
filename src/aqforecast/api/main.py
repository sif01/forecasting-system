"""FastAPI service for 24-hour PM2.5 forecasts and Prometheus metrics."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from aqforecast.api.model_loader import model_loader
from aqforecast.api.schemas import PredictRequest, PredictResponse
from aqforecast.config import load_config

app = FastAPI(
    title="AQ Forecast API",
    description="24-hour PM2.5 forecast for Bangalore.",
    version="1.0.0",
)
REQUESTS = Counter("aq_api_requests_total", "API requests", ["endpoint", "status"])
LATENCY = Histogram("aq_api_request_duration_seconds", "Request latency", ["endpoint"])
PREDICTION = Histogram(
    "aq_prediction_pm25",
    "Forecast PM2.5 values",
    buckets=(0, 5, 10, 20, 35, 50, 100, 250, 500, float("inf")),
)
MODEL_VERSION = Gauge("aq_model_version", "Loaded model version", ["version"])
LIVE_MAE = Gauge("aq_live_mae", "Rolling live forecast mean absolute error")
DRIFT_SHARE = Gauge("aq_drift_share", "Share of monitored features with drift")


@app.middleware("http")
async def observe_requests(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    path = (
        request.url.path
        if request.url.path
        in {"/health", "/ready", "/model-info", "/predict", "/metrics", "/admin/reload"}
        else "other"
    )
    REQUESTS.labels(path, str(response.status_code // 100) + "xx").inc()
    LATENCY.labels(path).observe(time.perf_counter() - start)
    return response


@app.on_event("startup")
def startup() -> None:
    if os.getenv("AQ_MODEL_URI") or os.getenv("MLFLOW_TRACKING_URI"):
        try:
            model_loader.load()
            MODEL_VERSION.labels(model_loader.version or "unknown").set(1)
        except Exception:
            # Liveness stays available; readiness reports missing registry/model access.
            pass


@app.get("/health", tags=["ops"])
async def health() -> dict:
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
async def ready() -> dict:
    if not model_loader.is_ready:
        raise HTTPException(status_code=503, detail="Champion model is not loaded")
    return {"status": "ready"}


@app.get("/model-info", tags=["ops"])
async def model_info() -> dict:
    return model_loader.metadata


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    try:
        if not model_loader.is_ready:
            raise HTTPException(status_code=503, detail="Champion model is not loaded")
        values = request.recent_observations
        import requests

        cfg = load_config()
        if values is None:
            payload = requests.get(
                cfg["api"]["air_quality_url"],
                params={
                    "latitude": cfg["location"]["latitude"],
                    "longitude": cfg["location"]["longitude"],
                    "hourly": "pm2_5",
                    "past_days": 8,
                    "forecast_days": 1,
                    "timezone": "UTC",
                },
                timeout=cfg["api"]["request_timeout"],
            )
            payload.raise_for_status()
            hourly = payload.json().get("hourly", {})
            times = pd.to_datetime(hourly.get("time", []), utc=True)
            latest = pd.Timestamp.now(tz="UTC").floor("h")
            idx = [i for i, stamp in enumerate(times) if stamp <= latest]
            if not idx:
                raise HTTPException(
                    status_code=502,
                    detail="Open-Meteo has no recent PM2.5 observations",
                )
            values = hourly["pm2_5"][: idx[-1] + 1]
        if len(values) < 145:
            raise HTTPException(
                status_code=422, detail="Provide at least 145 hourly PM2.5 values"
            )
        if any(v < 0 or v > 1000 for v in values):
            raise HTTPException(
                status_code=422, detail="PM2.5 values must be between 0 and 1000"
            )
        series = pd.Series(values, dtype=float)
        cfg = load_config()
        row: dict[str, float] = {
            "pm2_5_current": float(series.iloc[-1]),
            "pm2_5_lag_144": float(series.iloc[-145]),
        }
        for lag in cfg["features"]["lags"]:
            row[f"pm2_5_lag_{lag}"] = float(series.iloc[-1 - lag])
        for window in cfg["features"]["rolling_windows"]:
            chunk = series.iloc[-1 - window : -1]
            row[f"pm2_5_mean_{window}"] = float(chunk.mean())
            row[f"pm2_5_std_{window}"] = float(chunk.std())
        local = datetime.now(UTC).astimezone(ZoneInfo(cfg["location"]["timezone"]))
        row.update(
            hour_sin=float(np.sin(2 * np.pi * local.hour / 24)),
            hour_cos=float(np.cos(2 * np.pi * local.hour / 24)),
            dow_sin=float(np.sin(2 * np.pi * local.weekday() / 7)),
            dow_cos=float(np.cos(2 * np.pi * local.weekday() / 7)),
        )
        # Future weather covariates are fetched for the forecast hour from the live API.
        weather = requests.get(
            cfg["api"]["weather_forecast_url"],
            params={
                "latitude": cfg["location"]["latitude"],
                "longitude": cfg["location"]["longitude"],
                "hourly": ",".join(cfg["features"]["weather_cols"]),
                "past_days": 3,
                "forecast_days": 1,
                "timezone": "UTC",
            },
            timeout=cfg["api"]["request_timeout"],
        )
        weather.raise_for_status()
        weather_hourly = weather.json().get("hourly", {})
        weather_times = pd.to_datetime(weather_hourly.get("time", []), utc=True)
        weather_idx = [
            i
            for i, stamp in enumerate(weather_times)
            if stamp <= pd.Timestamp.now(tz="UTC").floor("h")
        ]
        if not weather_idx:
            raise HTTPException(
                status_code=502, detail="Open-Meteo has no recent weather covariates"
            )
        for col in cfg["features"]["weather_cols"]:
            row[col] = float(weather_hourly[col][weather_idx[-1]])
        aq_cols = cfg["features"]["aq_cols"]
        if aq_cols:
            aq_payload = requests.get(
                cfg["api"]["air_quality_url"],
                params={
                    "latitude": cfg["location"]["latitude"],
                    "longitude": cfg["location"]["longitude"],
                    "hourly": ",".join(aq_cols),
                    "past_days": 3,
                    "forecast_days": 1,
                    "timezone": "UTC",
                },
                timeout=cfg["api"]["request_timeout"],
            )
            aq_payload.raise_for_status()
            aq_hourly = aq_payload.json().get("hourly", {})
            aq_times = pd.to_datetime(aq_hourly.get("time", []), utc=True)
            aq_idx = [
                i
                for i, stamp in enumerate(aq_times)
                if stamp <= pd.Timestamp.now(tz="UTC").floor("h")
            ]
            if not aq_idx:
                raise HTTPException(
                    status_code=502,
                    detail="Open-Meteo has no recent pollutant covariates",
                )
            for col in aq_cols:
                row[col] = float(aq_hourly[col][aq_idx[-1]])
        value = model_loader.predict(pd.DataFrame([row]))
        PREDICTION.observe(value)
        return PredictResponse(
            forecast_pm2_5_24h=value,
            model_version=model_loader.version or "unknown",
            forecast_timestamp_utc=(
                datetime.now(UTC) + timedelta(hours=24)
            ).isoformat(),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Unable to produce forecast: {exc}"
        ) from exc


@app.post("/admin/reload", tags=["ops"])
async def reload_model(x_admin_token: str | None = Header(default=None)) -> dict:
    expected = os.getenv("AQ_ADMIN_TOKEN")
    if not expected or x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Invalid admin token")
    try:
        model_loader.load()
        MODEL_VERSION.labels(model_loader.version or "unknown").set(1)
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Unable to load champion: {exc}"
        ) from exc
    return {"status": "reloaded", "version": model_loader.version}


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    try:
        with open("reports/live_metrics.json", encoding="utf-8") as handle:
            status = json.load(handle)
        if status.get("live_mae") is not None:
            LIVE_MAE.set(status["live_mae"])
        if status.get("drift_share") is not None:
            DRIFT_SHARE.set(status["drift_share"])
    except (FileNotFoundError, ValueError):
        pass
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
