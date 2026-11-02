# Model Card – AQ Forecast PM2.5 (Bangalore)

> Full content will be written in Phase 7 once final metrics are available.
> Skeleton shown here to establish the structure.

## Model details

- **Model type:** XGBoost regressor
- **Task:** Forecast hourly PM2.5 (µg/m³) 24 hours ahead for Bangalore, India
- **Version:** See MLflow registry for current champion version
- **Developed by:** (your name / org)

## Intended use

- **Primary use:** Operational daily air quality forecast for Bangalore
- **Out-of-scope:** Other cities, other pollutants without re-training

## Data

- **Source:** Open-Meteo air quality API (CAMS model output) + Open-Meteo weather
- **⚠️ Important caveat:** PM2.5 values from Open-Meteo are **model-based
  (CAMS/Copernicus)**, not ground-station measurements. They represent an
  atmospheric model's estimate, not a direct sensor reading. Comparisons
  with CPCB ground-station data will show systematic biases.
- **Location:** Bangalore (lat 12.9716, lon 77.5946)
- **Temporal coverage:** Backfilled 180 days; updated daily

## Metrics

*To be filled in Phase 7 with final test-set numbers.*

| Metric | XGBoost | Persistence baseline | Seasonal-naive baseline |
|---|---|---|---|
| MAE (µg/m³) | TBD | TBD | TBD |
| RMSE (µg/m³) | TBD | TBD | TBD |
| sMAPE (%) | TBD | TBD | TBD |

## Known limitations and failure modes

- **Monsoon season (June–September):** Rainfall-driven PM2.5 drops may be
  poorly captured if the CAMS model underestimates washout.
- **Festival spikes (Diwali, Ugadi):** Firework-related PM2.5 spikes are not
  predicted by weather/AQ lag features alone; expect under-prediction.
- **Sensor outages:** Open-Meteo API outages cause gaps; the pipeline uses
  exponential back-off and alerts on validation failure.

## Ethical notes

- This is an environmental forecasting tool. Predictions carry uncertainty;
  they should not be the sole basis for health or policy decisions.
- Model-based CAMS data should not be used as a substitute for regulatory
  air quality monitoring.
