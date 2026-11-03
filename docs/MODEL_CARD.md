# Model card: Bangalore PM2.5 forecast

## Intended use

Forecast hourly PM2.5 concentration 24 hours ahead for Bangalore, India, as an educational forecasting and MLOps demonstration. It is not designed for personal health decisions, regulatory reporting, or emergency response.

## Data and method

Air quality is sourced from Open-Meteo's Air Quality API and weather from its Historical Weather API. The PM2.5 values are CAMS model-based estimates, not ground-station measurements. Features use recent PM2.5 history, weather covariates, rolling statistics, and cyclical local time. XGBoost is evaluated on chronological data splits against persistence and weekly seasonal-naive baselines. See configuration for the exact features and promotion threshold.

## Metrics

MAE is the primary metric; RMSE and sMAPE are also logged. Current run-specific results are stored under `reports/` and in MLflow. There are no verified production metrics until the pipeline has been run on the target DagsHub tracking server.

## Known limitations and failure modes

- CAMS estimates can differ from local sensor readings and may smooth local events.
- Monsoon conditions, festival-season firecrackers, construction, traffic changes, and abrupt weather shifts can cause spikes the model has not learned.
- A 24-hour single-step forecast does not describe uncertainty or intermediate hourly values.
- Historical weather reanalysis is not identical to weather forecasts available at prediction time, creating a potential training-serving mismatch.
- Missing API coverage, stale data, or insufficient history should stop prediction rather than be silently imputed.

## Ethical notes

Air-quality forecasts can affect health choices. Users should consult official local monitoring and public-health guidance. The API must not be represented as authoritative measurement or medical advice. Data source attribution to Open-Meteo and CAMS is required.
