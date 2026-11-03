# Air Quality Forecasting System – Bangalore PM2.5

![CI](https://github.com/sif01/forecasting-system/actions/workflows/ci.yml/badge.svg)

An end-to-end MLOps pipeline that **forecasts Bangalore's PM2.5 for the next
24 hours**, retrains automatically every day, validates new models against a
promotion gate, serves predictions through a REST API, and monitors data and
prediction drift.

> **The modeling is intentionally simple. The pipeline is the deliverable.**

---

## Architecture overview

```
Open-Meteo API (free, no key required)
  → ingest    fetch + retry + UTC storage
  → validate  Pandera schema checks
  → features  lags, rolling stats, cyclical time encoding
  → train     XGBoost, logged to MLflow
  → evaluate  vs. persistence and seasonal-naive baselines
  → promote   champion / challenger gate (MLflow aliases)
  → API       FastAPI + Uvicorn  →  POST /predict
  → monitor   Evidently drift reports + Prometheus / Grafana
  → retrain   GitHub Actions daily cron (02:00 IST)
```

Full Mermaid architecture diagram: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## Quickstart

### Prerequisites

- Python 3.11
- Docker & Docker Compose (for the full stack)
- A [DagsHub](https://dagshub.com) account (free) for DVC remote + MLflow

### 1 – Clone and install

```bash
git clone https://github.com/<owner>/<repo>.git
cd forecasting-system

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

make install
```

### 2 – Configure credentials

```bash
cp .env.example .env
# Edit .env with your DagsHub username, token, and MLflow tracking URI

# Configure DagsHub object storage credentials locally; these stay out of Git.
dvc remote modify origin --local access_key_id "$DAGSHUB_TOKEN"
dvc remote modify origin --local secret_access_key "$DAGSHUB_TOKEN"
```

### 3 – Run the pipeline

```bash
# Backfill 180 days of data and run the full pipeline
make ingest
make train
```

`make ingest` backfills data and validates it. `make train` runs `dvc repro`, including incremental ingestion, validation, feature creation, training, baseline evaluation, promotion, and drift reports. The first MLflow run uses the local MLflow store unless `MLFLOW_TRACKING_URI` is set.

### 4 – Start the API and observability stack

```bash
make up
# API:        http://localhost:8000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin / admin)
```

### 5 – Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{}'
```

---

## Required GitHub Secrets

Configure these in **Settings → Secrets and variables → Actions**:

| Secret name | Description |
|---|---|
| `DAGSHUB_USERNAME` | Your DagsHub username |
| `DAGSHUB_TOKEN` | DagsHub personal access token |
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/<user>/<repo>.mlflow` |
| `API_RELOAD_SECRET` | Random string for `POST /admin/reload` |

For the scheduled retraining workflow, DVC uses the DagsHub token as both its S3-compatible access and secret key. Set the repository variable `DEPLOY_TARGET` to `render` or `cloud-run` for deployment. Render additionally needs `RENDER_DEPLOY_HOOK` and `RENDER_URL` secrets. Cloud Run needs `GCP_SERVICE_ACCOUNT_KEY`, and the variables `CLOUD_RUN_SERVICE` and `CLOUD_RUN_REGION`.

The deployment workflow builds and publishes `ghcr.io/sif01/forecasting-system:latest`. Configure the selected hosting service to use this image and provide the MLflow tracking and DagsHub credentials as runtime environment variables.

---

## Project structure

See [`PROJECT_SPEC.md`](PROJECT_SPEC.md) for the full target layout.

Key directories:

| Path | Purpose |
|---|---|
| `src/aqforecast/` | Python package – all pipeline logic |
| `configs/config.yaml` | Location, features, thresholds (edit here, not in code) |
| `params.yaml` | XGBoost hyperparameters tracked by DVC |
| `dvc.yaml` | Pipeline stage definitions |
| `tests/` | pytest test suite |
| `docs/` | Architecture, decisions, model card, runbook |
| `docker/` | Dockerfiles, Prometheus config, Grafana provisioning |

---

## Docs

- [Architecture](docs/ARCHITECTURE.md)
- [Decision log](docs/DECISIONS.md)
- [Model card](docs/MODEL_CARD.md)
- [Runbook](docs/RUNBOOK.md)

---

## Branch protection reminder (human action required)

Enable branch protection on `main`:
1. Go to **Settings → Branches → Add rule**
2. Branch name pattern: `main`
3. ✅ Require a pull request before merging
4. ✅ Require status checks to pass: select **CI / Lint, type-check, and test**
5. ✅ Require branches to be up to date before merging

