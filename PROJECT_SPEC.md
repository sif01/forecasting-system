# PROJECT SPEC: Daily-Retrained Air Quality Forecasting System (Bangalore)

> **How to use this file (for the human):** Put this file in the root of an empty GitHub repo, open the repo in Google Antigravity, and give the Manager agent this prompt:
> *"Read PROJECT_SPEC.md fully. First produce an implementation plan for Phase 0 only and wait for my approval. After each phase, stop, summarize what you did, and wait for approval before starting the next phase."*

---

## 1. Instructions to the agent (read first)

1. Work **one phase at a time**. Do not start a phase until the human approves.
2. Before coding each phase, write a short plan (files to create, commands to run, acceptance checks).
3. Use **small, meaningful commits** with conventional commit messages (`feat:`, `fix:`, `ci:`, `docs:`, `test:`, `chore:`). Never commit large data files, secrets, or `.env` files.
4. Use **feature branches and pull requests**, never push directly to `main` after Phase 0. One branch per phase, named `phase-N-short-name`. Open a PR with a description and a checklist of acceptance criteria.
5. Every phase must end with its **acceptance checks passing**. Run them and show the output.
6. Prefer simple, readable code over clever code. The human must be able to explain every file. Add short comments explaining *why*, not *what*.
7. Pin all dependency versions. Fix random seeds. Keep all tunables in `configs/config.yaml`, not hard-coded.
8. If something is ambiguous or a required credential is missing, **ask the human** rather than guessing.
9. Never invent API fields. Check Open-Meteo's documentation or make a real test call to confirm field names.
10. Keep a running `docs/DECISIONS.md` log of non-obvious choices and why they were made.

---

## 2. Project goal

Build an end-to-end MLOps system that **forecasts Bangalore's PM2.5 for the next 24 hours** (hourly), using recent PM2.5 history plus weather features. The system retrains **automatically every day**, validates the new model against the current production model, **promotes it only if it is better**, serves predictions through an API, and monitors data and prediction drift.

The modeling is intentionally simple. **The pipeline is the deliverable.**

### Location and data
- Location: Bangalore, India. Latitude `12.9716`, longitude `77.5946`. Timezone `Asia/Kolkata`.
- Source: **Open-Meteo** (free, no API key).
  - Air quality API: `https://air-quality-api.open-meteo.com/v1/air-quality` (hourly `pm2_5`, `pm10`, and optionally `nitrogen_dioxide`, `ozone`; supports `past_days`).
  - Weather forecast API: `https://api.open-meteo.com/v1/forecast` (hourly `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `wind_direction_10m`, `surface_pressure`, `precipitation`; supports `past_days`).
  - Weather historical API: `https://archive-api.open-meteo.com/v1/archive` for backfilling.
- Caveat to document in the model card: Open-Meteo air quality values are **model-based (CAMS)**, not ground-station measurements.

### Target and metrics
- Target: `pm2_5` at horizon of 24 hours ahead (a single-step "same time tomorrow" forecast is acceptable for v1; multi-horizon is a stretch goal).
- Primary metric: **MAE**. Secondary: RMSE and sMAPE.
- Baselines that every model must be compared against: (a) persistence (value 24h ago), (b) a seasonal-naive baseline.

---

## 3. Tech stack (defaults, do not change without asking)

| Concern | Tool |
|---|---|
| Language | Python 3.11 |
| Dependency management | `pip` + `requirements.txt` (pinned) and `requirements-dev.txt` |
| Data versioning | DVC |
| Remote storage and tracking server | DagsHub (DVC remote + hosted MLflow). Human will provide credentials |
| Experiment tracking and registry | MLflow (use **model aliases** `champion` / `challenger`, not deprecated stages) |
| Pipeline definition | `dvc.yaml` with `params.yaml` |
| Modeling | XGBoost (primary), scikit-learn baselines |
| Data validation | Pandera |
| Serving | FastAPI + Uvicorn, Pydantic schemas |
| Drift monitoring | Evidently |
| Metrics and dashboards | `prometheus-client`, Prometheus, Grafana |
| Containers | Docker (multi-stage) + Docker Compose |
| CI/CD | GitHub Actions |
| Linting and typing | Ruff, Black, mypy (lenient) |
| Testing | pytest, pytest-cov |
| Deployment | Render (Docker web service) or Google Cloud Run. Ask the human which |
| Optional UI | Streamlit dashboard for forecast vs actual |

---

## 4. Target repo structure

```
.
├── .github/
│   └── workflows/
│       ├── ci.yml                 # lint, test, build on PR
│       ├── retrain.yml            # daily cron: ingest, validate, train, evaluate, promote
│       └── deploy.yml             # build, push, deploy on merge to main
├── configs/
│   └── config.yaml                # location, features, horizons, thresholds
├── data/
│   ├── raw/                       # DVC-tracked
│   └── processed/                 # DVC-tracked
├── docs/
│   ├── ARCHITECTURE.md            # includes a Mermaid diagram
│   ├── DECISIONS.md
│   ├── MODEL_CARD.md
│   └── RUNBOOK.md                 # how to rollback, retrain manually, debug
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.train
│   ├── prometheus.yml
│   └── grafana/                   # provisioned datasource + dashboard JSON
├── src/
│   └── aqforecast/
│       ├── __init__.py
│       ├── config.py              # loads config.yaml + params.yaml
│       ├── ingest.py              # fetch from Open-Meteo with retries
│       ├── validate.py            # Pandera schemas
│       ├── features.py            # lags, rolling stats, time features
│       ├── train.py               # train + log to MLflow
│       ├── evaluate.py            # compare candidate vs champion vs baselines
│       ├── promote.py             # promotion gate + alias update
│       ├── monitor.py             # Evidently drift reports
│       └── api/
│           ├── main.py            # FastAPI app
│           ├── schemas.py
│           └── model_loader.py    # loads champion from MLflow registry
├── tests/
│   ├── test_ingest.py
│   ├── test_validate.py
│   ├── test_features.py
│   ├── test_train.py
│   ├── test_promote.py
│   └── test_api.py
├── dvc.yaml
├── params.yaml
├── docker-compose.yml
├── Makefile
├── pyproject.toml                 # ruff, black, mypy, pytest config
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
├── .dvcignore
└── README.md
```

---

## 5. Phases

### Phase 0: Repo and CI skeleton
**Tasks**
- Initialize repo structure above (empty modules with docstrings are fine).
- Add `.gitignore` (Python, data, `.env`, MLflow artifacts, `__pycache__`), `.env.example`, `pyproject.toml`, `Makefile` with targets: `install`, `lint`, `test`, `ingest`, `train`, `serve`, `up`, `down`.
- Add `ci.yml`: on pull request run Ruff, Black `--check`, mypy, and pytest with coverage.
- Add a PR template and issue templates (bug, feature).
- Add a placeholder test so CI is green.
- Write a first README with project goal and setup instructions.

**Human setup (agent should remind me):** enable branch protection on `main` requiring the CI check and a PR before merge.

**Acceptance**
- `make lint` and `make test` pass locally.
- CI runs green on a PR.

### Phase 1: Data ingestion and validation
**Tasks**
- `ingest.py`: fetch hourly air quality and weather for a configurable date range, merge on timestamp, handle timezone (store in UTC, convert for display), implement retry with exponential backoff and a request timeout. Save to `data/raw/` as Parquet, partitioned or named by date range.
- Provide a **backfill mode** (e.g., last 180 days) and an **incremental mode** (last N days appended, de-duplicated by timestamp).
- `validate.py`: Pandera schema checks for column names and dtypes, `pm2_5` range (0 to a sane upper bound, configurable), no duplicate timestamps, no gaps larger than a configurable threshold, null fraction under a threshold. Fail loudly with a readable report.
- Initialize DVC, configure DagsHub remote, and track `data/raw/`.
- Unit tests with mocked HTTP responses (do not hit the real API in tests).

**Acceptance**
- `make ingest` produces a validated Parquet file.
- `dvc push` succeeds. `dvc pull` on a fresh clone restores the data.
- Tests cover success, retry, malformed response, and validation failure.

### Phase 2: Features, training, tracking
**Tasks**
- `features.py`: lag features (1, 2, 3, 6, 12, 24, 48 h), rolling mean/std (3, 6, 24 h), hour-of-day and day-of-week with cyclical encoding, weather features, and the 24h-ahead target. **No leakage**: all features must use only information available at forecast time. Add a test that proves this.
- `train.py`: time-based split (train / validation / test, never random). Train XGBoost with params from `params.yaml`. Log params, metrics (MAE, RMSE, sMAPE), feature importance plot, the data hash/DVC revision, and the git commit SHA to MLflow. Register the model in the MLflow registry.
- `evaluate.py`: compute baselines (persistence and seasonal-naive) on the same test window and log them. The model must be compared against them in the logged run.
- Define the pipeline in `dvc.yaml`: `ingest -> validate -> features -> train -> evaluate`, with `params.yaml` driving it. `dvc repro` must work end to end.

**Acceptance**
- `dvc repro` runs the full pipeline.
- A run appears in MLflow with params, metrics, artifacts, and a registered model version.
- Re-running with the same data and seed gives identical metrics.

### Phase 3: Promotion gate (champion / challenger)
**Tasks**
- `promote.py`: load the current `champion` alias model (if any) and the new `challenger`. Evaluate **both on the same most-recent window of data** (e.g., last 7 days, configurable). Promote the challenger only if it improves MAE by at least a configurable margin and passes data validation. The first model is promoted automatically after beating baselines.
- Log the decision (promoted / rejected, with metric values) to MLflow and write it to `reports/promotion_decision.json`.
- Implement **rollback**: a CLI command to repoint `champion` to a previous version.
- Unit tests for: first promotion, challenger better, challenger worse, and equal-within-margin.

**Acceptance**
- Tests pass for all four cases.
- Manual rollback works and is documented in `docs/RUNBOOK.md`.

### Phase 4: Serving API and Docker
**Tasks**
- FastAPI app with:
  - `GET /health` (liveness) and `GET /ready` (model loaded)
  - `GET /model-info` (champion version, training date, metrics)
  - `POST /predict` (accepts recent observations or fetches latest data itself; returns the 24h-ahead forecast with the model version used)
  - `GET /metrics` (Prometheus format: request count, latency histogram, prediction value distribution, model version gauge)
- Pydantic request/response schemas with validation and clear error messages.
- API loads the `champion` model from the MLflow registry at startup, and can reload it via an authenticated `POST /admin/reload` or on a timer.
- Multi-stage `Dockerfile.api` (small final image, non-root user, healthcheck).
- `docker-compose.yml` with services: `api`, `prometheus`, `grafana`, and optionally a local `mlflow` for offline development.
- API tests using FastAPI `TestClient` with a stubbed model.

**Acceptance**
- `docker compose up` brings up the API, and `curl /predict` returns a forecast.
- `/metrics` is scraped by Prometheus.
- API tests pass in CI.

### Phase 5: Monitoring
**Tasks**
- `monitor.py`: generate an Evidently **data drift** report (reference = training window, current = last 7 days) and a **target / prediction drift** report. Save HTML to `reports/` and a machine-readable summary to JSON.
- Log the logged-prediction vs actual comparison: once actuals arrive (24h later), compute rolling live MAE and expose it as a Prometheus gauge.
- Provision a Grafana dashboard with: request rate and latency, prediction vs actual over time, live MAE, drift status, and current model version.
- Define a **drift trigger** in config: if drift share exceeds a threshold, flag it in the retrain workflow output.

**Acceptance**
- Drift report generated from real data.
- Grafana dashboard loads automatically from provisioned files with no manual clicking.

### Phase 6: Continuous training and deployment
**Tasks**
- `retrain.yml`: scheduled with `cron` (daily, e.g., 02:00 IST, remembering GitHub cron is UTC) plus `workflow_dispatch` for manual runs. Steps: checkout, set up Python, restore cache, `dvc pull`, run incremental ingest, validate, `dvc repro`, run promotion gate, run drift report, `dvc push`, commit updated `dvc.lock` and data pointers back to a dedicated branch or open an automated PR, and upload reports as workflow artifacts.
- The workflow must **fail clearly** if validation fails, and must **not** promote on failure.
- All credentials come from GitHub Secrets. Document the exact secret names in README.
- `deploy.yml`: on merge to `main`, build and push the API image to a container registry (GitHub Container Registry by default), then deploy to the chosen platform (ask the human: Render or Cloud Run). Add a post-deploy smoke test hitting `/health` and `/predict`.
- Optional: a small Streamlit page showing latest forecast, history, and model version.

**Acceptance**
- A manual `workflow_dispatch` of `retrain.yml` completes and produces a promotion decision.
- A merge to `main` deploys, and the smoke test passes.
- The live URL returns a forecast.

### Phase 7: Documentation and polish
**Tasks**
- `docs/ARCHITECTURE.md` with a Mermaid diagram of the full flow (Open-Meteo -> ingest -> validate -> DVC -> train -> MLflow -> gate -> registry -> API -> Prometheus/Grafana -> drift -> retrain).
- `docs/MODEL_CARD.md`: intended use, data source and its limitations (CAMS model-based data), metrics, baselines, known failure modes (e.g., monsoon, festival-season spikes), and ethical notes.
- `docs/RUNBOOK.md`: how to retrain manually, roll back a model, debug a failed workflow, and rotate secrets.
- README with badges (CI status), quickstart, architecture image, and a screenshot of the Grafana dashboard and MLflow UI.
- Final check that a **fresh clone** can reproduce everything using only the README.

**Acceptance**
- A teammate can clone, follow the README, and run the pipeline and the API.

---

## 6. Stretch goals (only if all phases are done)
- Multi-horizon forecasts (1h to 24h) with one model per horizon or a multi-output model.
- Compare XGBoost vs Prophet vs a simple LSTM and log all in MLflow.
- Add OpenAQ ground-station data as an independent "truth" signal to validate Open-Meteo values.
- Shadow deployment: run the challenger alongside the champion and log both predictions without serving the challenger.
- Alerting: send a Slack or email message when a model is rejected or drift is detected.
- SHAP explanations on the `/predict` response.

---

## 7. Definition of done (whole project)
- [ ] All phases merged via PRs with passing CI and a clean, meaningful commit history
- [ ] `dvc repro` and `dvc pull` work from a fresh clone
- [ ] MLflow shows many runs, a model registry with version history, and a `champion` alias
- [ ] Daily retrain workflow has run successfully on schedule at least twice
- [ ] Promotion gate has demonstrably both promoted and rejected a model (keep logs as evidence)
- [ ] Docker Compose stack runs locally with one command
- [ ] API deployed and reachable at a public URL
- [ ] Grafana dashboard and Evidently reports working
- [ ] Test coverage of at least 70% on `src/`
- [ ] README, architecture diagram, model card, and runbook complete

---

## 8. Things the human must provide (agent: ask for these when needed)
1. A DagsHub account and repo (for the DVC remote and hosted MLflow), plus the access token
2. GitHub repo with Actions enabled, and the secrets configured
3. Choice of deployment target (Render or Cloud Run) and its credentials
4. Any course-specific requirements not covered here (list them and the agent will adapt the plan)
