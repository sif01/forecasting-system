# Runbook

Operational guide for manual operation and recovery.

---

## Manual retrain

```bash
# 1. Activate your Python 3.11 environment
source .venv/bin/activate

# 2. Configure DVC credentials locally (never commit config.local)
dvc remote modify origin --local access_key_id "$DAGSHUB_TOKEN"
dvc remote modify origin --local secret_access_key "$DAGSHUB_TOKEN"

# 3. Restore tracked data when this checkout has a DVC lock file
if [ -f dvc.lock ]; then dvc pull; fi

# 4. Run the full pipeline (ingest → validate → features → train → evaluate → promote → monitor)
dvc repro
```

On a fresh clone with no raw Parquet data, the pipeline's incremental ingest stage automatically performs the initial backfill. `dvc repro` invokes that stage directly, so a separate manual ingest is unnecessary. Set `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD` in the environment to use hosted MLflow. The default tracking store is local.

---

## Rolling back the champion model

If a newly promoted model is performing poorly in production, roll back to
a previous version:

```bash
python -m aqforecast.promote --rollback-version <version_number>
```

This repoints the `champion` alias in the MLflow Model Registry to the
specified version. The API will pick up the rollback on the next
`POST /admin/reload` call or restart.

Inspect registered versions in MLflow before rollback, then reload the API with `POST /admin/reload` and the `X-Admin-Token` header, or restart the service.

---

## Debugging a failed retrain workflow

1. Open the failed GitHub Actions run → check the step that failed.
2. Common causes:
   - **Validation failure:** Open-Meteo returned malformed data or a gap
     larger than `max_gap_hours`. Check the error report in the workflow
     artifact. Do not promote until data is clean.
   - **MLflow connection error:** Check `MLFLOW_TRACKING_URI` and
     `DAGSHUB_TOKEN` in GitHub Secrets.
   - **DVC push/pull failure:** Check `DAGSHUB_TOKEN` and DVC remote URL.

Download the workflow's reports artifact and inspect `promotion_decision.json` and the Evidently JSON/HTML reports. A validation failure must be resolved before retraining or promotion; do not bypass the validation stage.

---

## Rotating secrets

1. Generate a new DagsHub token in your DagsHub account settings.
2. Update `DAGSHUB_TOKEN` in GitHub repository Secrets
   (Settings → Secrets and variables → Actions).
3. Update your local `.env` file.
4. Trigger a manual `workflow_dispatch` of `retrain.yml` to verify.

## Required GitHub secrets

- `DAGSHUB_TOKEN`: DagsHub token used for DVC and MLflow.
- `DAGSHUB_USERNAME`: DagsHub owner name (`sif01` for this repository).
- `MLFLOW_TRACKING_URI`: hosted MLflow URI for this repository.
- `API_RELOAD_SECRET`: secret protecting the model reload endpoint.
- `RENDER_DEPLOY_HOOK`: deploy hook URL for the configured Render Docker service.
- `RENDER_URL`: public base URL used by the post-deploy health and forecast checks.

Set the repository variable `DEPLOY_TARGET` to `render`. In Render, create a Docker web service that uses the GHCR image published by `deploy.yml`, then add `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, and `AQ_ADMIN_TOKEN` as runtime environment variables. Make the GHCR package readable by the Render service. The deploy workflow is not live until this service, its deploy hook, and its public URL are configured.

