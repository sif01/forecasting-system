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

## Setup order

1. Create a DagsHub token and copy the repository's MLflow tracking URI. Add `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`, and `MLFLOW_TRACKING_URI` as repository Actions secrets. The retraining preflight fails if any of these are missing.
2. Generate a strong `API_RELOAD_SECRET` and add it as a repository Actions secret. GitHub provides `GITHUB_TOKEN` automatically; do not create it yourself. Enable Actions to create pull requests in **Settings → Actions → General**.
3. Create the Render Docker web service from `ghcr.io/sif01/forecasting-system:latest`. If the GHCR image is private, configure a Render registry credential with package read access. Set Render's `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD` environment variables. Set Render's `AQ_ADMIN_TOKEN` to exactly the same value as GitHub's `API_RELOAD_SECRET`.
4. Copy the Render deploy hook and public base URL into GitHub Actions secrets named `RENDER_DEPLOY_HOOK` and `RENDER_URL`. Add the repository Actions variable `DEPLOY_TARGET=render`; the deploy workflow now fails clearly if this is missing or invalid.
5. For the Cloud Run alternative only, use `GCP_SERVICE_ACCOUNT_KEY` as a GitHub Actions secret and `CLOUD_RUN_SERVICE` plus `CLOUD_RUN_REGION` as Actions variables, then set `DEPLOY_TARGET=cloud-run` instead of `render`.
6. Run **Actions → Daily Retrain → Run workflow** on `main`. Review the promotion decision and any `No champion yet` warning before expecting the API to serve predictions. A first candidate is promoted only if it beats both configured naive baselines; otherwise retraining remains successful but serving stays unavailable until a champion is promoted.

Set the repository variable `DEPLOY_TARGET` to `render`. In Render, create a Docker web service that uses the GHCR image published by `deploy.yml`, then add `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, and `AQ_ADMIN_TOKEN` as runtime environment variables. Make the GHCR package readable by the Render service. The deploy workflow is not live until this service, its deploy hook, and its public URL are configured.

