# Runbook

Operational guide for common tasks. Detailed procedures will be added
phase-by-phase.

---

## Manual retrain

```bash
# 1. Activate your virtual environment
source .venv/bin/activate

# 2. Pull latest data pointers from DVC remote
dvc pull

# 3. Run incremental ingest (last 3 days)
python -m aqforecast.ingest --mode incremental

# 4. Run the full pipeline (validate → features → train → evaluate)
dvc repro

# 5. Run the promotion gate
python -m aqforecast.promote --run-id <mlflow_run_id>
```

*Full procedure added in Phase 6.*

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

*Detailed procedure and version lookup instructions added in Phase 3.*

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

*Expanded in Phase 6.*

---

## Rotating secrets

1. Generate a new DagsHub token in your DagsHub account settings.
2. Update `DAGSHUB_TOKEN` in GitHub repository Secrets
   (Settings → Secrets and variables → Actions).
3. Update your local `.env` file.
4. Trigger a manual `workflow_dispatch` of `retrain.yml` to verify.

