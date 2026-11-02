# Architecture

> Detailed Mermaid diagram and component descriptions will be added in Phase 7.

## Overview

The system follows a daily MLOps loop:

```
Open-Meteo API
  → ingest.py     (fetch + retry)
  → validate.py   (Pandera schema checks)
  → DVC tracking  (data versioning)
  → features.py   (lag/rolling/cyclical features)
  → train.py      (XGBoost + MLflow logging)
  → evaluate.py   (baselines: persistence, seasonal-naive)
  → promote.py    (champion/challenger gate)
  → MLflow Registry (champion alias)
  → FastAPI        (POST /predict)
  → Prometheus / Grafana (observability)
  → monitor.py    (Evidently drift reports)
  → retrain.yml   (GitHub Actions daily cron → loop)
```

The full Mermaid diagram will be produced in Phase 7 once all components exist.
