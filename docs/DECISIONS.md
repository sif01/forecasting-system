# Decision Log

This file records non-obvious technical choices made during development,
following the rule: explain *why*, not just *what*.

---

## 2026-11-02 – Phase 0

### D-001: Open-Meteo CAMS data over ground-station measurements

**Decision:** Use Open-Meteo's air quality API (backed by the Copernicus
Atmosphere Monitoring Service model) as the primary PM2.5 source rather
than trying to ingest from ground-station networks (e.g., OpenAQ, CPCB).

**Rationale:**
- Open-Meteo is free, requires no API key, and provides a stable, consistent
  REST API with `past_days` support — ideal for automated daily retraining.
- Ground-station data from CPCB (India's Central Pollution Control Board)
  requires manual downloads or unstable scraping; OpenAQ coverage for
  Bangalore is patchy and frequently missing.
- The trade-off is that CAMS values are *model-based*, not direct measurements.
  This is prominently documented in the model card (docs/MODEL_CARD.md).

**Implication:** The model predicts CAMS-modelled PM2.5, not sensor PM2.5.
Comparison with ground truth requires a separate data source (stretch goal).

---

### D-002: Single 24h-ahead target (not multi-horizon)

**Decision:** Predict a single value — PM2.5 at t+24 — rather than a
full 24-step sequence.

**Rationale:** The spec calls this acceptable for v1. Multi-horizon is a
stretch goal. A single-output XGBoost model is simpler, faster to iterate,
and easier to explain. The architecture does not preclude adding horizons later.

---

### D-003: Chronological train/val/test split (no random shuffling)

**Decision:** Time-series data is always split chronologically.

**Rationale:** Random shuffling would leak future information into training
(the model sees tomorrow's data while "training" on yesterday). This is a
fundamental MLOps hygiene rule for time-series forecasting, not a tunable.

---

### D-004: MLflow model aliases (`champion` / `challenger`) over deprecated stages

**Decision:** Use the MLflow Model Registry's *alias* feature, not the old
stage labels (`Staging`, `Production`).

**Rationale:** MLflow deprecated stage-based promotion in v2.x in favour of
aliases, which are more flexible and explicit. This future-proofs the registry
integration and matches the spec requirement.

---

### D-005: Pinned dependency versions in requirements.txt

**Decision:** Every package version is pinned to an exact version (`==`).

**Rationale:** Reproducibility is a core project goal. Unpinned ranges
(`>=`) allow silent upgrades that break experiments or change metric values.
Versions will be deliberately updated when needed, not automatically.

---

## 2026-11-03 – Pipeline implementation

### D-006: Chunk long air-quality backfills

**Decision:** Split requests into 90-day windows and join each with the corresponding weather archive interval.

**Rationale:** Open-Meteo's Air Quality API limits `past_days` to 92, while the project backfill default is 180 days. Keeping each request below that limit avoids silently requesting unsupported history.

### D-007: Keep the latest PM2.5 value and prior-week value as explicit baseline features

**Decision:** Store `pm2_5_current` and `pm2_5_lag_144` in the feature frame for baseline scoring.

**Rationale:** The persistence baseline needs the observed value at forecast time. For a t+24 target, weekly seasonal-naive needs the same-hour observation seven days before that target, which is t-144. Both are valid historical covariates and do not expose the future target.

### D-008: DagsHub object storage credentials remain local

**Decision:** Track only the DVC S3-compatible endpoint in `.dvc/config`; supply access and secret keys through local DVC config or CI secrets.

**Rationale:** DagsHub's DVC integration uses an S3-compatible endpoint, and credentials must not be committed.

