"""
test_promote.py – Unit tests for aqforecast.promote.

Tests (implemented in Phase 3):
  - First promotion: no existing champion → challenger promoted automatically
    after beating baselines.
  - Challenger better: MAE improvement exceeds threshold → promoted.
  - Challenger worse: MAE does not improve → rejected, champion unchanged.
  - Equal within margin: improvement is below threshold → rejected.
  - Rollback: champion alias is repointed to the specified version.
"""
