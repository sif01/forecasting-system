"""
test_validate.py – Unit tests for aqforecast.validate.

Tests (implemented in Phase 1):
  - Valid DataFrame passes without error.
  - Out-of-range pm2_5 values raise SchemaError.
  - Duplicate timestamps raise SchemaError.
  - Gaps larger than max_gap_hours raise SchemaError.
  - High null fraction raises SchemaError.
"""

