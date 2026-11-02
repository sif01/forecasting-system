"""
test_ingest.py – Unit tests for aqforecast.ingest.

Tests (implemented in Phase 1):
  - Successful fetch returns expected columns and dtypes.
  - Retry logic fires on transient HTTP 5xx errors.
  - Malformed API response raises a clear exception.
  - Incremental mode de-duplicates overlapping timestamps.

All HTTP calls are mocked; this test suite never hits the real API.
"""

