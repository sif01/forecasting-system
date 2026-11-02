"""
test_api.py – Unit tests for the FastAPI application.

Tests (implemented in Phase 4):
  - GET /health returns 200 and {"status": "ok"}.
  - GET /ready returns 503 when model is not loaded, 200 when loaded.
  - POST /predict with a stubbed model returns a valid PredictResponse.
  - POST /predict with invalid input returns 422 (Pydantic validation error).
  - GET /metrics returns Prometheus-format text.
  - POST /admin/reload with wrong secret returns 401.

All tests use FastAPI's TestClient with the real model loader replaced
by a stub that always returns a fixed prediction value.
"""
