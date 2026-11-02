"""
test_train.py – Unit tests for aqforecast.train.

Tests (implemented in Phase 2):
  - Training on a small synthetic dataset completes without error.
  - Fixed random seed produces identical metrics on repeated runs.
  - An MLflow run is created with required params and metrics logged.
  - The model is registered in the MLflow registry.
"""
