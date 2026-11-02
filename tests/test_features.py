"""
test_features.py – Unit tests for aqforecast.features.

Tests (implemented in Phase 2):
  - Output DataFrame has all expected lag and rolling columns.
  - No leakage: features at time t use only data from t-1 and earlier.
    (Verified by checking that the feature matrix, when shifted back by
    horizon hours, never correlates perfectly with the target.)
  - Cyclical encodings are in [-1, 1] and have the correct period.
  - Target column equals pm2_5 shifted back by 24 h.
"""

