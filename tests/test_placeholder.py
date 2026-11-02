"""
test_placeholder.py – Trivial passing test to keep CI green in Phase 0.

This file will be replaced by real tests as each phase is implemented.
"""


def test_package_importable() -> None:
    """The aqforecast package must be importable without errors."""
    import aqforecast  # noqa: F401

    assert aqforecast.__version__ == "0.1.0"


def test_config_loads() -> None:
    """Config loader must return an object with the expected location fields."""
    from aqforecast.config import cfg

    assert cfg.location.latitude == 12.9716
    assert cfg.location.longitude == 77.5946
    assert cfg.location.timezone == "Asia/Kolkata"


def test_params_loads() -> None:
    """Params loader must return an object with XGBoost and split keys."""
    from aqforecast.config import params

    assert params.xgboost.random_state == 42
    assert 0 < params.split.train_frac < 1

