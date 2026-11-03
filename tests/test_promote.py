"""Promotion gate boundary tests."""

from aqforecast.promote import promotion_decision


def test_first_model_must_beat_baselines():
    assert promotion_decision(4, None, 5, 0.01)[0]
    assert not promotion_decision(5, None, 5, 0.01)[0]


def test_better_challenger_promotes():
    assert promotion_decision(9, 10, 12, 0.01)[0]


def test_worse_challenger_is_rejected():
    assert not promotion_decision(10.1, 10, 12, 0.01)[0]


def test_improvement_below_margin_is_rejected():
    assert not promotion_decision(9.95, 10, 12, 0.01)[0]
