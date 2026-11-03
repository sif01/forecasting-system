"""Promotion gate boundary tests."""

from aqforecast.promote import champion_status_message, promotion_decision


def test_first_model_must_beat_baselines():
    assert promotion_decision(4, None, 5, 0.01)[0]
    assert not promotion_decision(5, None, 5, 0.01)[0]


def test_better_challenger_promotes():
    assert promotion_decision(9, 10, 12, 0.01)[0]


def test_worse_challenger_is_rejected():
    assert not promotion_decision(10.1, 10, 12, 0.01)[0]


def test_improvement_below_margin_is_rejected():
    assert not promotion_decision(9.95, 10, 12, 0.01)[0]


def test_first_model_rejection_clearly_reports_no_champion():
    message = champion_status_message(
        None, "4", False, "first model did not beat both baselines"
    )

    assert message.startswith("No champion yet:")
    assert "retrain completed" in message
    assert "needs a champion" in message


def test_rejected_challenger_reports_existing_champion():
    message = champion_status_message(
        "3", "4", False, "challenger failed configured MAE improvement"
    )

    assert message == "Champion model version 3 remains active."


def test_promoted_challenger_reports_new_champion():
    message = champion_status_message(
        None, "4", True, "first model beats both baselines"
    )

    assert message == "Champion model version 4 is available."
