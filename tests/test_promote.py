"""Promotion gate boundary tests."""

from unittest.mock import Mock

import mlflow
import pytest

from aqforecast.promote import (
    _get_challenger_version,
    _get_champion_version,
    champion_status_message,
    promotion_decision,
)


def test_first_model_must_beat_baselines():
    assert promotion_decision(4, None, 5, 0.01)[0]
    assert not promotion_decision(5, None, 5, 0.01)[0]


def test_better_challenger_promotes():
    assert promotion_decision(9, 10, 12, 0.01)[0]


def test_worse_challenger_is_rejected():
    assert not promotion_decision(10.1, 10, 12, 0.01)[0]


def test_improvement_below_margin_is_rejected():
    assert not promotion_decision(9.95, 10, 12, 0.01)[0]


def test_missing_champion_alias_means_no_champion_yet():
    client = Mock()
    client.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException(
        "Champion alias does not exist", error_code="RESOURCE_DOES_NOT_EXIST"
    )

    assert _get_champion_version(client, "aqforecast") is None


def test_other_registry_errors_are_not_treated_as_missing_champion():
    client = Mock()
    error = mlflow.exceptions.MlflowException(
        "Registry unavailable", error_code="INTERNAL_ERROR"
    )
    client.get_model_version_by_alias.side_effect = error

    with pytest.raises(mlflow.exceptions.MlflowException) as raised:
        _get_champion_version(client, "aqforecast")

    assert raised.value is error


def test_challenger_alias_must_belong_to_current_run():
    client = Mock()
    client.get_model_version_by_alias.return_value = Mock(run_id="run-1", version="2")

    assert _get_challenger_version(client, "aqforecast", "run-1") == "2"

    with pytest.raises(ValueError, match="run-2"):
        _get_challenger_version(client, "aqforecast", "run-2")


def test_challenger_registry_errors_propagate():
    client = Mock()
    error = mlflow.exceptions.MlflowException(
        "Registry unavailable", error_code="INTERNAL_ERROR"
    )
    client.get_model_version_by_alias.side_effect = error

    with pytest.raises(mlflow.exceptions.MlflowException) as raised:
        _get_challenger_version(client, "aqforecast", "run-1")

    assert raised.value is error


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
