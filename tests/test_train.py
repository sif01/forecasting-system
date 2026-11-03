"""Training split tests ensure chronological partitioning."""

import pandas as pd
import pytest

from aqforecast.train import split_data


def test_split_is_chronological_and_disjoint():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=100, freq="h", tz="UTC"),
            "target_pm2_5_24h": range(100),
        }
    )
    train, validation, test = split_data(frame.sample(frac=1, random_state=5))
    assert train.timestamp.max() < validation.timestamp.min()
    assert validation.timestamp.max() < test.timestamp.min()
    assert len(train) == 70
    assert len(validation) == 15
    assert len(test) == 15


def test_split_rejects_insufficient_rows():
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")}
    )
    with pytest.raises(ValueError, match="Insufficient"):
        split_data(frame)
