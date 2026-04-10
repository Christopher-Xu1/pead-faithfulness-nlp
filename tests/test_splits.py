import pandas as pd

from src.data.split_dataset import time_based_split


def test_time_based_split_order():
    df = pd.DataFrame(
        {
            "event_date": pd.date_range("2020-01-01", periods=20, freq="D"),
            "text": [f"t{i}" for i in range(20)],
            "label": [i % 2 for i in range(20)],
        }
    )

    train, val, test = time_based_split(df)

    assert len(train) == 14
    assert len(val) == 3
    assert len(test) == 3

    assert train["event_date"].max() <= val["event_date"].min()
    assert val["event_date"].max() <= test["event_date"].min()
