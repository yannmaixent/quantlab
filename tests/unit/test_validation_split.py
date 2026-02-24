import pandas as pd
from quant.validation.split import time_train_test_split


def test_time_split_basic():
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    prices = pd.DataFrame({"close": range(200)}, index=idx)

    split = time_train_test_split(prices, train_ratio=0.7, min_bars=50)
    assert len(split.train) > 0
    assert len(split.test) > 0
    assert split.train.index.max() < split.test.index.min()