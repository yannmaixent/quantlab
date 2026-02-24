import pandas as pd
from quant.validation.walkforward import walk_forward_splits


def test_walk_forward_splits_produce_windows():
    idx = pd.date_range("2020-01-01", periods=600, freq="D")
    prices = pd.DataFrame({"close": range(600)}, index=idx)

    splits = walk_forward_splits(prices, train_bars=200, test_bars=50, step_bars=50)
    assert len(splits) > 0
    # Ensure ordering
    assert splits[0].train.index.max() < splits[0].test.index.min()