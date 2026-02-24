import pandas as pd
import numpy as np
from quant.risk.vol_target import realized_volatility


def test_realized_volatility_basic():
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    equity = pd.Series(100 * (1.001 ** np.arange(200)), index=idx)
    vol = realized_volatility(equity, window=30)
    assert len(vol) == len(equity)