import pandas as pd
import pytest
from quant.data.loader import DataSpec, load_prices_yfinance, OHLCV_COLS

@pytest.mark.integration
def test_yfinance_loader_contract():
    spec = DataSpec(symbol= "SPY", start="2020-01-01", end="2020-03-01")
    prices = load_prices_yfinance(spec)

    assert isinstance(prices.index, pd.DatetimeIndex)
    assert prices.index.is_monotonic_increasing
    assert prices.index.is_unique
    assert prices.index.tz is None
    assert list(prices.columns) == OHLCV_COLS
    assert prices[["open", "high", "low", "close"]].isna().sum().sum() == 0
    assert len(prices) >= 20