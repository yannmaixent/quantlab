import pandas as pd

from quant.backtest.engine_vector import run_backtest
from quant.backtest.types import BacktestConfig
from quant.strategies.buy_and_hold import BuyAndHold

def test_buy_and_hold_equity_increases_when_price_increases():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.DataFrame(
        {
            "open": [1,2,3,4,5],
            "high": [1,2,3,4,5],
            "low": [1,2,3,4,5],
            "close": [1,2,3,4,5],
            "volume": [10,10,10,10,10],
        },
        index=idx,
    )

    cfg = BacktestConfig(symbol= "SPY", initial_cash=100.0)
    res = run_backtest(prices, BuyAndHold(), cfg)

    assert res.equity_curve.iloc[-1] > res.equity_curve.iloc[0]
    assert (res.positions == 1.0).all()