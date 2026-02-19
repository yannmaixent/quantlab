import pandas as pd

from quant.backtest.engine_vector import run_backtest
from quant.backtest.types import BacktestConfig
from quant.strategies.base import StrategyOutput

class Dummy:
    name = "dummy"
    def generate(self, prices: pd.DataFrame, config: BacktestConfig) -> StrategyOutput:
        w = pd.Series(0.0, index=prices.index)
        return StrategyOutput(target_weights=w)
    
def test_run_backtest_returns_consistent_shapes():
    idx = pd.date_range("2020-01-01", periods= 5, freq="D")
    prices = pd.DataFrame(
        {
            "open": [1,2,3,4,5],
            "high": [1,2,3,4,5],
            "low": [1,2,3,4,5],
            "close": [1,2,3,4,5],
            "volume": [10,10,10,10,10],
        },
        index=idx
    )

    cfg = BacktestConfig(symbol="SPY")
    res = run_backtest(prices, Dummy(), cfg)

    assert len(res.equity_curve) == len(prices)
    assert len(res.positions) == len(prices)
    assert res.meta["engine"].startswith("vector")