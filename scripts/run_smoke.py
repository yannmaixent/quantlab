import pandas as pd

from quant.data.loader import DataSpec, load_prices_yfinance
from quant.backtest.types import BacktestConfig
from quant.backtest.engine_vector import run_backtest

class DummyBuyHold:
    name = "dummy_buy_hold"

    def generate_targets(self, prices: pd.DataFrame, config: BacktestConfig) -> pd.Series:
        # placeholder: hold 0 shares for now (Day 3 will implement real target positions)
        return pd.Series(0.0, index=prices.index)
    

if __name__ == "__main__":
    spec = DataSpec(symbol="SPY", start="2020-01-01", end="2021-01-01")
    prices = load_prices_yfinance(spec)

    cfg = BacktestConfig(symbol="SPY", start=spec.start, end=spec.end, benchmark="SPY")
    res = run_backtest(prices=prices, strategy=DummyBuyHold(), config=cfg)

    print(prices.head())
    print(res.meta)
    print("equity_curve len:", len(res.equity_curve))