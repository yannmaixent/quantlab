import pandas as pd


from quant.backtest.engine_vector import run_backtest
from quant.backtest.types import BacktestConfig
from quant.strategies.buy_hold import BuyAndHold
from quant.data.loader import DataSpec, load_prices_yfinance

class BuyHold:
    name = "dummy_buy_hold"

    """
    def generate_targets(self, prices: pd.DataFrame, config: BacktestConfig) -> pd.Series:
        # placeholder: hold 0 shares for now (Day 3 will implement real target positions)
        return pd.Series(0.0, index=prices.index)
    """

if __name__ == "__main__":
    spec = DataSpec(symbol="SPY", start="2020-01-01", end="2021-01-01")
    prices = load_prices_yfinance(spec)

    cfg = BacktestConfig(
        symbol="SPY", 
        initial_cash= 10000.0, 
        fees_bps=10.0, 
        slippage_bps=5.0,
    )
    
    res = run_backtest(prices, BuyAndHold(), cfg)

    print("Final equity:", res.equity_curve.iloc[-1])
    print("Sharpe:", res.metrics["sharpe"])
    print("Max DD:", res.metrics["max_drawdown"])
    print("All metrics:", res.metrics)