from __future__ import annotations

from typing import List
import pandas as pd

from quant.data.loader import DataSpec, load_prices_yfinance
from quant.backtest.engine_vector import run_backtest
from quant.backtest.types import BacktestConfig
from quant.strategies.buy_hold import BuyAndHold


def run_scan(
    symbols: List[str],
    start: str = "2020-01-01",
    end: str = "2021-01-01",
    interval: str = "1d",
    initial_cash: float = 10_000.0,
    fees_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> pd.DataFrame:
    rows = []

    for symbol in symbols:
        try:
            spec = DataSpec(symbol=symbol, start=start, end=end, interval=interval)
            prices = load_prices_yfinance(spec)

            cfg = BacktestConfig(
                symbol=symbol,
                initial_cash=initial_cash,
                fees_bps=fees_bps,
                slippage_bps=slippage_bps,
            )

            res = run_backtest(prices, BuyAndHold(), cfg)

            rows.append({"symbol": symbol, **res.metrics})

        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)

    return df
