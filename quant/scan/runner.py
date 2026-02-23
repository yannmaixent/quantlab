from __future__ import annotations

from typing import List, Optional
import pandas as pd

from quant.data.loader import DataSpec, load_prices_yfinance
from quant.backtest.engine_vector import run_backtest
from quant.backtest.types import BacktestConfig
from quant.strategies.buy_hold import BuyAndHold


def _compute_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite quant score:
    - Reward high CAGR
    - Reward high Sharpe
    - Penalize large drawdowns
    """
    if df.empty:
        return df

    df = df.copy()
    eps = 1e-9

    df["cagr_z"] = (df["cagr"] - df["cagr"].mean()) / (df["cagr"].std() + eps)
    df["sharpe_z"] = (df["sharpe"] - df["sharpe"].mean()) / (df["sharpe"].std() + eps)
    df["dd_z"] = (df["max_drawdown"] - df["max_drawdown"].mean()) / (df["max_drawdown"].std() + eps)

    # max_drawdown is negative -> subtract its zscore
    df["score"] = df["cagr_z"] + df["sharpe_z"] - df["dd_z"]
    return df


def run_scan(
    symbols: List[str],
    start: str = "2020-01-01",
    end: str = "2021-01-01",
    interval: str = "1d",
    initial_cash: float = 10_000.0,
    fees_bps: float = 10.0,
    slippage_bps: float = 5.0,
    top_n: Optional[int] = None,
    lookback_months: Optional[int] = None,
    momentum_filter: bool = True,
) -> pd.DataFrame:
    """
    Run Buy & Hold backtest on multiple symbols and return ranked DataFrame.

    Enhancements (Day 9):
    - lookback_months: use only last N months of data
    - momentum_filter: exclude symbols with non-positive total return over the backtest window
    """

    rows = []

    for symbol in symbols:
        try:
            spec = DataSpec(symbol=symbol, start=start, end=end, interval=interval)
            prices = load_prices_yfinance(spec)

            if prices is None or prices.empty:
                print(f"[WARNING] No data for {symbol}")
                continue

            # Rolling window slicing (last N months)
            if lookback_months is not None:
                cutoff = prices.index.max() - pd.DateOffset(months=lookback_months)
                prices = prices.loc[prices.index >= cutoff]

                if prices.empty or len(prices) < 5:
                    print(f"[WARNING] Not enough data after lookback for {symbol}")
                    continue

            cfg = BacktestConfig(
                symbol=symbol,
                initial_cash=initial_cash,
                fees_bps=fees_bps,
                slippage_bps=slippage_bps,
            )

            res = run_backtest(prices, BuyAndHold(), cfg)

            # Momentum filter: keep only positive total return over window
            if momentum_filter and res.metrics.get("total_return", 0.0) <= 0.0:
                continue

            rows.append({"symbol": symbol, **res.metrics})

        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = _compute_score(df)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    if top_n is not None:
        df = df.head(top_n)

    return df