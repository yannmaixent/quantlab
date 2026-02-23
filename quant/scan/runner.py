from __future__ import annotations

from typing import List, Optional
import pandas as pd

from quant.data.loader import DataSpec, load_prices_yfinance
from quant.backtest.engine_vector import run_backtest
from quant.backtest.types import BacktestConfig
from quant.strategies.buy_hold import BuyAndHold


# =========================
# Composite scoring logic
# =========================

def _compute_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite quant score.

    Score logic:
    - Reward high CAGR
    - Reward high Sharpe
    - Penalize large drawdowns
    """

    if df.empty:
        return df

    df = df.copy()

    eps = 1e-9

    # Z-score normalization
    df["cagr_z"] = (df["cagr"] - df["cagr"].mean()) / (df["cagr"].std() + eps)
    df["sharpe_z"] = (df["sharpe"] - df["sharpe"].mean()) / (df["sharpe"].std() + eps)
    df["dd_z"] = (df["max_drawdown"] - df["max_drawdown"].mean()) / (df["max_drawdown"].std() + eps)

    # Composite score
    # Drawdown is negative -> subtract its z-score
    df["score"] = df["cagr_z"] + df["sharpe_z"] - df["dd_z"]

    return df


# =========================
# Main scan function
# =========================

def run_scan(
    symbols: List[str],
    start: str = "2020-01-01",
    end: str = "2021-01-01",
    interval: str = "1d",
    initial_cash: float = 10_000.0,
    fees_bps: float = 10.0,
    slippage_bps: float = 5.0,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run Buy & Hold backtest on multiple symbols
    and return ranked DataFrame with metrics + composite score.
    """

    rows = []

    for symbol in symbols:
        try:
            # ---- Load data via DataSpec
            spec = DataSpec(
                symbol=symbol,
                start=start,
                end=end,
                interval=interval,
            )

            prices = load_prices_yfinance(spec)

            if prices is None or prices.empty:
                print(f"[WARNING] No data for {symbol}")
                continue

            # ---- Backtest config
            cfg = BacktestConfig(
                symbol=symbol,
                initial_cash=initial_cash,
                fees_bps=fees_bps,
                slippage_bps=slippage_bps,
            )

            # ---- Run backtest
            res = run_backtest(prices, BuyAndHold(), cfg)

            # ---- Collect metrics
            row = {
                "symbol": symbol,
                **res.metrics,
            }

            rows.append(row)

        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")

    # =========================
    # Build DataFrame
    # =========================

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Compute composite score
    df = _compute_score(df)

    # Rank by score
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # Keep only top N if requested
    if top_n is not None:
        df = df.head(top_n)

    return df
