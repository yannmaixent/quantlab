from __future__ import annotations

import pandas as pd

from quant.backtest.types import BacktestConfig, BacktestResult
from quant.strategies.base import VectorStrategy, StrategyOutput

def _compute_equity_from_weights(
        close: pd.Series,
        weights: pd.Series,
        initial_cash:float,
) -> pd.Series:
    """
    Corporate V1 assumption:
    - Close-to-close returns.
    - weights are applied with a 1-bar delay to avoid lookahead bias:
        equity[t] = equity[t-1] * (1 + weights[t-1] * ret[t])
    """

    close = close.astype(float)
    rets = close.pct_change().fillna(0.0)

    w = weights.astype(float).clip(lower=0.0, upper=1.0)
    w = w.reindex(close.index).ffill().fillna(0.0)

    w_lag = w.shift(1).fillna(0.0)
    growth = 1.0 + (w_lag * rets)

    equity = (growth.cumprod() *  float(initial_cash)).astype(float)
    equity.name = "equity"
    return equity


def run_backtest(prices: pd.DataFrame, strategy: VectorStrategy, config: BacktestConfig) -> BacktestResult:
     """
     Vector backtest (V1, corporate-ready):
     - Strategy returns targets weights (0..1 for single asset V1)
     - Equity computed from close-to-close returns using lagegd weights.
     """

     idx = prices.index
     close = prices["close"]

     out: StrategyOutput = strategy.generate(prices, config)
     weights = out.target_weights.reindex(idx).astype(float)

     equity_curve = _compute_equity_from_weights(
          close=close,
          weights=weights,
          initial_cash=config.initial_cash,
     )

     # For V1, postions is "weight" (not shares). We'll convert to shares later (execution/portfolio layer)

     positions = weights.copy()
     positions.name = "weight"

     return BacktestResult(
          meta= {"symbol": config.symbol, "strategy": strategy.name, "engine": "vector_v1_weights"},
          equity_curve=equity_curve,
          positions=positions,
          trades=pd.DataFrame(columns=["ts", "symbol", "side", "qty", "price", "fee", "slippage", "notional"]),
          metrics={},
          artifacts={"returns": close.pct_change().fillna(0.0)},
     )
