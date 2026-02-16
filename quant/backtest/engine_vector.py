from __future__ import annotations

from typing import Protocol
import pandas as pd

from .types import BacktestConfig, BacktestResult


class VectorStrategy(Protocol):
    name: str

    def generate_targets(self, prices: pd.DataFrame, config: BacktestConfig) -> pd.Series:
        """
        Return target position (e.g., number of shares) indexed like prices
        """
    


def run_backtest(prices: pd.DataFrame, strategy: VectorStrategy, config: BacktestConfig) -> BacktestResult:
    """
    Vector backtest orchestrator (skeleton)
    Day 1: returns a correctly-shaped placeholder BacktestResult
    """

    idx = prices.index
    empty = pd.Series(index=idx, dtype=float)

    return BacktestResult(
        meta={"symbol": config.symbol, "strategy": strategy.name, "engine": "vector"},
        equity_curve=empty.copy(),
        positions=empty.copy(),
        trades=pd.DataFrame(
            columns=["ts", "symbol", "side", "qty", "price", "fee", "shape", "slippage", "notional"]
        ),
        metrics={},
        artifacts={},
    )

