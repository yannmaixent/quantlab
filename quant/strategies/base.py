from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import pandas as pd

from quant.backtest.types import BacktestConfig

@dataclass(frozen=True)
class StrategyOutput:
    """
    targets_weights: desired portfolio weight in the asset (0..1 fro single-asset V1)
    index must match prices.index.
    """

    target_weights : pd.Series

class VectorStrategy(Protocol):
    name: str
    def generate(self, prices: pd.DataFrame, config: BacktestConfig) -> StrategyOutput:
        """
        Empty for now
        """