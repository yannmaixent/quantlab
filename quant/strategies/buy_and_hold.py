from __future__ import annotations

import pandas as pd
from quant.backtest.types import BacktestConfig
from .base import StrategyOutput

class BuyAndHold:
    name = "buy_and_hold"

    def generate(self, prices: pd.DataFrame, config: BacktestConfig) -> StrategyOutput:
        w = pd.Series(1.0, index=prices.index, dtype=float)
        return StrategyOutput(target_weights=w)

