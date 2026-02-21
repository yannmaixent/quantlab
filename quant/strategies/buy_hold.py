from __future__ import annotations

import pandas as pd
from quant.strategies.base import StrategyOutput
from quant.backtest.types import BacktestConfig

class BuyAndHold:
    name = "buy_and_hold"

    def generate(self, prices: pd.DataFrame, config: BacktestConfig) -> StrategyOutput:
        weights = pd.Series(1.0, index=prices.index, dtype=float)
        return StrategyOutput(target_weights=weights)
