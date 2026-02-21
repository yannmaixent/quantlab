from __future__ import annotations

import pandas as pd

from quant.backtest.types import BacktestConfig, BacktestResult
from quant.strategies.base import VectorStrategy, StrategyOutput

def _simulate_execution(
          prices: pd.DataFrame,
          weights: pd.Series,
          initial_cash: float,
          fees_bps: float,
          slippage_bps: float,
):
     close = prices["close"].astype(float)
     idx = prices.index

     weights = weights.reindex(idx).ffill().fillna(0.0).clip(0.0, 1.0)

     equity = []
     shares = []
     cash =initial_cash
     current_shares = 0.0

     for t in range(len(idx)):
          price = close.iloc[t]

          target_weight = weights.iloc[t]
          portfolio_value = cash + current_shares * price
          target_value = portfolio_value * target_weight
          target_shares = target_value / price

          delta_shares = target_shares - current_shares
          trade_notional = delta_shares * price

          fee = abs(trade_notional) * fees_bps / 10000.0
          slippage = abs(trade_notional) * slippage_bps / 10000.0

          cash -= trade_notional
          cash -= fee
          cash -= slippage

          current_shares = target_shares

          equity.append(cash + current_shares * price)
          shares.append(current_shares)

     equity_series = pd.Series(equity, index=idx, name="equity")
     shares_series = pd.Series(shares, index=idx, name="shares")

     return equity_series, shares_series



def run_backtest(
     prices: pd.DataFrame, 
     strategy: VectorStrategy, 
     config: BacktestConfig,
) -> BacktestResult:

     """
     Vector backtest V2 (execution-aware):

     - strategy returns target weights (0..1 single asset V1).
     - Engine converts weights -> shares.
     - Cash is dynamically tracked.
     - Fees and slipppage applied in basis points.
     """
     
     idx = prices.index
     close = prices["close"].astype(float)

     # --- Strategy output ---
     out: StrategyOutput = strategy.generate(prices, config)
     weights = (
          out.target_weights
          .reindex(idx)
          .ffill()
          .fillna(0.0)
          .clip(0.0, 1.0)
          .astype(float)     
     )

     # --- Execution simulation ---
     equity_curve, shares = _simulate_execution(
          prices=prices,
          weights=weights,
          initial_cash = config.initial_cash,
          fees_bps = config.fees_bps,
          slippage_bps=config.slippage_bps,
     )

     positions = shares

     # --- Build result ---
     return BacktestResult(
          meta={
               "symbol": config.symbol,
               "strategy": strategy.name,
               "engine": "vector_v2_execution",
          },
          equity_curve = equity_curve,
          positions=positions,
          trades=pd.DataFrame(
               columns=["ts", "symbol","side", "qty", "price", "fee", "slippage", "notional"]
          ),
          metrics = {},
          artifacts={
               "weights": weights,
               "returns":close.pct_change().fillna(0.0)
          },
     )