from __future__ import annotations

import math
import pandas as pd
import numpy as np

from quant.metrics.performance import (
    compute_total_return,
    compute_cagr,
    compute_annualized_volatility,
    compute_sharpe_ratio,
    compute_max_drawdown,
)


TRADING_DAYS = 252


def build_equal_weight_portfolio(
    prices: pd.DataFrame,
    rebalance_every: int = 21,
) -> pd.DataFrame:
    """
    Create equal-weight allocation with periodic rebalancing.
    """

    idx = prices.index
    symbols = prices.columns
    n = len(symbols)

    weights = pd.DataFrame(0.0, index=idx, columns=symbols)

    for i in range(0, len(idx), rebalance_every):
        weights.iloc[i] = 1.0 / n

    weights = weights.ffill().fillna(0.0)

    return weights


def run_portfolio_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    initial_cash: float = 10_000.0,
) -> dict:
    """
    Simple portfolio backtest (vectorized, no fees V1).
    """

    returns = prices.pct_change().fillna(0.0)

    portfolio_returns = (weights.shift(1) * returns).sum(axis=1)

    equity = (1 + portfolio_returns).cumprod() * initial_cash

    metrics = {
        "total_return": compute_total_return(equity),
        "cagr": compute_cagr(equity),
        "volatility": compute_annualized_volatility(equity),
        "sharpe": compute_sharpe_ratio(equity),
        "max_drawdown": compute_max_drawdown(equity),
    }

    return {
        "equity_curve": equity,
        "weights": weights,
        "metrics": metrics,
    }