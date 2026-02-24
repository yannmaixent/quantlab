from __future__ import annotations

import math
import pandas as pd
import numpy as np

TRADING_DAYS = 252


def realized_volatility(
    equity: pd.Series,
    window: int,
) -> pd.Series:
    """
    Annualized rolling realized volatility from equity curve.
    """
    returns = equity.pct_change().fillna(0.0)
    vol = returns.rolling(window=window).std(ddof=0) * math.sqrt(TRADING_DAYS)
    return vol


def apply_vol_targeting(
    base_weights: pd.Series,
    equity_curve: pd.Series,
    target_vol: float = 0.15,
    window: int = 63,
    max_leverage: float = 2.0,
) -> pd.Series:
    """
    Adjust weights to target annualized volatility.

    target_vol: desired annual vol (e.g. 0.15 = 15%)
    max_leverage: cap on scaling
    """

    vol = realized_volatility(equity_curve, window=window)

    adj_weights = base_weights.copy().astype(float)

    for i in range(len(adj_weights)):
        current_vol = vol.iloc[i]

        if np.isnan(current_vol) or current_vol <= 1e-8:
            continue

        scale = target_vol / current_vol
        scale = min(scale, max_leverage)

        adj_weights.iloc[i] = adj_weights.iloc[i] * scale

    return adj_weights.clip(0.0, max_leverage)