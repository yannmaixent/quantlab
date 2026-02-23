from __future__ import annotations

import math
from typing import Optional
import numpy as np
import pandas as pd


TRADING_DAYS = 252


def equity_returns(equity: pd.Series) -> pd.Series:
    """
    Close-to-close equity returns (pct change).
    """
    if equity is None or equity.empty:
        return pd.Series(dtype=float)
    return equity.pct_change().fillna(0.0)


def rolling_volatility(equity: pd.Series, window: int) -> pd.Series:
    """
    Annualized rolling volatility from equity returns.
    """
    r = equity_returns(equity)
    vol = r.rolling(window=window).std(ddof=0) * math.sqrt(TRADING_DAYS)
    vol.name = "rolling_volatility"
    return vol


def rolling_sharpe(equity: pd.Series, window: int, risk_free_rate: float = 0.0) -> pd.Series:
    """
    Annualized rolling Sharpe ratio:
    Sharpe = mean(excess_ret) / std(ret) * sqrt(252)

    risk_free_rate is annualized (e.g. 0.02).
    """
    r = equity_returns(equity)

    rf_daily = risk_free_rate / TRADING_DAYS
    excess = r - rf_daily

    mu = excess.rolling(window=window).mean()
    sigma = r.rolling(window=window).std(ddof=0)

    eps = 1e-12
    sharpe = (mu / (sigma + eps)) * math.sqrt(TRADING_DAYS)
    sharpe.name = "rolling_sharpe"
    return sharpe


def rolling_max_drawdown(equity: pd.Series, window: int) -> pd.Series:
    """
    Rolling max drawdown computed per window:
    for each rolling window, compute max drawdown inside that window.

    Returns a series (negative values or 0).
    """
    if equity is None or equity.empty:
        return pd.Series(dtype=float)

    eq = equity.astype(float)

    def _window_maxdd(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return 0.0
        running_max = np.maximum.accumulate(x)
        dd = x / running_max - 1.0
        return float(np.min(dd))

    out = eq.rolling(window=window).apply(_window_maxdd, raw=True)
    out.name = "rolling_max_drawdown"
    return out


def stability_score(rolling_sharpe_series: pd.Series) -> float:
    """
    Produce a simple 0..1 stability score from rolling Sharpe:
    - higher mean Sharpe -> higher score
    - lower std Sharpe -> higher score

    This is a V1 heuristic (product-friendly).
    """
    if rolling_sharpe_series is None or rolling_sharpe_series.dropna().empty:
        return 0.0

    s = rolling_sharpe_series.dropna()
    mean_s = float(s.mean())
    std_s = float(s.std(ddof=0))

    # squash mean Sharpe into 0..1
    mean_part = 1.0 / (1.0 + math.exp(-mean_s))  # logistic
    # penalize instability
    std_part = 1.0 / (1.0 + std_s)

    score = mean_part * std_part
    # clamp
    return float(max(0.0, min(1.0, score)))