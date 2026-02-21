from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

def compute_total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)

def compute_cagr(equity: pd.Series) -> float:
    n_years = len(equity) / TRADING_DAYS
    if n_years == 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1/n_years) - 1.0)

def compute_annualized_volatility(equity: pd.Series) -> float:
    returns = equity.pct_change().dropna()
    return float(returns.std() * np.sqrt(TRADING_DAYS))

def compute_sharpe_ratio(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    returns = equity.pct_change().dropna()
    if returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / TRADING_DAYS
    return float(
        excess_returns.mean() / excess_returns.std() * np.sqrt(TRADING_DAYS)
    )

def compute_drawdown(equity: pd.Series) -> pd.Series:
    cumulative_max = equity.cummax()
    drawdonw = equity / cumulative_max - 1.0
    return drawdonw

def compute_max_drawdown(equity: pd.Series) -> float:
    return float(compute_drawdown(equity).min())