from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """
    Backtest configuration (UI/API friendly).

    - rolling_window: used for rolling metrics (e.g. rolling Sharpe)
    - risk_free_rate: annualized risk-free rate (e.g. 0.02 for 2%)
    """
    symbol: str
    initial_cash: float = 10_000.0
    fees_bps: float = 0.0
    slippage_bps: float = 0.0

    # Optional descriptive fields (useful for CLI/UI)
    execution: str = "close"
    start: Optional[str] = None
    end: Optional[str] = None
    benchmark: Optional[str] = None

    # --- Day 11: rolling metrics params (UI-ready) ---
    rolling_window: int = 63
    risk_free_rate: float = 0.0
    # Day 14 risk engine
    vol_target: float | None = None      # e.g. 0.15
    vol_window: int = 63
    max_leverage: float = 2.0


@dataclass(frozen=True)
class BacktestResult:
    meta: Dict[str, Any]
    equity_curve: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    artifacts: Dict[str, Any]