from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import pandas as pd

ExecutionTiming = Literal["close", "next_open"]

@dataclass(frozen=True)
class BacktestConfig:
    symbol: str
    initial_cash: float = 10_000.0
    fees_bps: float = 0.0
    slippage_bps: float = 0.0
    execution: ExecutionTiming = "close"
    start: Optional[str] = None # "YYYY-MM-DD"
    end: Optional[str] = None
    benchmark: Optional[str] = None

@dataclass(frozen=True)
class Trade:
    ts: pd.Timestamp
    symbol: str
    side: Literal["BUY", "SELL"]
    qty: float
    price: float
    fee: float = 0.0
    slippage: float = 0.0
    notional: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    meta: dict[str, Any]
    equity_curve: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float]
    artifacts: dict[str, Any] = field(default_factory=dict)
