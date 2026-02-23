from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import pandas as pd

from quant.data.loader import DataSpec, load_prices_yfinance
from quant.backtest.engine_vector import run_backtest
from quant.backtest.types import BacktestConfig
from quant.strategies.buy_hold import BuyAndHold
from quant.reporting.report import BacktestReport


@dataclass(frozen=True)
class PortfolioResult:
    """
    Portfolio equity built by combining individual strategy equity curves.
    """
    meta: dict
    equity_curve: pd.Series
    per_symbol_metrics: pd.DataFrame


def _align_equity_curves(curves: Dict[str, pd.Series]) -> pd.DataFrame:
    df = pd.concat(curves, axis=1)  # columns are symbols
    df = df.dropna(how="any")       # keep common dates
    return df


def run_equal_weight_portfolio(
    symbols: list[str],
    start: str,
    end: str,
    interval: str,
    initial_cash: float,
    fees_bps: float,
    slippage_bps: float,
) -> PortfolioResult:
    """
    V1: Run Buy&Hold per symbol with initial_cash * weight,
    then sum equity curves to get portfolio equity.
    """
    if not symbols:
        raise ValueError("symbols list is empty")

    weight = 1.0 / len(symbols)
    curves: Dict[str, pd.Series] = {}
    metrics_rows = []

    for sym in symbols:
        spec = DataSpec(symbol=sym, start=start, end=end, interval=interval)
        prices = load_prices_yfinance(spec)

        cfg = BacktestConfig(
            symbol=sym,
            initial_cash=initial_cash * weight,
            fees_bps=fees_bps,
            slippage_bps=slippage_bps,
        )

        res = run_backtest(prices, BuyAndHold(), cfg)
        curves[sym] = res.equity_curve
        metrics_rows.append({"symbol": sym, **res.metrics})

    eq_df = _align_equity_curves(curves)
    portfolio_equity = eq_df.sum(axis=1)
    portfolio_equity.name = "equity"

    per_symbol_metrics = pd.DataFrame(metrics_rows).sort_values("symbol").reset_index(drop=True)

    return PortfolioResult(
        meta={
            "engine": "portfolio_v1_equal_weight_sum_equity",
            "symbols": symbols,
            "start": start,
            "end": end,
            "interval": interval,
        },
        equity_curve=portfolio_equity,
        per_symbol_metrics=per_symbol_metrics,
    )