from __future__ import annotations

import pandas as pd

from quant.backtest.types import BacktestConfig, BacktestResult
from quant.strategies.base import VectorStrategy, StrategyOutput
from quant.risk.vol_target import apply_vol_targeting

from quant.metrics.performance import (
    compute_total_return,
    compute_cagr,
    compute_annualized_volatility,
    compute_sharpe_ratio,
    compute_max_drawdown,
)

from quant.metrics.rolling import (
    rolling_sharpe,
    rolling_volatility,
    rolling_max_drawdown,
    stability_score,
)


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
    cash = initial_cash
    current_shares = 0.0

    for t in range(len(idx)):
        price = close.iloc[t]

        target_weight = float(weights.iloc[t])
        portfolio_value = cash + current_shares * price
        target_value = portfolio_value * target_weight
        target_shares = target_value / price if price != 0 else 0.0

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
    Vector backtest V2 (execution-aware) + Day 11 rolling robustness:

    - Strategy returns target weights (0..1 single-asset V1).
    - Engine converts weights -> shares with dynamic cash.
    - Fees and slippage applied in basis points.
    - Computes base performance metrics (CAGR, Sharpe, MaxDD, Vol).
    - Computes rolling metrics (rolling Sharpe/Vol/MaxDD) using config.rolling_window.
    - Adds a stability_score based on rolling Sharpe.
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
        initial_cash=config.initial_cash,
        fees_bps=config.fees_bps,
        slippage_bps=config.slippage_bps,
    )

    # --- Risk Engine (optional) ---
    if config.vol_target is not None:
        adj_weights = apply_vol_targeting(
            base_weights=weights,
            equity_curve=equity_curve,
            target_vol=config.vol_target,
            window=config.vol_window,
            max_leverage=config.max_leverage,
        )

        equity_curve, shares = _simulate_execution(
            prices=prices,
            weights=adj_weights,
            initial_cash=config.initial_cash,
            fees_bps=config.fees_bps,
            slippage_bps=config.slippage_bps,
        )
        weights = adj_weights

    # --- Base metrics ---
    metrics: dict[str, float] = {
        "total_return": compute_total_return(equity_curve),
        "cagr": compute_cagr(equity_curve),
        "volatility": compute_annualized_volatility(equity_curve),
        "sharpe": compute_sharpe_ratio(equity_curve),
        "max_drawdown": compute_max_drawdown(equity_curve),
    }

    # --- Day 11: Rolling metrics (UI-ready: param via config) ---
    w = int(getattr(config, "rolling_window", 63))
    rf = float(getattr(config, "risk_free_rate", 0.0))

    metrics["rolling_window"] = float(w)
    metrics["risk_free_rate"] = float(rf)

    # Only compute if we have enough data
    if w > 1 and len(equity_curve) >= (w + 2):
        rs = rolling_sharpe(equity_curve, window=w, risk_free_rate=rf)
        rv = rolling_volatility(equity_curve, window=w)
        rdd = rolling_max_drawdown(equity_curve, window=w)

        rs_clean = rs.dropna()
        rv_clean = rv.dropna()
        rdd_clean = rdd.dropna()

        if not rs_clean.empty:
            metrics["rolling_sharpe_mean"] = float(rs_clean.mean())
            metrics["rolling_sharpe_std"] = float(rs_clean.std(ddof=0))
            metrics["rolling_vol_mean"] = float(rv_clean.mean()) if not rv_clean.empty else float("nan")
            metrics["rolling_maxdd_worst"] = float(rdd_clean.min()) if not rdd_clean.empty else float("nan")
            metrics["stability_score"] = float(stability_score(rs_clean))
        else:
            metrics["rolling_sharpe_mean"] = float("nan")
            metrics["rolling_sharpe_std"] = float("nan")
            metrics["rolling_vol_mean"] = float("nan")
            metrics["rolling_maxdd_worst"] = float("nan")
            metrics["stability_score"] = 0.0
    else:
        # Not enough history for rolling window
        metrics["rolling_sharpe_mean"] = float("nan")
        metrics["rolling_sharpe_std"] = float("nan")
        metrics["rolling_vol_mean"] = float("nan")
        metrics["rolling_maxdd_worst"] = float("nan")
        metrics["stability_score"] = 0.0

    positions = shares

    # --- Build result ---
    return BacktestResult(
        meta={
            "symbol": config.symbol,
            "strategy": strategy.name,
            "engine": "vector_v2_execution",
        },
        equity_curve=equity_curve,
        positions=positions,
        trades=pd.DataFrame(
            columns=["ts", "symbol", "side", "qty", "price", "fee", "slippage", "notional"]
        ),
        metrics=metrics,
        artifacts={
            "weights": weights,
            "returns": close.pct_change().fillna(0.0),
            # Optional (useful later for UI)
            "rolling": {
                "rolling_sharpe": rolling_sharpe(equity_curve, window=w, risk_free_rate=rf) if (w > 1 and len(equity_curve) >= (w + 2)) else pd.Series(dtype=float),
                "rolling_volatility": rolling_volatility(equity_curve, window=w) if (w > 1 and len(equity_curve) >= (w + 2)) else pd.Series(dtype=float),
                "rolling_max_drawdown": rolling_max_drawdown(equity_curve, window=w) if (w > 1 and len(equity_curve) >= (w + 2)) else pd.Series(dtype=float),
            },
        },
    )