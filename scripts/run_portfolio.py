from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant.portfolio.portfolio_backtest import run_equal_weight_portfolio
from quant.metrics.performance import (
    compute_total_return,
    compute_cagr,
    compute_annualized_volatility,
    compute_sharpe_ratio,
    compute_max_drawdown,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Run equal-weight portfolio backtest on top-N symbols.")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2021-01-01")
    p.add_argument("--interval", default="1d")
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--fees-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--out", default="artifacts_portfolio")
    args = p.parse_args()

    pr = run_equal_weight_portfolio(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        interval=args.interval,
        initial_cash=args.cash,
        fees_bps=args.fees_bps,
        slippage_bps=args.slippage_bps,
    )

    eq = pr.equity_curve

    metrics = {
        "total_return": compute_total_return(eq),
        "cagr": compute_cagr(eq),
        "volatility": compute_annualized_volatility(eq),
        "sharpe": compute_sharpe_ratio(eq),
        "max_drawdown": compute_max_drawdown(eq),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    eq.to_csv(out_dir / "portfolio_equity.csv", header=True)
    pr.per_symbol_metrics.to_csv(out_dir / "per_symbol_metrics.csv", index=False)

    payload = {"meta": pr.meta, "metrics": metrics}
    (out_dir / "portfolio_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== PORTFOLIO SUMMARY ===")
    print(f"Symbols: {args.symbols}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"CAGR: {metrics['cagr']:.2%}")
    print(f"Volatility: {metrics['volatility']:.2%}")
    print(f"Sharpe: {metrics['sharpe']:.3f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Exported to: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())