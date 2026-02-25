from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd

from quant.data.loader import DataSpec, load_prices_yfinance
from quant.portfolio.engine_portfolio import (
    build_equal_weight_portfolio,
    run_portfolio_backtest,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Multi-asset portfolio backtest.")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2021-01-01")
    p.add_argument("--interval", default="1d")
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--rebalance", type=int, default=21)
    p.add_argument("--out", default="artifacts_portfolio")

    args = p.parse_args()

    price_dict = {}

    for s in args.symbols:
        spec = DataSpec(symbol=s, start=args.start, end=args.end, interval=args.interval)
        df = load_prices_yfinance(spec)
        price_dict[s] = df["close"]

    prices = pd.DataFrame(price_dict).dropna()

    weights = build_equal_weight_portfolio(
        prices,
        rebalance_every=args.rebalance,
    )

    result = run_portfolio_backtest(
        prices=prices,
        weights=weights,
        initial_cash=args.cash,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result["equity_curve"].to_csv(out_dir / "equity_curve.csv")
    result["weights"].to_csv(out_dir / "weights.csv")

    (out_dir / "metrics.json").write_text(
        json.dumps(result["metrics"], indent=2),
        encoding="utf-8",
    )

    print("\n=== PORTFOLIO SUMMARY ===")
    print("Symbols:", args.symbols)
    print("Total Return:", result["metrics"]["total_return"])
    print("Sharpe:", result["metrics"]["sharpe"])
    print("Max Drawdown:", result["metrics"]["max_drawdown"])
    print("Exported to:", out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())