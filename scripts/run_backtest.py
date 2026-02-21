from __future__ import annotations

import argparse

from quant.backtest.engine_vector import run_backtest
from quant.backtest.types import BacktestConfig
from quant.data.loader import DataSpec, load_prices_yfinance
from quant.reporting.report import BacktestReport
from quant.strategies.buy_hold import BuyAndHold


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2021-01-01")
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--fees_bps", type=float, default=10.0)
    p.add_argument("--slippage_bps", type=float, default=5.0)
    p.add_argument("--out", default="artifacts")
    args = p.parse_args()

    spec = DataSpec(symbol=args.symbol, start=args.start, end=args.end)
    prices = load_prices_yfinance(spec)

    cfg = BacktestConfig(
        symbol=args.symbol,
        initial_cash=args.cash,
        fees_bps=args.fees_bps,
        slippage_bps=args.slippage_bps,
    )

    res = run_backtest(prices, BuyAndHold(), cfg)

    report = BacktestReport(res)
    print(report.summary_str())
    report.export(args.out)
    print(f"Exported to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
