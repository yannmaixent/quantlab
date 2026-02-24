from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant.data.loader import DataSpec, load_prices_yfinance
from quant.backtest.types import BacktestConfig
from quant.strategies.buy_hold import BuyAndHold
from quant.validation.wf_runner import run_walk_forward


def main() -> int:
    p = argparse.ArgumentParser(description="Walk-forward validation + robustness distribution.")
    p.add_argument("--symbol", required=True)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2021-01-01")
    p.add_argument("--interval", default="1d")
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--fees-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)

    # rolling metrics params
    p.add_argument("--rolling-window", type=int, default=63)
    p.add_argument("--rf", type=float, default=0.0)

    # walk-forward params
    p.add_argument("--train-bars", type=int, default=252)
    p.add_argument("--test-bars", type=int, default=63)
    p.add_argument("--step-bars", type=int, default=63)

    p.add_argument("--out", default="artifacts_wf")
    args = p.parse_args()

    spec = DataSpec(symbol=args.symbol, start=args.start, end=args.end, interval=args.interval)
    prices = load_prices_yfinance(spec)

    cfg = BacktestConfig(
        symbol=args.symbol,
        initial_cash=args.cash,
        fees_bps=args.fees_bps,
        slippage_bps=args.slippage_bps,
        rolling_window=args.rolling_window,
        risk_free_rate=args.rf,
    )

    report = run_walk_forward(
        prices=prices,
        strategy=BuyAndHold(),
        config=cfg,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report.per_window.to_csv(out_dir / f"{args.symbol}_wf_windows.csv", index=False)
    (out_dir / f"{args.symbol}_wf_summary.json").write_text(
        json.dumps({"meta": report.meta, "summary": report.summary}, indent=2),
        encoding="utf-8",
    )

    print("\n=== WALK-FORWARD SUMMARY ===")
    for k, v in report.summary.items():
        print(f"{k}: {v}")
    print(f"Exported to: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())