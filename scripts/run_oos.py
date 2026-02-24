from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant.data.loader import DataSpec, load_prices_yfinance
from quant.backtest.engine_vector import run_backtest
from quant.backtest.types import BacktestConfig
from quant.strategies.buy_hold import BuyAndHold

from quant.validation.split import time_train_test_split
from quant.validation.robustness import build_report


def main() -> int:
    p = argparse.ArgumentParser(description="Out-of-sample validation (train/test) + robustness score.")
    p.add_argument("--symbol", required=True)
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2021-01-01")
    p.add_argument("--interval", default="1d")
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--fees-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)

    # Day 12 params
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--min-bars", type=int, default=50)
    p.add_argument("--rolling-window", type=int, default=63)
    p.add_argument("--rf", type=float, default=0.0)
    p.add_argument("--out", default="artifacts_oos")

    p.add_argument("--vol-target", type=float, default=None)
    p.add_argument("--vol-window", type=int, default=63)
    p.add_argument("--max-leverage", type=float, default=2.0)

    args = p.parse_args()

    spec = DataSpec(symbol=args.symbol, start=args.start, end=args.end, interval=args.interval)
    prices = load_prices_yfinance(spec)

    split = time_train_test_split(prices, train_ratio=args.train_ratio, min_bars=args.min_bars)

    cfg = BacktestConfig(
        symbol=args.symbol,
        initial_cash=args.cash,
        fees_bps=args.fees_bps,
        slippage_bps=args.slippage_bps,
        rolling_window=args.rolling_window,
        risk_free_rate=args.rf,

        # Day 14 risk engine (paramétrable)
        vol_target=args.vol_target,
        vol_window=args.vol_window,
        max_leverage=args.max_leverage,
    )

    res_train = run_backtest(split.train, BuyAndHold(), cfg)
    res_test = run_backtest(split.test, BuyAndHold(), cfg)

    report = build_report(
        meta={
            "symbol": args.symbol,
            "start": args.start,
            "end": args.end,
            "interval": args.interval,
            "train_ratio": args.train_ratio,
            "min_bars": args.min_bars,
            "rolling_window": args.rolling_window,
            "rf": args.rf,
        },
        train_metrics=res_train.metrics,
        test_metrics=res_test.metrics,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.symbol}_oos_report.json").write_text(
        json.dumps(
            {
                "meta": report.meta,
                "train_metrics": report.train_metrics,
                "test_metrics": report.test_metrics,
                "robustness_score": report.robustness_score,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== OOS REPORT ===")
    print(f"Symbol: {args.symbol}")
    print(f"Robustness score: {report.robustness_score:.3f}")
    print("Train Sharpe:", report.train_metrics.get("sharpe"))
    print("Test Sharpe :", report.test_metrics.get("sharpe"))
    print("Train MaxDD :", report.train_metrics.get("max_drawdown"))
    print("Test MaxDD  :", report.test_metrics.get("max_drawdown"))
    print("Train Vol   :", report.train_metrics.get("volatility"))
    print("Test Vol    :", report.test_metrics.get("volatility"))
    print("Test Stability:", report.test_metrics.get("stability_score"))
    print("Exported to :", out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())