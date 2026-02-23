import argparse
from quant.scan.runner import run_scan


def main():
    p = argparse.ArgumentParser(
        description="Run multi-symbol quant scan with composite scoring + rolling lookback."
    )

    p.add_argument("--symbols", nargs="+", required=True, help="List of symbols (e.g. SPY QQQ VTI)")
    p.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default="2021-01-01", help="End date (YYYY-MM-DD)")
    p.add_argument("--interval", default="1d", help="Interval (e.g. 1d, 1wk)")

    p.add_argument("--initial-cash", type=float, default=10_000.0, help="Initial cash")
    p.add_argument("--fees-bps", type=float, default=10.0, help="Fees in bps")
    p.add_argument("--slippage-bps", type=float, default=5.0, help="Slippage in bps")

    p.add_argument("--top", type=int, default=None, help="Return only top N symbols")
    p.add_argument("--out", default="scan_results.csv", help="Output CSV file")

    # Day 9 additions
    p.add_argument("--lookback", type=int, default=None, help="Rolling lookback window in months")
    p.add_argument(
        "--no-momentum-filter",
        action="store_true",
        help="Disable momentum filter (keep negative total return symbols)",
    )

    args = p.parse_args()

    df = run_scan(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        interval=args.interval,
        initial_cash=args.initial_cash,
        fees_bps=args.fees_bps,
        slippage_bps=args.slippage_bps,
        top_n=args.top,
        lookback_months=args.lookback,
        momentum_filter=not args.no_momentum_filter,
    )

    print("\n=== SCAN RESULTS ===")
    if df.empty:
        print("No results.")
        return 0

    print(df)
    df.to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main()) ##YBO