import argparse
from quant.scan.runner import run_scan


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-symbol quant scan with composite scoring."
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="List of symbols to scan (e.g. SPY QQQ VTI)",
    )

    parser.add_argument(
        "--start",
        default="2020-01-01",
        help="Start date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end",
        default="2021-01-01",
        help="End date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--interval",
        default="1d",
        help="Data interval (e.g. 1d, 1wk)",
    )

    parser.add_argument(
        "--initial-cash",
        type=float,
        default=10_000.0,
        help="Initial capital for backtest",
    )

    parser.add_argument(
        "--fees-bps",
        type=float,
        default=10.0,
        help="Transaction fees in basis points",
    )

    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
        help="Slippage in basis points",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Return only top N ranked symbols",
    )

    parser.add_argument(
        "--out",
        default="scan_results.csv",
        help="Output CSV file",
    )

    args = parser.parse_args()

    df = run_scan(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        interval=args.interval,
        initial_cash=args.initial_cash,
        fees_bps=args.fees_bps,
        slippage_bps=args.slippage_bps,
        top_n=args.top,
    )

    print("\n=== SCAN RESULTS ===")

    if df.empty:
        print("No results.")
        return

    print(df)

    df.to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
