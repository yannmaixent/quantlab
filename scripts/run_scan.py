import argparse
from quant.scan.runner import run_scan


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2021-01-01")
    p.add_argument("--interval", default="1d")
    p.add_argument("--out", default="scan_results.csv")
    args = p.parse_args()

    df = run_scan(args.symbols, start=args.start, end=args.end, interval=args.interval)

    print("\n=== SCAN RESULTS ===")
    print(df)

    df.to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}")
