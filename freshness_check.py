import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import csv
import yaml

from ledger import _ensure_ledger


DATA_DIR = Path("./data")


def expected_days(start: str, end: str, freq: str) -> pd.DatetimeIndex:
    """Generate expected trading days (simplified: weekdays only)."""
    days = pd.date_range(start=start, end=end, freq="B")  # business days
    return days


def check_symbol(symbol: str, freq: str, start: str, end: str):
    """Check coverage for a single symbol/freq against expected days."""
    ledger = _ensure_ledger()
    subset = ledger[(ledger["symbol"] == symbol) & (ledger["freq"] == freq)]
    if subset.empty:
        print(f"[Freshness] No entries found for {symbol} {freq}")
        return {
            "symbol": symbol,
            "freq": freq,
            "start": start,
            "end": end,
            "expected": 0,
            "covered": 0,
            "missing": 0,
            "completeness": 0.0,
        }

    # Actual coverage from ledger
    covered = pd.date_range(
        start=subset["date_range_start"].min(),
        end=subset["date_range_end"].max(),
        freq="D"
    )

    expected = expected_days(start, end, freq)

    missing = expected.difference(covered)

    completeness = 1 - len(missing) / len(expected) if len(expected) > 0 else 1.0

    print(f"\n=== {symbol} {freq} ===")
    print(f"Range requested: {start} → {end}")
    print(f"Coverage: {covered.min().date()} → {covered.max().date()}")
    print(f"Expected days: {len(expected)}, Covered days: {len(covered)}, Missing: {len(missing)}")
    print(f"Completeness: {completeness:.1%}")
    if len(missing) > 0:
        print("Missing days:", [d.strftime("%Y-%m-%d") for d in missing[:10]])
        if len(missing) > 10:
            print("... (truncated)")

    # Note: For intraday completeness (1h, 30m, etc.), check rows_expected vs rows_observed in ledger.

    return {
        "symbol": symbol,
        "freq": freq,
        "start": start,
        "end": end,
        "expected": len(expected),
        "covered": len(covered),
        "missing": len(missing),
        "completeness": completeness,
    }


def load_universe(name: str):
    with open("config/symbols.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config.get(name, [])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True, help="Symbols to check")
    ap.add_argument("--freq", default="1d", help="Frequency: 1d, 1h, 30m, etc.")
    ap.add_argument("--start", default="2025-01-01", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"), help="End date (YYYY-MM-DD)")
    ap.add_argument("--universe", type=str, default=None, help="Predefined universe from config/symbols.yaml (e.g. spy20, nq20)")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.universe:
        symbols = load_universe(args.universe)
    else:
        symbols = args.symbols
    summaries = []
    for symbol in symbols:
        summary = check_symbol(symbol, args.freq, args.start, args.end)
        summaries.append(summary)
    df = pd.DataFrame(summaries)
    df.to_csv("freshness_summary.csv", index=False)
    print("\nSummary written to freshness_summary.csv")


if __name__ == "__main__":
    main()