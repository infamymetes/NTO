import uuid
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

LEDGER_PATH = Path("./data/ledger.parquet")


def _ensure_ledger() -> pd.DataFrame:
    """Load ledger if it exists, otherwise create an empty DataFrame with schema."""
    expected_columns = [
        "id", "timestamp_pull", "symbol", "freq",
        "date_range_start", "date_range_end",
        "rows_expected", "rows_observed",
        "action", "source", "schema_version",
        "close_type", "status", "notes"
    ]
    if LEDGER_PATH.exists():
        df = pd.read_parquet(LEDGER_PATH)
        if not all(col in df.columns for col in expected_columns):
            print("[Warning] Ledger schema version mismatch may exist.")
        return df

    # Define schema
    df = pd.DataFrame(columns=expected_columns)
    df.to_parquet(LEDGER_PATH, index=False)
    return df


def log_pull(
    symbol: str,
    freq: str,
    date_range_start: str,
    date_range_end: str,
    rows_expected: int,
    rows_observed: int,
    action: str,
    source: str,
    schema_version: str = "v1.0",
    close_type: str = "adjusted",
    status: str = "complete",
    notes: str = ""
):
    """Append a new pull entry to the ledger."""
    df = _ensure_ledger()

    start_date = pd.to_datetime(date_range_start).date()
    end_date = pd.to_datetime(date_range_end).date()

    if rows_expected is None:
        if freq in ("1d", "daily"):
            # Compute business days inclusive
            bdays = pd.date_range(start=start_date, end=end_date, freq="B")
            rows_expected = len(bdays)
        else:
            rows_expected = None

    entry = {
        "id": str(uuid.uuid4()),
        "timestamp_pull": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "freq": freq,
        "date_range_start": start_date,
        "date_range_end": end_date,
        "rows_expected": rows_expected,
        "rows_observed": rows_observed,
        "action": action,
        "source": source,
        "schema_version": schema_version,
        "close_type": close_type,
        "status": status,
        "notes": notes,
    }

    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_parquet(LEDGER_PATH, index=False)
    print(f"[Ledger] Logged pull: {symbol} {freq} {date_range_start}→{date_range_end} ({rows_observed}/{rows_expected})")


def check_status(symbol: str = None, freq: str = None) -> pd.DataFrame:
    """Return a filtered ledger for quick inspection."""
    df = _ensure_ledger()
    if symbol:
        df = df[df["symbol"] == symbol]
    if freq:
        df = df[df["freq"] == freq]
    df_sorted = df.sort_values("timestamp_pull", ascending=False)
    # Print human-readable snapshot
    snapshot_cols = ["timestamp_pull", "symbol", "freq", "source", "action", "status"]
    print(df_sorted[snapshot_cols].to_string(index=False))
    return df_sorted


def list_gaps(symbol: str, freq: str, expected_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return missing dates for a symbol/freq compared to expected_days."""
    df = _ensure_ledger()
    df = df[(df["symbol"] == symbol) & (df["freq"] == freq)]

    covered = pd.date_range(
        start=df["date_range_start"].min(),
        end=df["date_range_end"].max(),
        freq="D"
    )
    missing = expected_days.difference(covered)
    return missing


def summarize(symbol: str, freq: str):
    """Print summary of ledger entries for a symbol and freq."""
    df = _ensure_ledger()
    df = df[(df["symbol"] == symbol) & (df["freq"] == freq)]
    if df.empty:
        print(f"No ledger entries found for {symbol} at freq {freq}.")
        return
    first_date = df["date_range_start"].min()
    last_date = df["date_range_end"].max()
    total_rows_observed = df["rows_observed"].sum()
    total_rows_expected = df["rows_expected"].dropna().sum()
    print(f"Summary for {symbol} at freq {freq}:")
    print(f"  Date range: {first_date} to {last_date}")
    print(f"  Total rows observed: {total_rows_observed}")
    if total_rows_expected > 0:
        completeness = (total_rows_observed / total_rows_expected) * 100
        print(f"  Completeness: {completeness:.2f}%")
    else:
        print("  Completeness: N/A (rows_expected not available)")