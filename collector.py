import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger
from apscheduler.schedulers.blocking import BlockingScheduler
import yaml

from ledger import log_pull


DATA_DIR = Path("./data")
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(LOG_DIR / "collector.log")


# ---------------- Fetch Layer ----------------
def fetch_data(symbol: str, freq: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance. Future: plug in Polygon/Alpaca."""
    retries = 3
    delay = 1
    for attempt in range(retries):
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=freq,
            progress=False,
            auto_adjust=False,
        )
        if not df.empty:
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            df["symbol"] = symbol
            return df[["symbol", "Open", "High", "Low", "Close", "Volume"]]
        else:
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                logger.warning(f"[Collector] No data for {symbol} {freq} {start}→{end} after {retries} attempts")
                return df


# ---------------- Storage Layer ----------------
def save_partition(df: pd.DataFrame, symbol: str, freq: str):
    """Save daily-partitioned Parquet files by freq/symbol/date."""
    out_dir = DATA_DIR / freq / symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    # Partition daily
    for date, chunk in df.groupby(df.index.date):
        out_path = out_dir / f"{date}.parquet"
        chunk.to_parquet(out_path, index=True)


# ---------------- Collector ----------------
def run_collector(symbols, freq: str, start: str, end: str, source="yahoo"):
    for symbol in symbols:
        logger.info(f"[Collector] Pulling {symbol} {freq} {start}→{end}")

        df = fetch_data(symbol, freq, start, end)
        if df.empty:
            log_pull(
                symbol=symbol, freq=freq, date_range_start=start, date_range_end=end,
                rows_expected=0, rows_observed=0, action="append",
                source=source, status="failed", notes="Empty fetch"
            )
            continue

        if freq != "1d":
            logger.warning(f"[Collector] Frequency {freq} detected. rows_expected calculation is simplistic; check ledger for intraday completeness.")

        # Save to repo with action detection
        out_dir = DATA_DIR / freq / symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        action = "append"
        for date, chunk in df.groupby(df.index.date):
            out_path = out_dir / f"{date}.parquet"
            if out_path.exists():
                action = "overwrite"
            chunk.to_parquet(out_path, index=True)

        # Ledger entry
        log_pull(
            symbol=symbol, freq=freq,
            date_range_start=start, date_range_end=end,
            rows_expected=None,  # ledger.py now auto-computes rows_expected for daily freq
            rows_observed=len(df),
            action=action,
            source=source,
            status="complete",
        )


def load_universe(name: str):
    with open("config/symbols.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config.get(name, [])


def schedule_jobs():
    scheduler = BlockingScheduler(timezone="US/Eastern")

    def nightly_daily_bars():
        start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        end = datetime.now().strftime("%Y-%m-%d")
        run_collector(load_universe("sp500"), "1d", start, end)

    def intraday_core():
        start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        end = datetime.now().strftime("%Y-%m-%d")
        run_collector(["SPY", "QQQ", "RSP"], "1h", start, end)

    # Nightly daily bars job at 17:15 ET Mon-Fri
    scheduler.add_job(
        nightly_daily_bars,
        'cron',
        day_of_week='mon-fri',
        hour=17,
        minute=15,
        id='nightly_daily_bars'
    )

    # Intraday hourly job on the hour 10-16 ET Mon-Fri
    scheduler.add_job(
        intraday_core,
        'cron',
        day_of_week='mon-fri',
        hour='10-16',
        minute=0,
        id='intraday_hourly'
    )

    scheduler.start()


# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=False, help="Symbols to pull")
    ap.add_argument("--universe", type=str, default=None, help="Predefined universe from config/symbols.yaml (e.g. spy20, nq20)")
    ap.add_argument("--freq", default="1d", help="Frequency: 1d, 1h, 30m, etc.")
    ap.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    ap.add_argument("--schedule", action="store_true", help="Run in scheduled mode")
    ap.add_argument("--source", default="yahoo", help="Data source")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.schedule:
        schedule_jobs()
    else:
        start = args.start or (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        end = args.end or datetime.now().strftime("%Y-%m-%d")
        if args.universe:
            symbols = load_universe(args.universe)
        else:
            symbols = args.symbols or ["SPY", "QQQ", "RSP"]
        run_collector(symbols, args.freq, start, end, source=args.source)


if __name__ == "__main__":
    main()