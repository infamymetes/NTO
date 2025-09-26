# analysis.py
# Central analysis helpers: data loading and normalization

import pandas as pd
import yfinance as yf
from pathlib import Path


def get_data(symbols, freq="1d", start=None, end=None, source="yahoo"):
    """
    Fetch price data for given symbols.

    Parameters
    ----------
    symbols : list[str] or str
        One or more ticker symbols.
    freq : str
        Frequency / interval (yfinance format: "1d", "1h", "30m", etc.)
    start : str or None
        Start date (YYYY-MM-DD). Default None.
    end : str or None
        End date (YYYY-MM-DD). Default None.
    source : str
        Data source, currently only "yahoo" supported.

    Returns
    -------
    pd.DataFrame
        Price DataFrame with datetime index (UTC) and symbols as columns.
    """
    if isinstance(symbols, str):
        symbols = [symbols]

    if source != "yahoo":
        raise NotImplementedError(f"Source '{source}' not yet supported")

    df = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        interval=freq,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
    )

    # Handle multi-index (when multiple tickers)
    if isinstance(df.columns, pd.MultiIndex):
        closes = {sym: df[sym]["Close"] for sym in symbols if sym in df.columns.levels[0]}
        out = pd.DataFrame(closes)
    else:
        out = df[["Close"]].rename(columns={"Close": symbols[0]})

    out.index = pd.to_datetime(out.index).tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
    return out.sort_index()