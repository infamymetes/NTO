#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StatArb Engine v1.1 (single-file scaffold)
Author: Mike + ChatGPT
License: MIT
---------------------------------------------------------------------
Purpose
  • Scan a universe of tickers for cointegrated pairs across multiple rolling windows
  • Compute hedge ratio (beta), spread Z-score, and mean-reversion half-life
  • Score conviction and emit trade signals
  • Export Watchlist CSV (top candidates) and Diagnostics CSV (full details)

Changes in v1.1
  • Fixed price loading to handle adjusted close correctly (auto_adjust=False)
  • Added support for loading local CSV files with adjusted close prices

Usage Example
  python StatArbEngine_v1.py \
    --universe SPY QQQ RSP DIA XLK XLF XLE XLV XLY XLU IWM \
    --start 2021-01-01 \
    --interval 1d \
    --windows 30 60 90 180 252 \
    --pvalue 0.05 \
    --z_enter 2.0 --z_scale 2.5 --z_exit 0.5 \
    --hl_min 3 --hl_max 40 \
    --max_pairs 8 \
    --watchlist ./outputs/watchlist_pairs.csv \
    --diagnostics ./outputs/diagnostics.csv

Notes
  • Requires: pandas, numpy, statsmodels, yfinance
  • Internet access is required at runtime to fetch prices via yfinance unless using --local_dir
  • For intraday, set --interval 5m and adjust windows to bar counts
---------------------------------------------------------------------
"""
import numpy as np
import pandas as pd
import json as _json
import argparse
import itertools
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime, timedelta

# --------------------------- Interval Fallback Map ---------------------------
# Fallbacks for yfinance intraday intervals if no data is returned.
INTERVAL_FALLBACK = {
    "1m": "5m",
    "2m": "5m",
    "5m": "15m",
    "15m": "30m",
    "30m": "60m",
    "60m": "1d"
}

# Diagnostics logs for data fetch behavior
FALLBACK_LOG = []   # list of dicts: {"from": interval, "to": fallback, "start": start, "tickers": len(tickers)}
CLAMP_LOG = []      # list of dicts: {"interval": interval, "from": str(start), "to": str(min_start)}
def clamp_start_for_interval(interval: str, start_date) -> pd.Timestamp:
    """
    Clamp start_date based on Yahoo Finance interval constraints.
    1m/2m: last 7 days only.
    5m/15m/30m/60m/1h: last 60 days.
    Else: unchanged.
    """
    now = datetime.utcnow()
    if start_date.tzinfo is not None:
        start_date = start_date.tz_localize(None)

    if interval in ["1m", "2m"]:
        limit = now - timedelta(days=7)
        if start_date < limit:
            CLAMP_LOG.append({"interval": interval, "from": str(start_date), "to": str(limit)})
            print(f"[Warn] interval={interval} requires start not older than 7 days. Clamping from {start_date} to {limit}")
            return pd.Timestamp(limit)
    elif interval in ["5m", "15m", "30m", "60m", "1h"]:
        limit = now - timedelta(days=60)
        if start_date < limit:
            CLAMP_LOG.append({"interval": interval, "from": str(start_date), "to": str(limit)})
            print(f"[Warn] interval={interval} requires start not older than 60 days. Clamping from {start_date} to {limit}")
            return pd.Timestamp(limit)
    return start_date


# --------------------------- Logging Helper ---------------------------------
def log_error(run_dir: str, msg: str):
    try:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "errors.log"), "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass

SECTOR_MAP = {
    "XLK": "Tech", "XLF": "Financials", "XLE": "Energy", "XLV": "Healthcare",
    "XLY": "ConsumerDisc", "XLP": "ConsumerStap", "XLU": "Utilities",
    "XLI": "Industrials", "XLB": "Materials", "SPY": "Broad", "QQQ": "Broad", "IWM": "Broad", "DIA": "Broad"
}
# --- Johansen pairs global set ---
johansen_pairs = set()

# Defer heavy imports so the script can still show --help without them installed
try:
    import yfinance as yf
    from statsmodels.tsa.stattools import coint, adfuller
    import statsmodels.api as sm
except Exception as e:
    yf = None
    coint = None
    adfuller = None
    sm = None

# Optional plotting and PDF export
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
except Exception:
    SimpleDocTemplate = None


LIQUID_TICKERS = {"SPY","QQQ","IWM","DIA","SPX","XSP"}

# --------------------------- Utilities ---------------------------------
def ensure_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def zscore(series: pd.Series) -> pd.Series:
    mu = series.mean()
    sd = series.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros_like(series), index=series.index)
    return (series - mu) / sd


def ols_beta(y: pd.Series, x: pd.Series) -> float:
    """
    OLS hedge ratio beta for Y ~ a + b * X (we return b).
    Note: beta mode may be overridden by CLI.
    """
    X = sm.add_constant(x.values)
    model = sm.OLS(y.values, X, missing='drop')
    res = model.fit()
    return float(res.params[1])


def engle_granger_pvalue(y: pd.Series, x: pd.Series) -> float:
    """Two-step EG: regress Y~X, test residual ADF using statsmodels.coint shortcut."""
    score, pvalue, _ = coint(y, x)
    return float(pvalue)


def half_life(spread: pd.Series) -> float:
    """Estimate mean-reversion half-life via AR(1) on spread differences.
       ΔS_t = a + b*S_{t-1} + ε  =>  phi = 1 + b ; HL = -ln(2)/ln(phi)
       Fallback to correlation-based phi if regression fails.
    """
    s = spread.dropna()
    if len(s) < 5:
        return np.nan
    y = s.diff().dropna()
    x = s.shift(1).dropna().loc[y.index]
    try:
        X = sm.add_constant(x.values)
        model = sm.OLS(y.values, X)
        res = model.fit()
        b = float(res.params[1])
        phi = 1.0 + b
        if 0 < phi < 1:
            return float(-np.log(2) / np.log(phi))
    except Exception:
        pass

    # Fallback: AR(1) phi via lagged correlation
    try:
        phi = s.autocorr(lag=1)
        if phi is not None and 0 < phi < 1:
            return float(np.log(2) / abs(np.log(phi)))
    except Exception:
        pass
    return np.nan


# --------------------------- Advanced Stats Helpers ---------------------------
def rls_kalman_beta(y: pd.Series, x: pd.Series, lam: float = 0.99, delta0: float = 1e5) -> float:
    """
    Simple recursive least squares (Kalman-style) for time-varying beta between y and x.
    Returns the last beta estimate.
    """
    yv = y.dropna()
    xv = x.dropna()
    idx = yv.index.intersection(xv.index)
    if len(idx) < 10:
        return np.nan
    yv = yv.loc[idx].values.astype(float)
    xv = xv.loc[idx].values.astype(float)
    # RLS initialization
    theta = np.array([0.0, 0.0])  # [intercept, beta]
    P = np.eye(2) * delta0
    for i in range(len(idx)):
        phi = np.array([1.0, xv[i]])
        yhat = float(phi @ theta)
        err = yv[i] - yhat
        K = (P @ phi) / (lam + phi @ P @ phi)
        theta = theta + K * err
        P = (P - np.outer(K, phi) @ P) / lam
    return float(theta[1])

def johansen_trace_stat(df: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1):
    """
    Lightweight wrapper around statsmodels' Johansen test, if available.
    Returns (eigvals, trace_stats) or (None, None) if unavailable or errors.
    """
    try:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        res = coint_johansen(df.dropna(), det_order, k_ar_diff)
        return res.eig, res.lr1
    except Exception:
        return None, None

# --------------------------- Johansen Helper ---------------------------
from statsmodels.tsa.vector_ar.vecm import coint_johansen
def johansen_passes(df: pd.DataFrame, det_order=0, k_ar_diff=1, conf=0.95):
    """
    Returns (passed: bool, trace_r0: float or None, crit_val: float or None)
    """
    try:
        res = coint_johansen(df.dropna(), det_order, k_ar_diff)
        trace_r0 = res.lr1[0]
        crit_idx = {0.90: 0, 0.95: 1, 0.99: 2}[conf]
        crit_val = res.cvt[0, crit_idx]
        return trace_r0 > crit_val, trace_r0, crit_val
    except Exception:
        return False, None, None


@dataclass
class WindowDiagnostic:
    window: int
    pvalue: float
    beta: float
    half_life: float
    z_curr: float
    z_abs: float


@dataclass
class PairResult:
    left: str
    right: str
    best: WindowDiagnostic
    confirm: Optional[WindowDiagnostic]
    conviction_score: float
    conviction_band: str
    signal: str
    notes: str
    action: str
    expiration_guide: str
    option_map: str
    contracts_summary: str
    # v1.6 additions
    stationarity_p: float = np.nan
    stationary: bool = False
    vol_regime: str = "Normal"
    spread_vol: float = np.nan
    suggested_notional: float = np.nan
    # v2.1 diagnostic enrichments
    johansen_trace: float = np.nan
    johansen_crit: float = np.nan
    johansen_pass: bool = False
    flip: str = ""
    corr_drift_spy: float = np.nan
    # growth phase diagnostics
    growth_phase: str = ""
    phase_guidance: str = ""
    # S/R levels
    support_1s: float = np.nan
    resistance_1s: float = np.nan
    support_2s: float = np.nan
    resistance_2s: float = np.nan
    sr_signal: str = ""


# --- Portfolio Planner dataclasses ---
@dataclass
class PositionPlan:
    pair: str
    left: str
    right: str
    signal: str
    beta: float
    z: float
    half_life: float
    conviction: float
    suggested_notional: float
    action: str
    status: str = "pending"  # pending/open/closed

@dataclass
class UnitPlan:
    kind: str  # "CSP" or "CC"
    underlying: str
    dte_min: int
    dte_max: int
    delta: float
    target_shares: int = 100
    note: str = ""
def conviction_score(primary, secondary, z_scale, hl_min, hl_max) -> tuple[float, str]:
    """Return (score, band) based on pvalue, Z alignment, and half-life.
    Band is Low/Medium/High for quick scanning.
    """
    if primary is None:
        return 0.0, "Low"
    score = 0.0
    # P-value contribution
    score += max(0.0, (0.05 - primary.pvalue) * 20)
    # Z-score contribution
    score += min(abs(primary.z_curr) / z_scale, 3.0)
    # Half-life contribution
    if hl_min <= primary.half_life <= hl_max:
        score += 2.0
    # Confirmation
    if secondary is not None:
        score += 1.0
    # Banding
    if score >= 7.0:
        band = "High"
    elif score >= 4.0:
        band = "Medium"
    else:
        band = "Low"
    return score, band


# ----------------- Adaptive Entry Rule (Z, Z-derivatives, regime) -----------------
def adaptive_entry_rule(zscore, spread_vol, dz, ddz, pvbe_band, dvte_liquidity, macro_flag):
    base_threshold = 2.0
    dyn_threshold = base_threshold * (1.0 + 0.5 * (spread_vol > 1.5) - 0.25 * (spread_vol < 0.5))
    if macro_flag in ["spike", "shock"]:
        dyn_threshold += 0.5

    signal = "HOLD"
    conviction = 0.0

    if abs(zscore) >= dyn_threshold:
        if (zscore > 0 and dz > 0) or (zscore < 0 and dz < 0):
            accel_bonus = 0.5 if (ddz * zscore > 0) else 0.0
            band_bonus = 0.5 if pvbe_band in ["upper_breach", "lower_breach"] else 0.0
            liq_bonus = 0.5 if dvte_liquidity < 0 else 0.0
            conviction = 2.0 + accel_bonus + band_bonus + liq_bonus
            if zscore > 0:
                signal = "ENTER_SHORT"
            else:
                signal = "ENTER_LONG"
    return signal, conviction


def suggest_dte(hl: float) -> tuple[str, str]:
    """Return (liquid_leg_dte, illiquid_leg_dte) suggestion based on half-life."""
    if np.isnan(hl):
        return ("15–45 DTE", "60–120 DTE")
    if hl <= 5:
        return ("7–30 DTE", "30–60 DTE")
    if hl <= 20:
        return ("15–45 DTE", "45–90 DTE")
    return ("30–60 DTE", "60–120+ DTE")

def leg_liquidity(ticker: str) -> str:
    return "high" if ticker.upper() in LIQUID_TICKERS else "low"

def compute_contracts(beta: float, delta: float) -> tuple[int, int]:
    """Return minimal integer (left_contracts, right_contracts) that respect the hedge ratio using a given option delta.
    Left weight is +1, right weight is |beta|. We scale so the smaller raw contract count becomes 1.
    """
    left_raw = abs(1.0) / max(delta, 1e-6)
    right_raw = abs(beta) / max(delta, 1e-6)
    # if one side is zero (shouldn't happen), fall back to 1
    left_raw = max(left_raw, 1e-9)
    right_raw = max(right_raw, 1e-9)
    scale = 1.0 / min(left_raw, right_raw)
    left_c = int(round(left_raw * scale))
    right_c = int(round(right_raw * scale))
    left_c = max(left_c, 1)
    right_c = max(right_c, 1)
    return left_c, right_c

def build_option_mapping(left: str, right: str, beta: float, hl: float, delta_atm: float, delta_ditm: float) -> tuple[str, str, str]:
    """Return (expiration_guide, option_map, contracts_summary) strings.
    - expiration_guide: concise text derived from HL and liquidity
    - option_map: per-leg suggestion ATM vs Deep ITM with DTE windows
    - contracts_summary: "ATM: Lx left vs Rx right | DITM: Lx left vs Rx right"
    """
    liquid_dte, illiq_dte = suggest_dte(hl)
    l_liq = leg_liquidity(left)
    r_liq = leg_liquidity(right)
    # Per-leg style choice
    left_style = "ATM" if l_liq == "high" and hl <= 20 else "Deep ITM"
    right_style = "ATM" if r_liq == "high" and hl <= 20 else "Deep ITM"
    left_dte = liquid_dte if l_liq == "high" else illiq_dte
    right_dte = liquid_dte if r_liq == "high" else illiq_dte

    # Contracts mapping (normalized minimal integer counts)
    l_atm, r_atm = compute_contracts(beta, delta_atm)
    l_ditm, r_ditm = compute_contracts(beta, delta_ditm)

    expiration_guide = f"HL≈{hl:.1f} → Liquid: {liquid_dte}; Illiquid: {illiq_dte}"
    option_map = f"{left}: {left_style} {left_dte}; {right}: {right_style} {right_dte}"
    contracts_summary = f"ATM(Δ≈{delta_atm:.2f}): {l_atm}×{left} vs {r_atm}×{right} | DITM(Δ≈{delta_ditm:.2f}): {l_ditm}×{left} vs {r_ditm}×{right}"
    return expiration_guide, option_map, contracts_summary



# ---------------------- Data Caching Layer ---------------------------
DATA_CACHE_DIR = "./data_cache"

def ensure_cache_dir():
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

def cache_path(t: str) -> str:
    return os.path.join(DATA_CACHE_DIR, f"{t.upper()}.parquet")

def fetch_yf_single(ticker: str, start: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, interval=interval, progress=False, auto_adjust=False)
    s = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    s = s.dropna()
    s.name = ticker.upper()
    # Safer return: handle both Series and DataFrame
    if isinstance(s, pd.Series):
        return s.to_frame()
    elif isinstance(s, pd.DataFrame):
        return s
    else:
        raise TypeError(f"Unexpected type from yfinance: {type(s)}")

def load_prices_cached_yf(tickers: List[str], start: str, interval: str, refresh: bool=False) -> pd.DataFrame:
    ensure_cache_dir()
    frames = []
    for t in tickers:
        tU = t.upper()
        cp = cache_path(tU)
        cached = None
        if os.path.exists(cp) and not refresh:
            try:
                cached = pd.read_parquet(cp)
            except Exception:
                cached = None
        if cached is not None and not cached.empty:
            last_dt = cached.index.max()
            try:
                tail = fetch_yf_single(tU, (last_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), interval)
                merged = pd.concat([cached, tail], axis=0)
                merged = merged[~merged.index.duplicated(keep='last')]
            except Exception:
                merged = cached
        else:
            merged = fetch_yf_single(tU, start, interval)
        try:
            merged.sort_index(inplace=True)
            merged.to_parquet(cp)
        except Exception:
            pass
        frames.append(merged.rename(columns={merged.columns[0]: tU}))
    # Fallback and error handling:
    non_empty = [f for f in frames if not f.empty]
    if non_empty:
        return pd.concat(non_empty, axis=1)

    # fallback if possible
    if interval in INTERVAL_FALLBACK:
        fallback = INTERVAL_FALLBACK[interval]
        FALLBACK_LOG.append({"from": interval, "to": fallback, "start": str(start), "tickers": len(tickers)})
        print(f"[Warn] No data for interval={interval}, start={start}. Falling back to {fallback}...")
        return load_prices_cached_yf(tickers, start, fallback, refresh=refresh)

    # if no fallback defined
    raise ValueError(f"[Error] No data returned for tickers at interval={interval}, start={start}. No fallback available.")

def load_prices(tickers: List[str], start: str, interval: str,
                source: str = "yfinance", use_cache: bool = True, refresh: bool = False) -> pd.DataFrame:
    if source == "local":
        raise RuntimeError("Use load_prices_local() for local source")
    if yf is None:
        raise RuntimeError("yfinance not installed. Please `pip install yfinance statsmodels`")
    # Clamp start for intraday intervals
    if interval in ["1m","2m","5m","15m","30m","60m","1h"]:
        start = clamp_start_for_interval(interval, pd.to_datetime(start))
    if use_cache:
        df = load_prices_cached_yf(tickers, start, interval, refresh=refresh)
    else:
        frames = [fetch_yf_single(t, start, interval) for t in tickers]
        df = pd.concat(frames, axis=1)
    return df.dropna(how="all")

def load_prices_local(tickers: List[str], local_dir: str) -> pd.DataFrame:
    """Load price data from local CSV files in a directory.
    Each file should be named <ticker>.csv and contain 'Date' and 'Adj Close' columns.
    """
    dfs = []
    for ticker in tickers:
        filepath = os.path.join(local_dir, f"{ticker}.csv")
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Local price file not found: {filepath}")
        df = pd.read_csv(filepath, parse_dates=['Date'])
        if 'Adj Close' not in df.columns:
            raise ValueError(f"'Adj Close' column not found in {filepath}")
        df = df[['Date', 'Adj Close']].rename(columns={'Adj Close': ticker}).set_index('Date')
        dfs.append(df)
    combined = pd.concat(dfs, axis=1)
    combined = combined.dropna(how='all')
    return combined


def read_liquidity_scores(path: Optional[str]) -> dict:
    scores = {}
    if not path: return scores
    try:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            t = str(row.get("ticker", "")).upper().strip()
            s = row.get("score", None)
            if t and pd.notna(s):
                try:
                    scores[t] = int(s)
                except Exception:
                    pass
    except Exception:
        pass
    return scores


def apply_liquidity_filter(tickers: List[str], min_score: int, score_map: dict) -> List[str]:
    if min_score is None: return tickers
    anchors = {"SPY", "QQQ"}
    kept = []
    for t in tickers:
        if t.upper() in anchors:
            kept.append(t)
            continue
        if score_map.get(t.upper(), 0) >= min_score:
            kept.append(t)
    for a in anchors:
        if a in tickers and a not in kept:
            kept.append(a)
    return kept

# --- Portfolio Planner helpers ---
def parse_range_to_ints(vals):
    # expects a list like [15,45] or a str "15 45"
    if vals is None: return (15, 45)
    if isinstance(vals, (list, tuple)) and len(vals) >= 2:
        return int(vals[0]), int(vals[1])
    if isinstance(vals, str):
        parts = vals.split()
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])
    return (15, 45)

def plan_positions(results: List[PairResult],
                   equity: Optional[float],
                   per_ticker_cap: Optional[float],
                   max_concurrent: int) -> List[PositionPlan]:
    plans = []
    if not results:
        return plans
    used_per_ticker = {}
    open_count = 0
    for r in results:
        if max_concurrent and open_count >= max_concurrent:
            break
        # skip if no sizing available
        if equity is None or pd.isna(r.suggested_notional) or r.suggested_notional <= 0:
            continue
        # enforce per-ticker cap on both legs
        cap_ok = True
        if per_ticker_cap and equity:
            cap_dollars = per_ticker_cap * equity
            for sym in (r.left, r.right):
                if used_per_ticker.get(sym, 0.0) + r.suggested_notional > cap_dollars:
                    cap_ok = False
                    break
        if not cap_ok:
            continue

        plans.append(
            PositionPlan(
                pair=f"{r.left}-{r.right}",
                left=r.left,
                right=r.right,
                signal=r.signal,
                beta=float(r.best.beta),
                z=float(r.best.z_curr),
                half_life=float(r.best.half_life),
                conviction=float(r.conviction_score),
                suggested_notional=float(r.suggested_notional),
                action=r.action,
                status="pending"
            )
        )
        # update usage (simple add; conservative)
        for sym in (r.left, r.right):
            used_per_ticker[sym] = used_per_ticker.get(sym, 0.0) + float(r.suggested_notional)
        open_count += 1
    return plans

def save_position_plans(run_dir: str, plans: List[PositionPlan]):
    if not plans:
        return
    import csv, os
    path = os.path.join("./runs", "positions.csv")
    os.makedirs("./runs", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["RunDir","Timestamp","Pair","Left","Right","Signal","Beta","Z","HalfLife","Conviction","SuggestedNotional","Action","Status"])
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for p in plans:
            w.writerow([run_dir, ts, p.pair, p.left, p.right, p.signal, p.beta, p.z, p.half_life, p.conviction, p.suggested_notional, p.action, p.status])

def load_held_shares() -> dict:
    """
    Minimal placeholder: read ./runs/held_shares.csv if present with columns: ticker,shares
    Returns dict[ticker]->int
    """
    path = os.path.join("./runs", "held_shares.csv")
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
        return {str(row["ticker"]).upper(): int(row["shares"]) for _, row in df.iterrows() if pd.notna(row.get("ticker")) and pd.notna(row.get("shares"))}
    except Exception:
        return {}

def plan_csp(universe: List[str], targets: dict, equity: Optional[float],
             dte_range: Tuple[int,int], delta: float) -> List[UnitPlan]:
    if not universe or not targets:
        return []
    plans = []
    # Limit to highly liquid underlyings present in our universe
    liquid = [t for t in universe if t.upper() in LIQUID_TICKERS]
    for sym, tgt in targets.items():
        symu = sym.upper()
        if symu not in liquid:
            continue
        plans.append(UnitPlan(kind="CSP", underlying=symu, dte_min=dte_range[0], dte_max=dte_range[1], delta=float(delta), target_shares=int(tgt), note="Auto-generated CSP candidate"))
    return plans

def plan_cc(held: dict, dte_range: Tuple[int,int], delta: float) -> List[UnitPlan]:
    if not held:
        return []
    plans = []
    for sym, qty in held.items():
        if qty <= 0:
            continue
        # Only generate for lots of at least 100 shares
        lots = qty // 100
        if lots <= 0:
            continue
        plans.append(UnitPlan(kind="CC", underlying=str(sym).upper(), dte_min=dte_range[0], dte_max=dte_range[1], delta=float(delta), target_shares=lots*100, note="Auto-generated CC on held shares"))
    return plans

def save_units_watchlist(run_dir: str, csp_plans: List[UnitPlan], cc_plans: List[UnitPlan]):
    import os, csv
    rows = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for p in csp_plans:
        rows.append([run_dir, ts, p.kind, p.underlying, p.dte_min, p.dte_max, p.delta, p.target_shares, p.note])
    for p in cc_plans:
        rows.append([run_dir, ts, p.kind, p.underlying, p.dte_min, p.dte_max, p.delta, p.target_shares, p.note])
    if not rows:
        return
    os.makedirs("./runs", exist_ok=True)
    path = os.path.join("./runs", "units_watchlist.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["RunDir","Timestamp","Kind","Underlying","DTE_Min","DTE_Max","Delta","TargetShares","Note"])
        w.writerows(rows)

# ---------------------- Universe Construction Helper ---------------------------
def build_universe(args):
    """
    Build the trading universe based on args, presets, liquidity, and deduplication.
    This function should be called after applying any liquidity/pruning logic and before loading prices.
    Returns the final universe as a list of uppercase tickers.
    """
    # Start with the universe from args (already possibly filtered by liquidity)
    universe = list(args.universe) if hasattr(args, "universe") else []
    # If any preset is active and universe is empty, fill with preset universe
    if getattr(args, "preset_spy_scan", False) and not universe:
        universe = ["SPY", "QQQ"]
    elif getattr(args, "preset_sector_scan", False) and not universe:
        universe = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLY", "XLU"]
    elif getattr(args, "preset_macro_scan", False) and not universe:
        universe = ["SPY","QQQ","DIA","IWM","XLK","XLF","XLE","XLV","XLY","XLU",
                    "TLT","IEF","SHY","GLD","SLV","UUP","VIXY","XHB","KRE","XLI","XLB"]
    elif getattr(args, "preset_fedday", False) and not universe:
        universe = ["SPY","QQQ","IWM","DIA","XLK","XLF","XLE","XLV","XLY","XLP",
                    "TLT","IEF","SHY","KRE","XHB","GLD","UUP","VIXY"]
    elif getattr(args, "preset_alpha_scan", False) and not universe:
        universe = ["SPY", "QQQ", "IWM"]
    # Always deduplicate and uppercase
    universe = list({t.upper() for t in universe})
    # If event config provided a universe, that should take precedence (already handled in main)
    # Optionally enforce special rules (anchors always present)
    anchors = {"SPY", "QQQ"}
    for a in anchors:
        if a in args.universe and a not in universe:
            universe.append(a)
    # Return as sorted list for consistency
    return sorted(universe)

def build_knn_pairs(px: pd.DataFrame, anchors: list, k: int) -> set:
    if not k or k <= 0: return set()
    rets = px.pct_change()
    corr = rets.corr().fillna(0.0)
    pairs = set()
    cols = list(px.columns)
    for t in cols:
        neigh = corr[t].drop(index=t).sort_values(ascending=False).head(k).index.tolist()
        for n in neigh:
            a, b = sorted([t, n])
            pairs.add((a, b))
    for a in anchors:
        if a in cols:
            for t in cols:
                if t == a: continue
                x, y = sorted([a, t])
                pairs.add((x, y))
    return pairs


# v1.6: Regime flags helper
def regime_flags(spread: pd.Series) -> tuple[float, bool, str, float]:
    """Return (adf_p, stationary, vol_regime, spread_vol).
    vol_regime based on 20-day rolling std vs 60-day mean std.
    spread_vol = std of spread daily changes (diff) as a risk proxy.
    """
    s = spread.dropna()
    adf_p = np.nan
    stationary = False
    vol_regime = "Normal"
    spread_vol = np.nan
    if len(s) >= 30:
        try:
            adf_p = float(adfuller(s, autolag='AIC')[1])
            stationary = adf_p < 0.10
        except Exception:
            pass
        try:
            roll20 = s.rolling(20).std().iloc[-1]
            mean60 = s.rolling(60).std().mean()
            if pd.notna(roll20) and pd.notna(mean60) and mean60 > 0:
                ratio = float(roll20 / mean60)
                if ratio >= 2.0:
                    vol_regime = "Spike"
                elif ratio <= 0.5:
                    vol_regime = "Low"
                else:
                    vol_regime = "Normal"
        except Exception:
            pass
        try:
            spread_vol = float(s.diff().std())
        except Exception:
            pass
    return adf_p, stationary, vol_regime, spread_vol


def growth_phase(z_history: list[float], half_life: float, beta: float) -> tuple[str, str]:
    """
    Classify the regime of a pair into one of four growth-cycle phases.
    Inputs:
      - z_history: rolling Z-score history
      - half_life: estimated reversion horizon
      - beta: sensitivity of spread
    Outputs:
      - phase: Exponential / Maturity / Deterioration / Reset
      - guidance: textual advice for trading in that regime
    """
    z_arr = np.asarray(z_history, dtype=float).ravel()
    z_arr = z_arr[~np.isnan(z_arr)]

    if z_arr.size < 5:
        return "Unknown", "Insufficient history"

    recent_n = min(10, z_arr.size)
    try:
        slope = float(np.polyfit(np.arange(z_arr.size), z_arr, 1)[0])
    except Exception:
        slope = 0.0

    variance = float(np.var(z_arr[-recent_n:]))
    latest_z = float(z_arr[-1])

    if slope > 0.2 and latest_z > 1 and half_life < 5:
        return "Exponential", "Trade smaller divergences, size up, mean reverts quickly"
    elif abs(slope) < 0.1 and variance < 0.5 and 3 <= half_life <= 7:
        return "Maturity", "Stick to standard entries, HL stable, clean mean reverts"
    elif (slope < -0.2 or variance > 1.0) and half_life < 4:
        return "Deterioration", "Tight entries, faster exits, entropy high"
    elif abs(latest_z) > 2.5 or half_life > 8:
        return "Reset", "Expect longer holding periods, reversion is slower"
    else:
        return "Maturity", "Default regime"


def evaluate_pair(px: pd.DataFrame, y: str, x: str, windows: List[int],
                  pvalue_cut: float, hl_min: int, hl_max: int,
                  z_enter: float, z_scale: float, z_exit: float,
                  delta_atm: float, delta_ditm: float,
                  account_equity: Optional[float], risk_unit: float) -> Optional[PairResult]:
    diags: List[WindowDiagnostic] = []

    for W in windows:
        sub = px[[y, x]].dropna().tail(W)
        if len(sub) < max(20, W // 2):
            continue
        try:
            pv = engle_granger_pvalue(sub[y], sub[x])
            if pv >= pvalue_cut:
                continue
            beta = ols_beta(sub[y], sub[x]) if getattr(args_global, "beta_mode", "ols") == "ols" else rls_kalman_beta(sub[y], sub[x])
            if pd.isna(beta):
                return None
            spread = sub[y] - beta * sub[x]
            hl = half_life(spread)
            if not (hl_min <= hl <= hl_max):
                continue
            z = float(zscore(spread).iloc[-1])
            diags.append(WindowDiagnostic(W, pv, beta, hl, z, abs(z)))
        except Exception:
            continue

    if not diags:
        return None

    diags.sort(key=lambda d: (d.pvalue, -d.z_abs))
    primary = diags[0]
    secondary = next((d for d in diags[1:] if np.sign(d.z_curr) == np.sign(primary.z_curr)), None)

    z = primary.z_curr
    beta = primary.beta
    sub = px[[y, x]].dropna().tail(primary.window)
    spread = sub[y] - beta * sub[x]
    z_history = zscore(spread).values

    sr_levels = compute_sr_levels(spread)
    s1, r1, s2, r2 = sr_levels["Support_1s"], sr_levels["Resistance_1s"], sr_levels["Support_2s"], sr_levels["Resistance_2s"]
    
    current_spread = spread.iloc[-1]
    if current_spread > r2 or current_spread < s2:
        sr_signal = "Beyond ±2s"
    elif current_spread > r1 or current_spread < s1:
        sr_signal = "Beyond ±1s"
    else:
        sr_signal = "Inside Band"

    phase, guidance = growth_phase(z_history, primary.half_life, beta)
    adf_p, stationary, vol_regime, spread_vol = regime_flags(spread)
    macro_flag = (vol_regime or "Normal").lower()
    dz = np.gradient(z_history)[-1] if len(z_history) > 1 else 0.0
    ddz = np.gradient(np.gradient(z_history))[-1] if len(z_history) > 2 else 0.0
    
    pvbe_band, dvte_score = "inside", 0.0
    signal, conviction = adaptive_entry_rule(z, spread_vol, dz, ddz, pvbe_band, dvte_score, macro_flag)
    
    notes = []
    action = "HOLD"
    if signal != "HOLD" and conviction >= 2.5:
        action = signal
        notes.append(f"AdaptiveEntry: Conv={conviction:.2f}")

    if (signal == "ENTER_LONG" and y.upper() == "SPY") or (signal == "ENTER_SHORT" and x.upper() == "SPY"):
        signal = "LONG_SPY_OPPORTUNITY"

    cv_score, cv_band = conviction_score(primary, secondary, z_scale, hl_min, hl_max)

    j_pass, j_trace, j_crit = False, np.nan, np.nan
    try:
        sub_for_j = px[[y, x]].dropna().tail(primary.window)
        if len(sub_for_j) >= 60:
            j_pass, j_trace, j_crit = johansen_passes(sub_for_j, conf=0.95)
            if getattr(args_global, "johansen_trace_threshold", None) is not None and j_trace is not None:
                j_pass = bool(j_trace > float(args_global.johansen_trace_threshold))
    except Exception:
        pass

    if getattr(args_global, "johansen_filter", False) and not j_pass:
        return None

    pending_j_bonus = 0.0
    if getattr(args_global, "johansen_bonus", False) and j_trace is not None and j_crit is not None:
        johansen_gap = max(0.0, float(j_trace) - float(j_crit))
        pending_j_bonus = min(2.0, johansen_gap / 5.0)

    notes_str = f"β={beta:.3f}; HL={primary.half_life:.1f}; Z={z:.2f}; p={primary.pvalue:.4f}; W={primary.window}"
    if notes:
        notes_str += "; " + "; ".join(notes)

    suggested_notional, realized_vol = np.nan, np.nan
    if account_equity and pd.notna(spread_vol) and spread_vol > 0:
        risk_dollars = float(account_equity) * float(risk_unit)
        suggested_notional = risk_dollars / spread_vol
        if getattr(args_global, "target_vol", None):
            realized_vol = spread.pct_change().std()
            if pd.notna(realized_vol) and realized_vol > 0:
                suggested_notional *= (args_global.target_vol / realized_vol)
        if pd.notna(suggested_notional):
            size_factor = max(0.5, min(1.5, (conviction or 0.0) / 5.0))
            suggested_notional *= size_factor

    try:
        vol_for_conv = realized_vol if pd.notna(realized_vol) else spread.pct_change().std()
        vol_adj = max(0.5, 1.0 - min(0.5, float(vol_for_conv) / 0.05)) if pd.notna(vol_for_conv) and vol_for_conv > 0 else 1.0
    except Exception:
        vol_adj = 1.0
    
    cv_score = (cv_score + pending_j_bonus) * vol_adj
    if cv_score >= 7.0: cv_band = "High"
    elif cv_score >= 4.0: cv_band = "Medium"
    else: cv_band = "Low"

    exp_guide, opt_map, contracts = build_option_mapping(y, x, beta, primary.half_life, delta_atm, delta_ditm)
    
    return PairResult(
        y, x, primary, secondary, cv_score, cv_band, signal, notes_str, action,
        exp_guide, opt_map, contracts,
        adf_p, stationary, vol_regime, spread_vol, suggested_notional,
        float(j_trace) if j_trace is not None else np.nan,
        float(j_crit) if j_crit is not None else np.nan,
        bool(j_pass), "", np.nan,
        phase, guidance,
        s1, r1, s2, r2, sr_signal
    )


def log_pnl(run_dir, results, account_equity):
    """
    Log estimated PnL for each pair in the run based on risk-adjusted Z-score and spread volatility.
    Writes per-pair rows to pnl_history.csv in ./runs directory.
    """
    import os
    import pandas as pd
    pnl_hist_path = os.path.join("./runs", "pnl_history.csv")
    rows = []
    for r in results:
        # Estimate PnL as risk-adjusted move: notional * (Z / (hl-adjusted scale))
        scale = max(abs(r.best.half_life), 1.0)
        est_pnl = (r.suggested_notional or 0) * (r.best.z_curr / scale)
        rows.append({
            "RunDir": run_dir,
            "Pair": f"{r.left}-{r.right}",
            "Signal": r.signal,
            "Z": r.best.z_curr,
            "HalfLife": r.best.half_life,
            "SpreadVol": r.spread_vol,
            "SuggestedNotional": r.suggested_notional,
            "PnL": est_pnl,
            "AccountEquity": account_equity,
        })
    try:
        df = pd.DataFrame(rows)
        exists = os.path.isfile(pnl_hist_path)
        if not exists:
            df.to_csv(pnl_hist_path, index=False, mode="w")
        else:
            df.to_csv(pnl_hist_path, index=False, mode="a", header=False)
    except Exception as e:
        msg = f"[StatArb] Failed to log PnL: {e}"
        print(msg)
        log_error(run_dir, msg)

#
# ---------------- Performance Metrics Helper ----------------
def update_performance_metrics(run_dir):
    """Compute extended performance metrics and save to CSVs"""
    import os
    import pandas as pd
    import numpy as np
    pnl_path = os.path.join("./runs", "pnl_history.csv")
    eq_path = os.path.join("./runs", "equity_curve.csv")
    perf_path = os.path.join("./runs", "performance.csv")
    hist_path = os.path.join("./runs", "performance_history.csv")
    rolling_path = os.path.join("./runs", "performance_rolling.csv")

    if not os.path.exists(pnl_path) or not os.path.exists(eq_path):
        return
    try:
        pnl_df = pd.read_csv(pnl_path)
        eq_df = pd.read_csv(eq_path)
        if pnl_df.empty or eq_df.empty:
            return

        # --- Base metrics
        mean_pnl = pnl_df["PnL"].mean()
        std_pnl = pnl_df["PnL"].std(ddof=0)
        sharpe = mean_pnl / std_pnl if std_pnl > 0 else None

        downside = pnl_df[pnl_df["PnL"] < 0]["PnL"].std(ddof=0)
        sortino = mean_pnl / downside if downside and downside > 0 else None

        win_rate = float((pnl_df["PnL"] > 0).mean())
        avg_win = pnl_df.loc[pnl_df["PnL"] > 0, "PnL"].mean() if not pnl_df[pnl_df["PnL"] > 0].empty else 0.0
        avg_loss = abs(pnl_df.loc[pnl_df["PnL"] < 0, "PnL"].mean()) if not pnl_df[pnl_df["PnL"] < 0].empty else 0.0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        avg_pnl = float(mean_pnl)

        equity_cummax = eq_df["EquityEnd"].cummax()
        drawdowns = (eq_df["EquityEnd"] - equity_cummax) / equity_cummax
        max_drawdown = float(drawdowns.min())

        # CAGR
        days = (pd.to_datetime(eq_df["Date"]).iloc[-1] - pd.to_datetime(eq_df["Date"]).iloc[0]).days
        years = days / 365.25 if days > 0 else 0
        start_val = eq_df["EquityEnd"].iloc[0]
        end_val = eq_df["EquityEnd"].iloc[-1]
        cagr = (end_val / start_val) ** (1 / years) - 1 if years > 0 and start_val > 0 else None

        # Volatility of returns
        vol = std_pnl / eq_df["EquityEnd"].mean() if not eq_df.empty else None

        perf = {
            "Sharpe": sharpe,
            "Sortino": sortino,
            "WinRate": win_rate,
            "AvgPnL": avg_pnl,
            "Expectancy": expectancy,
            "MaxDrawdown": max_drawdown,
            "CAGR": cagr,
            "Volatility": vol
        }

        # Save single-run snapshot
        pd.DataFrame([perf]).to_csv(perf_path, index=False)

        # Append to performance history
        mode = "a" if os.path.exists(hist_path) else "w"
        pd.DataFrame([perf]).to_csv(hist_path, mode=mode, header=(mode=="w"), index=False)

        # Rolling Sharpe/Sortino (30 trades)
        window = 30
        rolling = []
        pnl_series = pnl_df["PnL"].reset_index(drop=True)
        for i in range(window, len(pnl_series)+1):
            sub = pnl_series[i-window:i]
            mean_sub = sub.mean()
            std_sub = sub.std(ddof=0)
            sharpe_sub = mean_sub / std_sub if std_sub > 0 else None
            downside_sub = sub[sub < 0].std(ddof=0)
            sortino_sub = mean_sub / downside_sub if downside_sub and downside_sub > 0 else None
            rolling.append({
                "Index": i,
                "RollingSharpe": sharpe_sub,
                "RollingSortino": sortino_sub
            })
        if rolling:
            pd.DataFrame(rolling).to_csv(rolling_path, index=False)

    except Exception as e:
        msg = f"[StatArb] Failed to update performance metrics: {e}"
        print(msg)
        log_error(run_dir, msg)

def update_equity_curve(run_dir):
    """Build equity_curve.csv from pnl_history.csv if account_equity is available"""
    import os, pandas as pd
    pnl_path = os.path.join("./runs", "pnl_history.csv")
    eq_path = os.path.join("./runs", "equity_curve.csv")
    if not os.path.exists(pnl_path):
        return
    try:
        pnl_df = pd.read_csv(pnl_path)
        if pnl_df.empty or "PnL" not in pnl_df.columns:
            return
        pnl_df["CumPnL"] = pnl_df["PnL"].cumsum()
        # If AccountEquity column exists, use its first value as base
        base_equity = pnl_df["AccountEquity"].iloc[0] if "AccountEquity" in pnl_df.columns else 100000.0
        pnl_df["EquityEnd"] = base_equity + pnl_df["CumPnL"]
        pnl_df["Date"] = pd.to_datetime("today")
        eq_df = pnl_df[["Date","EquityEnd"]]
        eq_df.to_csv(eq_path, index=False)
    except Exception as e:
        msg = f"[StatArb] Failed to update equity curve: {e}"
        print(msg)
        log_error(run_dir, msg)

def log_run_ledger(run_dir, args, results, sector_usage, circuit_breaker=False):
    """
    Append one-row summary of the run into ./runs/run_ledger.csv
    Includes parameters, counts, notional usage, and circuit breaker state.
    """
    import os, pandas as pd
    import uuid
    run_id = str(uuid.uuid4())[:8]
    ledger_path = os.path.join("./runs", "run_ledger.csv")
    row = {
        "RunID": run_id,
        "RunDir": run_dir,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "MacroEvent": args.macro_event or "",
        "Preset": (
            "SPY Scan" if args.preset_spy_scan else
            "Sector Scan" if args.preset_sector_scan else
            "Macro Scan" if args.preset_macro_scan else
            "Fed Day" if args.preset_fedday else
            args.preset or ""
        ),
        # Parameters
        "HLBand": f"{args.hl_min}-{args.hl_max}",
        "Z_enter": args.z_enter,
        "BetaMode": args.beta_mode,
        "RiskUnit": args.risk_unit,
        # Results summary
        "PairsFound": len(results),
        "Signals": ",".join(sorted(set(r.signal for r in results))),
        "HighConviction": sum(1 for r in results if r.conviction_band=="High"),
        "MediumConviction": sum(1 for r in results if r.conviction_band=="Medium"),
        "JohansenPasses": sum(1 for r in results if r.johansen_pass),
        # Risk usage
        "TotalNotional": sum(r.suggested_notional for r in results if pd.notna(r.suggested_notional)),
        "SectorUsage": _json.dumps(sector_usage),
        "CircuitBreaker": circuit_breaker,
        "Note": args.note or ""
    }
    
    # --- Sector exposure summary ---
    if sector_usage:
        max_sector = max(sector_usage.items(), key=lambda kv: kv[1])
        row.update({
            "MaxSector": max_sector[0],
            "MaxSectorUsage": max_sector[1],
            "NumSectorsActive": len(sector_usage)
        })
    
    # --- Performance snapshot enrichment ---
    perf_path = os.path.join("./runs", "performance.csv")
    if os.path.exists(perf_path):
        try:
            perf_df = pd.read_csv(perf_path)
            if not perf_df.empty:
                last_perf = perf_df.iloc[-1].to_dict()
                row.update({
                    "Sharpe": last_perf.get("Sharpe"),
                    "Sortino": last_perf.get("Sortino"),
                    "WinRate": last_perf.get("WinRate"),
                    "Expectancy": last_perf.get("Expectancy"),
                    "CAGR": last_perf.get("CAGR"),
                    "Volatility": last_perf.get("Volatility"),
                    "MaxDrawdown": last_perf.get("MaxDrawdown"),
                })
        except Exception as e:
            msg = f"[StatArb] Failed to attach performance snapshot: {e}"
            print(msg)
            log_error(run_dir, msg)

    # --- Rolling performance enrichment ---
    rolling_path = os.path.join("./runs", "performance_rolling.csv")
    if os.path.exists(rolling_path):
        try:
            roll_df = pd.read_csv(rolling_path)
            if not roll_df.empty:
                last_roll = roll_df.iloc[-1].to_dict()
                row.update({
                    "RollingSharpe": last_roll.get("RollingSharpe"),
                    "RollingSortino": last_roll.get("RollingSortino"),
                })
        except Exception as e:
            msg = f"[StatArb] Failed to attach rolling performance snapshot: {e}"
            print(msg)
            log_error(run_dir, msg)
    
    # --- Diagnostics enrichment summary ---
    diag_path = os.path.join(run_dir, "diagnostics.csv")
    if os.path.exists(diag_path):
        try:
            diag_df = pd.read_csv(diag_path)
            row["Flips"] = (diag_df["Flip"] == "Yes").sum() if "Flip" in diag_df.columns else 0
            row["AvgCorrDriftSPY"] = diag_df["CorrDriftSPY"].mean() if "CorrDriftSPY" in diag_df.columns else None
        except Exception as e:
            msg = f"[StatArb] Failed to attach diagnostics summary: {e}"
            print(msg)
            log_error(run_dir, msg)
    try:
        df = pd.DataFrame([row])
        exists = os.path.isfile(ledger_path)
        df.to_csv(ledger_path, mode="a", header=not exists, index=False)
        print(f"[StatArb] Run ledger updated: {os.path.abspath(ledger_path)}")
    except Exception as e:
        msg = f"[StatArb] Failed to update run ledger: {e}"
        print(msg)
        log_error(run_dir, msg)

args_global = None

def compute_sr_levels(spread: pd.Series):
    mu = spread.mean()
    sigma = spread.std()
    return {
        "MA": mu,
        "Support_1s": mu - sigma,
        "Resistance_1s": mu + sigma,
        "Support_2s": mu - 2*sigma,
        "Resistance_2s": mu + 2*sigma,
    }

def export_pair_timeseries(px: pd.DataFrame, left: str, right: str, window: int, beta: float, run_dir: str):
    """
    Export spread and Z-score series for a given pair into CSV.
    """
    try:
        sub = px[[left, right]].dropna().tail(window)
        spread = sub[left] - beta * sub[right]
        z = zscore(spread)
        sr = compute_sr_levels(spread)
        
        df = pd.DataFrame({
            "Spread": spread,
            "ZScore": z,
            "MA": sr["MA"],
            "Support_1s": sr["Support_1s"],
            "Resistance_1s": sr["Resistance_1s"],
            "Support_2s": sr["Support_2s"],
            "Resistance_2s": sr["Resistance_2s"]
        })
        out_path = os.path.join(run_dir, f"{left}-{right}_spread_timeseries.csv")
        df.to_csv(out_path)
        return out_path
    except Exception as e:
        raise RuntimeError(f"Failed to export pair timeseries: {e}")

# ---------------------------------------------------------------------------------
# --- START: MARKET INTERNALS ROLLUP (SPY vs RSP Health)
# ---------------------------------------------------------------------------------

def market_internals_rollup(watchlist_df: pd.DataFrame, run_dir: str):
    """
    Calculates sector rollups and computes SPY (cap-weighted) vs. RSP (equal-weighted)
    health scores to measure market breadth. Saves the output to a CSV file.
    """
    # Configuration for the rollup
    sector_map = {
        "Tech":      ["NVDA", "MSFT", "AAPL", "GOOG", "GOOGL", "AMZN", "META", "AVGO"],
        "Defensive": ["JNJ", "ABBV", "UNH", "WMT", "PG", "KO"],
        "Financials":["V", "JPM", "BRK-B", "MA"],
        "EnergyInd": ["XOM", "CVX", "T", "TMUS", "NEE"]
    }

    phase_bias_map = {
        "Exponential":   1.0,
        "Maturity":      0.5,
        "Deterioration":-1.0,
        "Reset":        -0.5,
        "Unknown":       0.0
    }

    weights_spy = {"Tech": 0.45, "Defensive": 0.25, "Financials": 0.20, "EnergyInd": 0.10}
    weights_rsp = {"Tech": 0.25, "Defensive": 0.25, "Financials": 0.25, "EnergyInd": 0.25}

    # Helper function for sector calculation
    def rollup_sector(df, name, tickers):
        # We need to match based on the pair, as watchlist_df has Left/Right columns
        sub = df[df["Left"].isin(tickers) | df["Right"].isin(tickers)].copy()
        if sub.empty:
            return None

        sub['PhaseBias'] = sub["GrowthPhase"].map(phase_bias_map).fillna(0)
        bias = sub['PhaseBias'].mean()

        breakout = (sub["GrowthPhase"].isin(["Exponential"])).mean()
        exhaustion = (sub["GrowthPhase"].isin(["Deterioration"])).mean()
        
        return {
            "Sector": name,
            "%Breakout": f"{breakout:.0%}",
            "%Exhaustion": f"{exhaustion:.0%}",
            "PhaseBias": round(bias, 2)
        }

    # Main logic
    rows = []
    for sector, tickers in sector_map.items():
        out = rollup_sector(watchlist_df, sector, tickers)
        if out:
            rows.append(out)
    
    if not rows:
        print("[MarketInternals] No matching tickers found in watchlist to build the rollup.")
        return
        
    roll = pd.DataFrame(rows).set_index("Sector")

    spy_health = (roll["PhaseBias"] * pd.Series(weights_spy)).sum()
    rsp_health = (roll["PhaseBias"] * pd.Series(weights_rsp)).sum()
    spread = spy_health - rsp_health
    
    summary_data = {
        "SPY Health": spy_health,
        "RSP Health": rsp_health,
        "Spread (SPY–RSP)": spread
    }
    
    for name, value in summary_data.items():
        roll.loc[name] = {"%Breakout": "—", "%Exhaustion": "—", "PhaseBias": round(value, 2)}
        
    result_df = roll.reset_index()
    
    # Save to file
    output_path = os.path.join(run_dir, "market_internals_rollup.csv")
    result_df.to_csv(output_path, index=False)
    print(f"[MarketInternals] Rollup saved to: {output_path}")
    print(result_df.to_markdown(index=False))


# ---------------------------------------------------------------------------------
# --- END: MARKET INTERNALS ROLLUP
# ---------------------------------------------------------------------------------


def main():
    # Parse arguments first
    import os
    import json
    ap = argparse.ArgumentParser(description="StatArb Engine v1.3 — pairs scanner")
    ap.add_argument("--portfolio_mode", type=str, choices=["on","off"], default="off",
                    help="If 'on', generate position plans and units watchlist CSVs (no broker integration).")
    # ... (rest of the arguments are unchanged) ...
    
    # --- New arguments for market internals rollup ---
    ap.add_argument("--market_internals_rollup", action="store_true",
                    help="Generate a market internals rollup based on SPY vs RSP sector weightings.")
    
    ap.add_argument("--max_concurrent", type=int, default=5, help="Max concurrent planned opens per run (position plans).")
    ap.add_argument("--per_ticker_cap", type=float, default=0.05, help="Per-ticker cap as fraction of equity (e.g., 0.05 = 5%).")
    ap.add_argument("--unit_targets", type=str, default=None,
                    help="Comma-separated mapping like SPY=100,IWM=100 for CSP accumulation targets.")
    ap.add_argument("--csp_dte", nargs="+", default=[15,45], help="CSP DTE range: e.g., --csp_dte 15 45")
    ap.add_argument("--csp_delta", type=float, default=0.30, help="Target delta for CSP.")
    ap.add_argument("--cc_dte", nargs="+", default=[15,30], help="CC DTE range: e.g., --cc_dte 15 30")
    ap.add_argument("--cc_delta", type=float, default=0.25, help="Target delta for CC.")
    ap.add_argument("--export_units", action="store_true", help="Write units_watchlist.csv for CSP/CC plans.")
    ap.add_argument("--sector_exposure_cap", type=float, default=None,
                    help="Cap per-sector suggested notional as fraction of account equity (e.g., 0.10 = 10%).")
    ap.add_argument("--target_vol", type=float, default=0.02,
                    help="Target daily volatility fraction for per-trade sizing (e.g., 0.02 = 2% of equity).")
    ap.add_argument("--risk_circuit_breaker", type=int, default=0,
                    help="If >0, max consecutive losers allowed before halting signals (requires pnl_history.csv)")
    ap.add_argument("--max_drawdown_limit", type=float, default=None,
                help="If set (e.g., 0.10 = 10%), trip circuit breaker when rolling equity drawdown exceeds this fraction.")
    ap.add_argument("--johansen_bonus", action="store_true",
                    help="Boost conviction if pair is part of cointegrated triple with SPY/QQQ")
    ap.add_argument("--johansen_filter", action="store_true",
                    help="Require pair to pass Johansen triple test (with SPY/QQQ) to be included")
    ap.add_argument("--beta_mode", type=str, choices=["ols","kalman"], default="ols",
                    help="Hedge-ratio estimator: ordinary least squares (ols) or recursive least squares (kalman)")
    ap.add_argument("--johansen_triples", action="store_true",
                    help="Compute Johansen cointegration stats for triples (anchor with SPY if present)")
    ap.add_argument("--johansen_trace_threshold", type=float, default=20.0,
                    help="TraceStat_r0 threshold for considering a triple cointegrated (used for johansen_bonus/filter)")
    ap.add_argument("--exposure_cap", type=float, default=0.25,
                    help="Cap total suggested notional as fraction of account equity (e.g., 0.25 = 25%). Requires account_equity.")
    ap.add_argument("--event_config", type=str, default=None,
                    help="Path to JSON event config to override universe/params (start/interval/windows/universe/z_enter/hl_max)")
    ap.add_argument("--macro_event", type=str, default=None,
                    help="Optional macro event tag (e.g., FOMC, CPI, Jobs) for run labeling")
    ap.add_argument("--universe", nargs='+', required=False, default=[],
                    help="Ticker universe (optional if using a preset)")
    ap.add_argument("--start", type=str, default="2021-01-01")
    ap.add_argument("--preset", type=str, choices=["YTD","1Y"], default=None,
                    help="Date presets: YTD = from Jan 1 of current year; 1Y = 1 year back from today")
    ap.add_argument("--preset_spy_scan", action="store_true",
                    help="Relaxed preset for SPY-long scans: lowers Z-enter, widens HL band, sets 1Y lookback, and forces spy_long_only")
    ap.add_argument("--preset_sector_scan", action="store_true",
                    help="Sector scan preset: SPY + key sectors (XLK, XLF, XLE, XLV, XLY, XLU, IWM) with relaxed thresholds")
    ap.add_argument("--preset_macro_scan", action="store_true",
                    help="Macro scan preset: broad cross-asset set including bonds, metals, dollar, banks, housing")
    ap.add_argument("--preset_fedday", action="store_true",
                    help="Fed-day preset: short intraday windows, rate-sensitive ETFs, macro universe")
    ap.add_argument("--preset_alpha_scan", action="store_true",
                    help="Lean daily alpha scan: SPY + QQQ + IWM with tightened exports")
    ap.add_argument("--preset_intraday_probe", action="store_true",
                    help="Run with intraday probe settings (5m interval, windows 30/60/90).")
    ap.add_argument("--interval", type=str, default="1d", help="yfinance interval (1d,1h,5m, etc.)")
    ap.add_argument("--windows", nargs='+', type=int, default=[30, 60, 90, 180, 252])
    ap.add_argument("--pvalue", type=float, default=0.05, help="cointegration p-value cutoff")
    ap.add_argument("--z_enter", type=float, default=2.0)
    ap.add_argument("--z_scale", type=float, default=2.5)
    ap.add_argument("--z_exit", type=float, default=0.5)
    ap.add_argument("--hl_min", type=int, default=3)
    ap.add_argument("--hl_max", type=int, default=40)
    ap.add_argument("--max_pairs", type=int, default=8, help="Top pairs to export in watchlist")
    ap.add_argument("--watchlist", type=str, default="./outputs/watchlist_pairs.csv")
    ap.add_argument("--diagnostics", type=str, default="./outputs/diagnostics.csv")
    ap.add_argument("--local_dir", type=str, default=None, help="Local directory with CSV files for price data")
    ap.add_argument("--delta_atm", type=float, default=0.5, help="Assumed ATM option delta for contract mapping")
    ap.add_argument("--delta_ditm", type=float, default=0.9, help="Assumed Deep ITM option delta for contract mapping")
    ap.add_argument("--correlations", type=str, default=None, help="Optional: path to write universe correlation matrix CSV")
    ap.add_argument("--source", type=str, default="yfinance",
                    choices=["yfinance","local"],
                    help="Price source")
    ap.add_argument("--cache", type=str, default="on",
                    choices=["on","off"], help="Use Parquet cache for yfinance")
    ap.add_argument("--refresh", action="store_true",
                    help="Force refresh cache from source")
    ap.add_argument("--liquidity_file", type=str, default=None,
                    help="CSV with columns: ticker,score (0-3)")
    ap.add_argument("--min_liquidity", type=int, default=None,
                    help="Min liquidity score to include (SPY/QQQ always kept)")
    ap.add_argument("--knn", type=int, default=0,
                    help="If >0, reduce pairs to k-nearest neighbors (plus anchors)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from last run's px_snapshot.csv instead of refetching")
    ap.add_argument("--note", type=str, default=None,
                    help="Optional free-form note to include in run_summary.md")
    # v1.6 additions
    ap.add_argument("--risk_unit", type=float, default=0.01, help="Risk per trade as fraction of account equity (e.g., 0.01 = 1%)")
    ap.add_argument("--account_equity", type=float, default=None, help="Account equity in dollars for sizing; if omitted, SuggestedNotional will be NaN")
    ap.add_argument("--rolling_window", type=int, default=60, help="Lookback window for rolling diagnostics per pair")
    ap.add_argument("--rolling_for_top", type=int, default=5, help="Compute rolling diagnostics for this many top pairs (0 = skip)")
    ap.add_argument("--pair", nargs=2, metavar=("LEFT", "RIGHT"),
                help="Run in pair mode for specific tickers")
    # New: plotting and export flags
    ap.add_argument("--plots_for_top", type=int, default=0, help="Generate charts for this many top pairs (0=skip)")
    ap.add_argument("--export_html", action="store_true", help="Export summary as HTML in addition to Markdown")
    ap.add_argument("--export_pdf", action="store_true", help="Export summary as PDF (requires reportlab)")
    ap.add_argument("--max_cache_age", type=int, default=None,
                    help="Maximum cache age in days (abort if older). Default=None (no limit)")
    ap.add_argument("--refresh_if_stale", action="store_true",
                    help="If set, automatically refresh cache when older than max_cache_age instead of aborting")
    ap.add_argument("--spy_long_only", action="store_true", help="Filter results to only long SPY opportunities")
    ap.add_argument("--vix_sensitive", action="store_true",
                    help="Scale z_enter threshold based on current VIX level (higher vol → higher threshold)")
    # --- New arguments for universe file and johansen_topn ---
    ap.add_argument("--universe_file", type=str, default=None,
                    help="CSV file with tickers (column: ticker). Merged with --universe")
    ap.add_argument("--johansen_topn", type=int, default=20,
                    help="Limit number of correlated names vs anchor for Johansen triples")
    args = ap.parse_args()
    global args_global
    pair_mode = args.pair is not None
    args_global = args

    # Setup run_dir early so it's available before use
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("./runs", run_ts)
    # Ensure output directory exists for all output artifacts
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    # Optional: load event config JSON
    if args.event_config and os.path.exists(args.event_config):
        try:
            with open(args.event_config, "r") as _f:
                cfg = _json.load(_f)
            if isinstance(cfg, dict):
                if "start" in cfg:
                    args.start = cfg["start"]
                elif "start_offset_days" in cfg:
                    offset = int(cfg.get("start_offset_days", 365))
                    args.start = (pd.Timestamp.today().normalize() - pd.DateOffset(days=offset)).strftime("%Y-%m-%d")
                if "interval" in cfg:
                    args.interval = cfg["interval"]
                if "windows" in cfg and isinstance(cfg["windows"], list) and cfg["windows"]:
                    args.windows = [int(w) for w in cfg["windows"]]
                if "z_enter" in cfg:
                    args.z_enter = float(cfg["z_enter"])
                if "hl_max" in cfg:
                    args.hl_max = int(cfg["hl_max"])
                if "universe" in cfg and isinstance(cfg["universe"], list):
                    args.universe = [str(t).upper() for t in cfg["universe"]]
                print(f"[StatArb] Loaded event config: {os.path.abspath(args.event_config)}")
                if "name" in cfg and not args.macro_event:
                    args.macro_event = str(cfg["name"])
        except Exception as e:
            print(f"[StatArb] Failed to load event config {args.event_config}: {e}")
    # Print preset banner line before any other output
    if args.preset_spy_scan:
        print("[StatArb] === Running with SPY Scan Preset ===")
    elif args.preset_sector_scan:
        print("[StatArb] === Running with Sector Scan Preset ===")
    elif args.preset_macro_scan:
        print("[StatArb] === Running with Macro Scan Preset ===")
    elif args.preset_fedday:
        print("[StatArb] === Running with Fed Day Preset ===")
    elif args.preset_alpha_scan:
        print("[StatArb] === Running with Alpha Scan Preset ===")
    elif args.preset:
        print(f"[StatArb] === Running with {args.preset.upper()} Preset ===")

    # Apply --preset logic to override --start if present
    if args.preset:
        today = pd.Timestamp.today().normalize()
        if args.preset.upper() == "YTD":
            start_date = f"{today.year}-01-01"
            args.start = start_date
            print(f"[StatArb] Using YTD preset start date: {args.start}")
        elif args.preset.upper() == "1Y":
            one_year_ago = today - pd.DateOffset(years=1)
            args.start = one_year_ago.strftime("%Y-%m-%d")
            print(f"[StatArb] Using 1Y preset start date: {args.start}")

    # Apply --preset_spy_scan and --preset_sector_scan logic
    if args.preset_spy_scan:
        today = pd.Timestamp.today().normalize()
        one_year_ago = today - pd.DateOffset(years=1)
        args.start = one_year_ago.strftime("%Y-%m-%d")
        args.z_enter = 1.5
        args.hl_max = max(args.hl_max, 60)
        args.spy_long_only = True
        if not args.universe:
            args.universe = ["SPY", "QQQ"]
        print(f"[StatArb] Using SPY scan preset: start={args.start}, "
              f"z_enter={args.z_enter}, hl_max={args.hl_max}, spy_long_only={args.spy_long_only}, universe={args.universe}")

    if args.preset_sector_scan:
        today = pd.Timestamp.today().normalize()
        one_year_ago = today - pd.DateOffset(years=1)
        args.start = one_year_ago.strftime("%Y-%m-%d")
        args.z_enter = 1.5
        args.hl_max = max(args.hl_max, 60)
        sector_universe = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLY", "XLU"]
        if not args.universe:
            args.universe = sector_universe
        else:
            args.universe = list({t.upper() for t in (args.universe + sector_universe)})
        print(f"[StatArb] Using Sector scan preset: start={args.start}, "
              f"universe={args.universe}, z_enter={args.z_enter}, hl_max={args.hl_max}")

    if args.preset_macro_scan:
        today = pd.Timestamp.today().normalize()
        one_year_ago = today - pd.DateOffset(years=1)
        args.start = one_year_ago.strftime("%Y-%m-%d")
        args.z_enter = 1.5
        args.hl_max = max(args.hl_max, 60)
        macro_universe = ["SPY","QQQ","DIA","IWM","XLK","XLF","XLE","XLV","XLY","XLU",
                          "TLT","IEF","SHY","GLD","SLV","UUP","VIXY","XHB","KRE","XLI","XLB"]
        if not args.universe:
            args.universe = macro_universe
        else:
            args.universe = list({t.upper() for t in (args.universe + macro_universe)})
        print(f"[StatArb] Using Macro scan preset: start={args.start}, "
              f"universe={args.universe}, z_enter={args.z_enter}, hl_max={args.hl_max}")

    if args.preset_alpha_scan:
        today = pd.Timestamp.today().normalize()
        one_year_ago = today - pd.DateOffset(years=1)
        args.start = one_year_ago.strftime("%Y-%m-%d")
        if not args.universe:
            args.universe = ["SPY", "QQQ", "IWM"]
        args.max_pairs = 5
        print(f"[StatArb] Using Alpha Scan preset: start={args.start}, universe={args.universe}, max_pairs={args.max_pairs}")

    # --- Fed Day Preset ---
    if args.preset_fedday:
        today = pd.Timestamp.today().normalize()
        one_week_ago = today - pd.DateOffset(weeks=1)
        args.start = one_week_ago.strftime("%Y-%m-%d")
        args.interval = "5m"
        args.windows = [15, 30, 60, 120]
        args.z_enter = 2.0
        args.hl_max = max(args.hl_max, 20)
        fed_universe = ["SPY","QQQ","IWM","DIA","XLK","XLF","XLE","XLV","XLY","XLP",
                        "TLT","IEF","SHY","KRE","XHB","GLD","UUP","VIXY"]
        if not args.universe:
            args.universe = fed_universe
        else:
            args.universe = list({t.upper() for t in (args.universe + fed_universe)})
        print(f"[StatArb] Using Fed-day preset: start={args.start}, interval={args.interval}, "
              f"windows={args.windows}, universe={args.universe}")

    # --- Intraday Probe Preset ---
    if args.preset_intraday_probe:
        args.interval = "5m"
        args.windows = [30, 60, 90]
        args.note = args.note or "Intraday probe preset (5m, windows=30/60/90)"
        if not args.universe:
            args.universe = ["SPY", "QQQ", "IWM"]
        print(f"[StatArb] Using Intraday Probe preset: interval={args.interval}, windows={args.windows}, universe={args.universe}")

    # --- VIX-sensitive z_enter scaling ---
    if args.vix_sensitive:
        try:
            vix_df = fetch_yf_single("^VIX", start=(pd.Timestamp.today() - pd.DateOffset(days=5)).strftime("%Y-%m-%d"), interval="1d")
            current_vix = float(vix_df.iloc[-1,0])
            scale_factor = max(1.0, current_vix / 20.0)
            old_z = args.z_enter
            args.z_enter = round(args.z_enter * scale_factor, 2)
            print(f"[StatArb] VIX-sensitive mode: VIX={current_vix:.2f} → z_enter scaled {old_z} → {args.z_enter}")
        except Exception as e:
            print(f"[StatArb] Failed to fetch VIX for sensitivity scaling: {e}")

    import glob
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("./runs", run_ts)
    px = None
    resumed = False
    if args.resume:
        # Find most recent run directory with px_snapshot.csv
        runs = sorted(
            [d for d in glob.glob("./runs/*") if os.path.isdir(d)],
            reverse=True
        )
        found = False
        for d in runs:
            px_snap = os.path.join(d, "px_snapshot.csv")
            if os.path.exists(px_snap):
                run_dir = d
                try:
                    px = pd.read_csv(px_snap, index_col=0, parse_dates=True)
                    resumed = True
                    found = True
                    print(f"[StatArb] Resuming from previous run: {run_dir}")
                    break
                except Exception as e:
                    print(f"[StatArb] Failed to load px_snapshot.csv from {d}: {e}")
        if not found:
            print("[StatArb] --resume set, but no previous run with px_snapshot.csv found. Creating new run.")
            os.makedirs(run_dir, exist_ok=True)
    else:
        os.makedirs(run_dir, exist_ok=True)

    # Override output paths to point into run_dir (output directory)
    args.output = run_dir  # Add output attribute for clarity
    args.watchlist = os.path.join(args.output, "watchlist.csv")
    args.diagnostics = os.path.join(args.output, "diagnostics.csv")
    if args.correlations is not None:
        args.correlations = os.path.join(args.output, "correlations.csv")

    # --- Load universe from file if provided ---
    if args.universe_file and os.path.exists(args.universe_file):
        try:
            u_df = pd.read_csv(args.universe_file)
            file_tickers = [str(t).upper() for t in u_df["ticker"].dropna().tolist()]
            args.universe = list(set(args.universe) | set(file_tickers))
            print(f"[StatArb] Loaded {len(file_tickers)} tickers from {args.universe_file}")
        except Exception as e:
            msg = f"[StatArb] Failed to load universe_file {args.universe_file}: {e}"
            print(msg)
            log_error(run_dir, msg)

    # Liquidity pruning and KNN setup
    score_map = read_liquidity_scores(args.liquidity_file)
    original_universe = args.universe
    if args.min_liquidity is not None:
        args.universe = apply_liquidity_filter(args.universe, args.min_liquidity, score_map)
        print(f"[StatArb] Liquidity filter kept {len(args.universe)}/{len(original_universe)} tickers: {args.universe}")

    # Ensure universe is not empty
    if not getattr(args, "universe", None):
        print("[Warn] No universe provided. Defaulting to ['SPY'].")
        args.universe = ["SPY"]

    # --- Universe build step (after liquidity filter, before loading prices) ---
    args.universe = build_universe(args)

    # Basic checks
    if yf is None or coint is None or sm is None:
        raise SystemExit("Missing dependencies. Please install with: pip install yfinance statsmodels pandas numpy")

    if not resumed:
        print("[StatArb] Loading prices...")
        if args.source == "local":
            try:
                px = load_prices_local(args.universe, args.local_dir or "./data")
            except Exception as e:
                print(f"[Fatal] Failed to load local prices: {e}")
                log_error(run_dir, f"Fatal load_prices_local: {e}")
                return
        else:
            try:
                px = load_prices(args.universe, args.start, args.interval,
                                 refresh=args.refresh)
            except Exception as e:
                print(f"[Fatal] Failed to load prices: {e}")
                log_error(run_dir, f"Fatal load_prices: {e}")
                return
        px = px.dropna(how="any")
        print(f"[StatArb] Universe: {list(px.columns)}; Rows: {len(px)}")
        # Save price DataFrame snapshot
        px.to_csv(os.path.join(run_dir, "px_snapshot.csv"))
    else:
        # Already loaded px from px_snapshot.csv
        print(f"[StatArb] Universe: {list(px.columns)}; Rows: {len(px)}")

    # Optional Johansen diagnostics for triples
    if args.johansen_triples:
        try:
            anchors = [a for a in ["SPY","QQQ"] if a in px.columns]
            anchor = anchors[0] if anchors else None
            triples_path = os.path.join(run_dir, "johansen_triples.csv")
            jrows = []
            cols = list(px.columns)
            if anchor:
                others = [c for c in cols if c != anchor]
                # sample top N by correlation with anchor to limit runtime
                topn = getattr(args, "johansen_topn", 20)
                corr = px.pct_change().corr()[anchor].drop(index=anchor).abs().sort_values(ascending=False).head(topn).index.tolist()
                candidates = corr
                for i in range(len(candidates)):
                    for j in range(i+1, len(candidates)):
                        a, b = candidates[i], candidates[j]
                        df3 = px[[anchor, a, b]].dropna().tail(max(args.windows))
                        if len(df3) >= 60:
                            # Use johansen_passes for dynamic critical value
                            passed, stat, crit = johansen_passes(df3, conf=0.95)
                            eig, tr = johansen_trace_stat(df3)
                            if eig is not None and tr is not None:
                                jrows.append({"Triple": f"{anchor}-{a}-{b}", "TraceStat_r0": float(tr[0]), "TraceStat_r1": float(tr[1])})
                            # Use dynamic pass/fail for johansen_pairs
                            if passed:
                                global johansen_pairs
                                johansen_pairs.update([(anchor,a),(a,anchor),(anchor,b),(b,anchor),(a,b),(b,a)])
            if jrows:
                pd.DataFrame(jrows).to_csv(triples_path, index=False)
                print(f"[StatArb] Johansen triple diagnostics written: {os.path.abspath(triples_path)}")
        except Exception as e:
            msg = f"[StatArb] Johansen diagnostics failed: {e}"
            print(msg)
            log_error(run_dir, msg)

    # Enforce cache age limits if requested
    if args.max_cache_age is not None:
        today = pd.Timestamp.today().normalize()
        cache_info = {}
        for t in px.columns:
            if px[t].dropna().empty:
                continue
            last_dt = px[t].dropna().index.max().normalize()
            age_days = (today - last_dt).days
            cache_info[t] = {"last_date": str(last_dt.date()), "age_days": int(age_days)}
            if age_days > args.max_cache_age:
                if args.refresh_if_stale:
                    print(f"[StatArb] Cache for {t} is {age_days} days old (last update: {last_dt.date()}). Auto-refreshing...")
                    # force refresh and reload
                    px = load_prices(args.universe, args.start, args.interval,
                                     source="yfinance", use_cache=True, refresh=True)
                    # Save snapshot and update run_dir snapshot
                    px.to_csv(os.path.join(run_dir, "px_snapshot.csv"))
                    break
                else:
                    raise SystemExit(
                        f"[StatArb] Cache for {t} is {age_days} days old (last update: {last_dt.date()}). "
                        f"--max_cache_age={args.max_cache_age} specified → please run with --refresh."
                    )
            elif age_days > 1:
                print(f"[WARN] Cache for {t} is {age_days} days old (last update: {last_dt.date()}).")
        # stash cache_info for metadata
    else:
        cache_info = {}

    # Optional: Correlation pre-check step
    if args.correlations is not None:
        # ensure_dir(args.correlations)  # Not needed: run_dir already exists
        corr = px.pct_change().corr()
        corr.to_csv(args.correlations)
        print(f"[StatArb] Correlation matrix written: {os.path.abspath(args.correlations)}")

    # KNN pairs setup
    anchors = [t for t in ["SPY", "QQQ"] if t in px.columns]
    knn_pairs = build_knn_pairs(px, anchors, args.knn) if args.knn and args.knn > 0 else set()

    results: List[PairResult] = []
    pairs_iter = itertools.combinations(list(px.columns), 2)
    if knn_pairs:
        pairs_iter = knn_pairs
    for pair in pairs_iter:
        y, x = pair if isinstance(pair, tuple) else pair
        pr = evaluate_pair(px, y, x, args.windows, args.pvalue,
                           args.hl_min, args.hl_max,
                           args.z_enter, args.z_scale, args.z_exit,
                           args.delta_atm, args.delta_ditm,
                           args.account_equity, args.risk_unit)
        if pr is not None:
            results.append(pr)

    if not results:
        msg = "[StatArb] No qualifying pairs found with current settings."
        print(msg)
        log_error(run_dir, msg)
        return

    if pair_mode:
        r = results[0]
        print(f"[PairMode] {r.left}-{r.right} | W={r.best.window} | p={r.best.pvalue:.4f} | beta={r.best.beta:.3f} | HL={r.best.half_life:.2f} | Z={r.best.z_curr:.2f} | JohansenPass={r.johansen_pass}")
        try:
            export_pair_timeseries(px, r.left, r.right, r.best.window, r.best.beta, run_dir)
            print(f"[PairMode] Spread/Z series saved → {os.path.join(run_dir, f'{r.left}-{r.right}_spread_timeseries.csv')}")
        except Exception as e:
            msg = f"[PairMode] Failed to export spread series: {e}"
            print(msg)
            log_error(run_dir, msg)
        try:
            summary_txt = os.path.join(args.output, f"{r.left}-{r.right}_summary.txt")
            with open(summary_txt, "w") as f:
                f.write(f"Pair: {r.left}-{r.right}\n")
                f.write(f"BestWindow: {r.best.window}\n")
                f.write(f"PValue: {r.best.pvalue:.6f}\n")
                f.write(f"Beta: {r.best.beta:.6f}\n")
                f.write(f"HalfLife: {r.best.half_life:.4f}\n")
                f.write(f"Z: {r.best.z_curr:.4f}\n")
                f.write(f"JohansenTrace: {r.johansen_trace}, Crit: {r.johansen_crit}, Pass: {r.johansen_pass}\n")
                f.write(f"Signal: {r.signal} | Conviction: {r.conviction_score:.2f} ({r.conviction_band})\n")
                f.write(f"Notes: {r.notes}\n")
            print(f"[PairMode] Summary written → {summary_txt}")
        except Exception as e:
            msg = f"[PairMode] Failed to write summary: {e}"
            print(msg)
            log_error(run_dir, msg)

    # Rank by conviction_score desc, then p-value asc
    results.sort(key=lambda r: (-r.conviction_score, r.best.pvalue))

    # --- Risk Circuit Breaker ---
    tripped = False
    consec_losses = 0

    # (A) Percent drawdown breaker (uses historical equity curve)
    if args.max_drawdown_limit and args.account_equity:
        eq_path = os.path.join("./runs", "equity_curve.csv")
        if os.path.exists(eq_path):
            try:
                eq_df = pd.read_csv(eq_path)
                if not eq_df.empty and "EquityEnd" in eq_df.columns:
                    peak = eq_df["EquityEnd"].cummax()
                    dd = (eq_df["EquityEnd"] - peak) / peak
                    curr_dd = float(dd.iloc[-1])
                    if curr_dd <= -abs(args.max_drawdown_limit):
                        msg = f"[Risk] Circuit breaker TRIPPED: drawdown {curr_dd:.2%} ≤ -{args.max_drawdown_limit:.2%}."
                        print(msg)
                        log_error(run_dir, msg)
                        tripped = True
            except Exception as e:
                msg = f"[Risk] Failed drawdown check: {e}"
                print(msg)
                log_error(run_dir, msg)

    # (B) Consecutive losses breaker (existing behavior)
    if not tripped and args.risk_circuit_breaker and args.account_equity:
        pnl_hist_path = os.path.join("./runs", "pnl_history.csv")
        if os.path.exists(pnl_hist_path):
            try:
                dfp = pd.read_csv(pnl_hist_path)
                if not dfp.empty and "PnL" in dfp.columns:
                    for val in reversed(dfp["PnL"].tolist()):
                        if val < 0:
                            consec_losses += 1
                            if consec_losses >= args.risk_circuit_breaker:
                                msg = f"[Risk] Circuit breaker TRIPPED: {consec_losses} consecutive losses."
                                print(msg)
                                log_error(run_dir, msg)
                                tripped = True
                                break
                        else:
                            break
            except Exception as e:
                msg = f"[Risk] Failed to read pnl_history.csv: {e}"
                print(msg)
                log_error(run_dir, msg)

    if tripped:
        # Write marker file and halt before exporting results
        with open(os.path.join(run_dir, "circuit_breaker.txt"), "w") as f:
            f.write(f"Tripped. ConsecutiveLosses={consec_losses}, MaxDDLimit={args.max_drawdown_limit}\n")
        return

    # Filter for SPY long-only opportunities if requested
    if getattr(args, "spy_long_only", False):
        orig_count = len(results)
        results = [r for r in results if r.signal == "LONG_SPY_OPPORTUNITY"]
        print(f"[StatArb] Filtered to SPY long opportunities: {len(results)} pairs")
        if not results:
            print("[StatArb] No SPY long opportunities found with current settings.")
            return

    # Enforce per-sector cap if requested
    sector_usage = {}
    for r in results:
        sec_left = SECTOR_MAP.get(r.left, "Other")
        sec_right = SECTOR_MAP.get(r.right, "Other")
        sector = sec_left if sec_left != "Other" else sec_right
        if args.sector_exposure_cap and args.account_equity and pd.notna(r.suggested_notional):
            cap_dollars = args.account_equity * args.sector_exposure_cap
            used = sector_usage.get(sector, 0.0)
            avail = max(cap_dollars - used, 0.0)
            if r.suggested_notional > avail:
                r.suggested_notional = avail
            sector_usage[sector] = sector_usage.get(sector, 0.0) + (r.suggested_notional or 0.0)

    # --- Portfolio Planner (v3 scaffolding) ---
    if args.portfolio_mode == "on":
        plans = plan_positions(results, args.account_equity, args.per_ticker_cap, args.max_concurrent)
        if plans:
            save_position_plans(run_dir, plans)
            print(f"[Portfolio] Planned {len(plans)} positions → ./runs/positions.csv")
        else:
            print("[Portfolio] No position plans generated (check sizing/caps/account_equity).")

        # Units planner (CSP/CC) — optional export
        targets = {}
        if args.unit_targets:
            try:
                for kv in str(args.unit_targets).split(","):
                    if "=" in kv:
                        k, v = kv.split("=")
                        targets[k.strip().upper()] = int(v)
            except Exception as e:
                msg = f"[Units] Failed to parse --unit_targets: {e}"
                print(msg)
                log_error(run_dir, msg)
                targets = {}

        csp_range = parse_range_to_ints(args.csp_dte)
        cc_range = parse_range_to_ints(args.cc_dte)
        held = load_held_shares()
        csp_plans = plan_csp(args.universe, targets, args.account_equity, csp_range, args.csp_delta) if targets else []
        cc_plans = plan_cc(held, cc_range, args.cc_delta) if held else []

        if args.export_units and (csp_plans or cc_plans):
            save_units_watchlist(run_dir, csp_plans, cc_plans)
            print(f"[Units] Watchlist updated ({len(csp_plans)} CSP, {len(cc_plans)} CC) → ./runs/units_watchlist.csv")

    # Export watchlist (top N)
    wl_rows = []
    for r in results[:args.max_pairs]:
        wl_rows.append({
            "Pair": f"{r.left}-{r.right}",
            "Left": r.left,
            "Right": r.right,
            "BestWindow": r.best.window,
            "PValue": f"{r.best.pvalue:.4f}",
            "Beta": f"{r.best.beta:.3f}",
            "HalfLife": f"{r.best.half_life:.1f}",
            "Z": f"{r.best.z_curr:.2f}",
            "Conviction": f"{r.conviction_score:.2f}",
            "ConvictionBand": r.conviction_band,
            "Signal": r.signal,
            "Action": r.action,
            "SR_Signal": r.sr_signal,
            "Support_1s": f"{r.support_1s:.2f}",
            "Resistance_1s": f"{r.resistance_1s:.2f}",
            "Support_2s": f"{r.support_2s:.2f}",
            "Resistance_2s": f"{r.resistance_2s:.2f}",
            "GrowthPhase": r.growth_phase,
            "PhaseGuidance": r.phase_guidance,
            "Notes": r.notes,
        })

    wl_df = pd.DataFrame(wl_rows)
    ensure_dir(args.watchlist)
    wl_df.to_csv(args.watchlist, index=False)
    print(f"[StatArb] Watchlist saved: {os.path.abspath(args.watchlist)}")

    # --- Integration Point for Market Internals Rollup ---
    if args.market_internals_rollup:
        if not wl_df.empty:
            market_internals_rollup(wl_df, run_dir)
        else:
            print("[MarketInternals] Watchlist is empty, skipping rollup.")

    # (The rest of the script continues...)

if __name__ == "__main__":
    main()