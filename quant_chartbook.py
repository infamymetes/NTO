# quant_chartbook.py
# Generates chartbook images for top-N pairs from watchlist using latest run snapshot.
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(__file__))
from analysis import get_data
import argparse
import yaml

# ---------------- Config ----------------
RUNS_DIR = "./runs"
TOP_N = 10                    # how many pairs to process
HORIZONS = [1, 3, 5, 10]      # forward bars (days or intraday bars)
Z_BINS = np.linspace(-3, 3, 25)
RATIO_LINES = [np.sqrt(2), 1.618, 2.0]

# -------------- Utilities ---------------
def newest_run_dir(base="./runs"):
    cand = [d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)]
    if not cand:
        raise FileNotFoundError("No run directories found under ./runs")
    return sorted(cand, key=lambda p: os.path.getmtime(p))[-1]

def load_snapshot(run_dir):
    # Expect a wide CSV with Date index and columns as tickers
    snap = os.path.join(run_dir, "px_snapshot.csv")
    if not os.path.exists(snap):
        raise FileNotFoundError(f"px_snapshot.csv not found in {run_dir}")
    df = pd.read_csv(snap, index_col=0, parse_dates=True)
    # Clean
    df = df.sort_index().dropna(how="all")
    return df

def load_diagnostics(run_dir):
    path = os.path.join(run_dir, "diagnostics.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None

def zscore(series: pd.Series) -> pd.Series:
    mu = series.mean()
    sd = series.std(ddof=0)
    if sd is None or sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros_like(series), index=series.index)
    return (series - mu) / sd

def ols_beta(y: pd.Series, x: pd.Series) -> float:
    # Minimal OLS beta (no statsmodels dependency)
    # beta = Cov(y,x) / Var(x)
    yv = y.dropna()
    xv = x.dropna()
    idx = yv.index.intersection(xv.index)
    yv = y.loc[idx].astype(float)
    xv = x.loc[idx].astype(float)
    vx = np.var(xv, ddof=0)
    if vx == 0 or np.isnan(vx):
        return 1.0
    cov = np.mean((xv - xv.mean()) * (yv - yv.mean()))
    b = cov / vx
    if np.isnan(b) or np.isinf(b):
        return 1.0
    return float(b)

def pick_beta(left, right, px, diag):
    # Prefer diagnostics Beta for this left-right pair if available
    if diag is not None and {"Left","Right","Beta"}.issubset(diag.columns):
        row = diag[(diag["Left"]==left) & (diag["Right"]==right)]
        if not row.empty and pd.notna(row.iloc[0]["Beta"]):
            return float(row.iloc[0]["Beta"])
    # Fallback to OLS on overlapping window
    if left in px.columns and right in px.columns:
        return ols_beta(px[left], px[right])
    return 1.0

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def compute_expectancy(spread: pd.Series, z: pd.Series, horizons, bins) -> pd.DataFrame:
    out = []
    for h in horizons:
        fwd = spread.shift(-h) - spread
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (z >= lo) & (z < hi)
            if mask.sum() > 0:
                exp_ret = fwd[mask].mean()
            else:
                exp_ret = np.nan
            out.append({"horizon": h, "bin_mid": (lo+hi)/2, "expectancy": exp_ret})
    return pd.DataFrame(out)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate chartbook images for top-N pairs")
    parser.add_argument("--symbols", nargs="+", required=True, help="Symbols list")
    parser.add_argument("--freq", default="daily", help="Frequency: daily/hourly/30m")
    parser.add_argument("--start", default="2024-01-01", help="Start date")
    parser.add_argument("--end", default=None, help="End date")
    parser.add_argument("--universe", type=str, default=None, help="Predefined universe from config/symbols.yaml (e.g. spy20, nq20)")
    return parser.parse_args()

def load_universe(name: str):
    with open("config/symbols.yaml", "r") as f:
        data = yaml.safe_load(f)
    if name not in data:
        raise ValueError(f"Universe '{name}' not found in config/symbols.yaml")
    symbols = data[name]
    if not isinstance(symbols, list):
        raise ValueError(f"Universe '{name}' should be a list of symbols")
    return symbols

# -------------- Main Workflow --------------
def main():
    args = parse_args()
    # Find latest run dir & load data
    run_dir = newest_run_dir(RUNS_DIR)
    if args.universe:
        symbols = load_universe(args.universe)
    else:
        symbols = args.symbols
    px = get_data(symbols=symbols, freq=args.freq, start=args.start, end=args.end)
    diag = load_diagnostics(run_dir)
    plots_dir = os.path.join(run_dir, "plots")
    ensure_dir(plots_dir)

    # Load watchlist & select top N pairs
    wl_path = os.path.join(run_dir, "watchlist.csv")
    if not os.path.exists(wl_path):
        raise FileNotFoundError(f"watchlist.csv not found in {run_dir}")
    wl = pd.read_csv(wl_path)
    if not {"Left","Right"}.issubset(wl.columns):
        raise ValueError("watchlist.csv must include columns: Left, Right")
    top_pairs = wl.head(TOP_N)[["Left","Right"]].astype(str).values.tolist()

    for left, right in top_pairs:
        if left not in px.columns or right not in px.columns:
            print(f"[Skip] Missing data for {left}-{right} in px snapshot")
            continue

        beta = pick_beta(left, right, px, diag)
        sub = px[[left, right]].dropna()
        if sub.empty:
            print(f"[Skip] No overlapping data for {left}-{right}")
            continue

        spread = sub[left] - beta * sub[right]
        z = zscore(spread)

        # ---- 1) Prices ----
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(sub.index, sub[left], label=left, alpha=0.85)
        ax.plot(sub.index, sub[right], label=right, alpha=0.85)
        ax.set_title(f"{left}-{right} Prices")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{left}-{right}_prices.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---- 2) Spread ----
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(spread.index, spread.values, label="Spread")
        ax.axhline(spread.mean(), linestyle="--", linewidth=1)
        ax.set_title(f"{left}-{right} Spread (β={beta:.3f})")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{left}-{right}_spread.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---- 3) Z-score ----
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(z.index, z.values, label="Z")
        ax.axhline(0, linestyle=":", linewidth=1)
        ax.axhline(2.0, linestyle="--", linewidth=1, label="+2σ")
        ax.axhline(-2.0, linestyle="--", linewidth=1, label="-2σ")
        ax.set_title(f"{left}-{right} Z-score")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{left}-{right}_zscore.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---- 4) Z-score Distribution ----
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(z.dropna().values, bins=30, density=True, alpha=0.6)
        ax.axvline(0, linestyle=":")
        for r in RATIO_LINES:
            ax.axvline(r, linestyle="--", alpha=0.7)
            ax.axvline(-r, linestyle="--", alpha=0.7)
        ax.axvline(2.0, linestyle="--")
        ax.axvline(-2.0, linestyle="--")
        ax.set_title(f"{left}-{right} Z-Score Distribution")
        ax.set_xlabel("Z-score")
        ax.set_ylabel("Density")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{left}-{right}_zdist.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---- 5) Expectancy Heatmap ----
        exp = compute_expectancy(spread, z, HORIZONS, Z_BINS)
        heat = exp.pivot(index="horizon", columns="bin_mid", values="expectancy")

        fig, ax = plt.subplots(figsize=(12,4.5))
        c = ax.pcolormesh(heat.columns.values, heat.index.values, heat.values, cmap="coolwarm", shading="auto")
        fig.colorbar(c, ax=ax, label="Avg forward spread change")
        ax.set_title(f"{left}-{right} Expectancy Heatmap (Z-bin × horizon)")
        ax.set_xlabel("Z-score bin (midpoints)")
        ax.set_ylabel("Forward horizon (bars)")
        for r in RATIO_LINES + [2.0]:
            ax.axvline(+r, linestyle="--", alpha=0.7)
            ax.axvline(-r, linestyle="--", alpha=0.7)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{left}-{right}_expectancy.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"[Chartbook] Saved {left}-{right} plots to {plots_dir}")

    print(f"[Done] Chartbook complete → {plots_dir}")

if __name__ == "__main__":
    main()