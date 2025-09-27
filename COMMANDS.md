# Command Hit List for Validation & Studies

This document provides a concise set of shell and Python commands for validating and running key data pipeline and research tools. Use these as a checklist for new deployments, troubleshooting, or regular data health checks.

---

## Usage Notes

- **Replace** variable placeholders like `<date>`, `<symbol>`, or `<params>` with actual values.
- **Run** commands from the root project directory unless otherwise specified.
- **Checkmarks** (✅) indicate successful validation points.

---

## 1. Collector

**Run Data Collection for a Symbol**
```bash
python collector.py --symbols <symbol> --start <YYYY-MM-DD> --end <YYYY-MM-DD>
```
**Arguments:**
- `--symbolss`: List of symbols to collect data for
- `--universe`: Predefined universe of symbols
- `--freq`: Data frequency (e.g., daily, intraday)
- `--start`: Start date for data collection (YYYY-MM-DD)
- `--end`: End date for data collection (YYYY-MM-DD)
- `--backfill`: Flag to backfill missing data
- `--schedule`: Schedule the collection job
- `--source`: Data source to use

**Validation:**
- [ ] Collector script runs without errors ✅
- [ ] Output file(s) generated in `data/` ✅
- [ ] Log shows correct date range and symbol ✅
- [ ] Ledger entry created with rows observed/expected ✅

---

## 2. Freshness Check

**Check Recent Data Availability**
```bash
python freshness_check.py --symbols <symbol>
```
**Arguments:**
- `--symbolss`: List of symbols to check
- `--universe`: Predefined universe of symbols
- `--all`: Check all available data
- `--freq`: Data frequency to check
- `--start`: Start date for freshness check (YYYY-MM-DD)
- `--end`: End date for freshness check (YYYY-MM-DD)

**Validation:**
- [ ] Script outputs most recent data timestamp ✅
- [ ] Timestamp is within expected freshness window ✅
- [ ] No missing data for last N days ✅
- [ ] CSV summary written (e.g., freshness_summary.csv) ✅

---

## 3. Chartbook

**Generate Visualizations**
```bash
python quant_chartbook.py --symbols <symbol> --date <YYYY-MM-DD>
```
**Arguments:**
- `--symbolss`: List of symbols to generate charts for
- `--universe`: Predefined universe of symbols
- `--freq`: Frequency of data to plot
- `--start`: Start date for charts (YYYY-MM-DD)
- `--end`: End date for charts (YYYY-MM-DD)

**Validation:**
- [ ] Chart images saved to `charts/` ✅
- [ ] Plots show expected trends/values ✅
- [ ] Script completes with no errors ✅

---

## 4. StatArb Engine

**Run a Backtest**
```bash
python StatArbEngine_v1.py --config configs/<config>.yaml
```
**Arguments:**
- `--symbolss`: List of symbols to include in backtest
- `--freq`: Data frequency for backtest
- `--start`: Start date for backtest (YYYY-MM-DD)
- `--end`: End date for backtest (YYYY-MM-DD)
- `--max_pairs`: Maximum number of pairs to consider
- `--diagnostics`: Enable diagnostic output
- `--freshness_report`: Generate data freshness report
- `--account_equity`: Starting account equity for simulation

**Validation:**
- [ ] Backtest completes without exceptions ✅
- [ ] Results saved in `results/` ✅
- [ ] Summary stats (e.g., Sharpe, PnL) printed ✅
- [ ] Diagnostics (diagnostics.csv) and watchlist (watchlist.csv) written ✅

---

## Quick Reference Table

| Tool            | Command Example                                             | Key Validation Points                  |
|-----------------|------------------------------------------------------------|----------------------------------------|
| Collector       | `python collector.py --symbols AAPL --start 2024-05-01 --end 2024-06-01` | Output file, correct symbol/dates      |
| Freshness Check | `python freshness_check.py --symbols AAPL`                  | Recent timestamp, no missing data      |
| Chartbook       | `python quant_chartbook.py --symbols AAPL --date 2024-06-01`| Chart images, trends, no errors        |
| StatArb Engine  | `python StatArbEngine_v1.py --config configs/base.yaml`    | Results file, summary stats, no errors |

---

For further details, see the individual tool documentation or contact the data engineering team.

---

## Smoke Test Suite

1. **Collector Smoke Test**
```bash
python collector.py --symbols SPY --freq 1d --start 2024-05-01 --end 2024-06-01
```
*Expected Validation:* Runs without errors, output files in `data/`, ledger entry created with rows observed/expected ✅

2. **Freshness Check Smoke Test**
```bash
python freshness_check.py --symbols SPY --freq 1d
```
*Expected Validation:* Outputs recent timestamp, no missing data, CSV summary written ✅

3. **Chartbook Smoke Test**
```bash
python quant_chartbook.py --symbols SPY --date 2024-06-01
```
*Expected Validation:* Chart images saved to `charts/`, plots show expected trends, no errors ✅

4. **StatArbEngine Smoke Test (Basic Run)**
```bash
python StatArbEngine_v1.py --config configs/spy_qqq.yaml
```
*Expected Validation:* Backtest completes, results saved, summary stats printed ✅

5. **StatArbEngine Smoke Test (With Freshness Report)**
```bash
python StatArbEngine_v1.py --config configs/spy_qqq.yaml --freshness_report
```
*Expected Validation:* Backtest completes, diagnostics and watchlist CSVs written, freshness report generated ✅