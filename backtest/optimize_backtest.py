#!/usr/bin/env python3
"""
Parameter optimization loop for the lightweight backtesting engine.

Features:
- Grid search over user-provided ranges (comma separated lists) for key parameters.
- Optional random sampling (--sample N) when the grid size is very large.
- Aggregates metrics (win_rate_pct, net_R, profit_factor, max_drawdown_R, return_pct, expectancy_R).
- Saves full results to CSV + JSON summary of top N.
- Supports multiple input CSV files (minute OHLC). Results aggregate (weight by trades).

Usage example:
python backtest/optimize_backtest.py \
  --csv backtest/EURUSD1.csv backtest/eurusd_prices_multiip4.csv \
  --thresholds 5,6,7 --windows 80,100 --lookaheads 15,20 \
  --rrs 1.0,1.2,1.5 --minlegs 3,4 --risk-pcts 0.005,0.01 \
  --max-configs 500 --top 25
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Dict, Tuple
import time

import pandas as pd

from engine import BacktestConfig, BacktestEngine, load_ohlc_csv


def _parse_list(s: str, cast=float):
    return [cast(x.strip()) for x in s.split(',') if x.strip()]


def build_param_grid(args) -> List[Dict]:
    thresholds = _parse_list(args.thresholds, int)
    windows = _parse_list(args.windows, int)
    lookaheads = _parse_list(args.lookaheads, int)
    rrs = _parse_list(args.rrs, float)
    minlegs = _parse_list(args.minlegs, int)
    risk_pcts = _parse_list(args.risk_pcts, float)
    fib_entry_mins = _parse_list(args.fib_entry_mins, float) if args.use_external else [None]
    fib_entry_maxs = _parse_list(args.fib_entry_maxs, float) if args.use_external else [None]

    grid = []
    for th, win, la, rr, ml, rp, fib_min, fib_max in itertools.product(
            thresholds, windows, lookaheads, rrs, minlegs, risk_pcts, fib_entry_mins, fib_entry_maxs):
        if args.use_external:
            # validate fib band
            if fib_min is None or fib_max is None or fib_min >= fib_max:
                continue
        grid.append({
            'threshold_points': th,
            'window_size': win,
            'lookahead': la,
            'rr': rr,
            'min_leg_distance_points': ml,
            'risk_pct': rp,
            'fib_entry_min': fib_min,
            'fib_entry_max': fib_max,
        })
    return grid


def score_rule(summary: Dict) -> float:
    """Composite score to rank parameter sets.
    Weighted blend of: net_R, win_rate_pct, profit_factor, -max_drawdown_R.
    """
    return (
        summary.get('net_R', 0) * 1.5 +
        summary.get('win_rate_pct', 0) * 0.05 +
        summary.get('profit_factor', 0) * 0.5 +
        (-summary.get('max_drawdown_R', 0) * 0.3) +
        summary.get('expectancy_R', 0) * 2.0
    )


def aggregate_results(run_summaries: List[Dict]) -> Dict:
    # Weighted by trades
    if not run_summaries:
        return {}
    total_trades = sum(s['total_trades'] for s in run_summaries if s['total_trades'])
    agg = {}
    if total_trades == 0:
        return {k: None for k in run_summaries[0].keys()}
    # Simple weighted average for numeric fields
    for key in run_summaries[0].keys():
        vals = []
        for s in run_summaries:
            v = s.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                weight = s['total_trades'] / total_trades if s['total_trades'] else 0
                vals.append((v, weight))
        if vals:
            agg[key] = sum(v * w for v, w in vals)
    # For discrete counts just sum
    for k in ('total_trades','wins','losses','timeouts'):
        agg[k] = sum(s.get(k,0) for s in run_summaries)
    return agg


def infer_timeframe_minutes(df):
    if 'timestamp' not in df.columns:
        return None
    ts = df['timestamp'].sort_values().to_numpy()
    if len(ts) < 3:
        return None
    diffs = (ts[1:] - ts[:-1]) / 1e9  # به ثانیه
    # حذف پرش‌های غیرعادی (گپ بزرگ)
    filtered = [d for d in diffs if d > 0 and d < 3600*6]
    if not filtered:
        return None
    sec_mode = min(filtered, key=lambda x: abs(x - (sum(filtered)/len(filtered))))
    mins = round(sec_mode / 60)
    return max(1, mins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', nargs='+', required=True, help='One or more OHLC CSV paths')
    ap.add_argument('--symbol', default='EURUSD')
    ap.add_argument('--thresholds', default='6', help='Comma list')
    ap.add_argument('--windows', default='100')
    ap.add_argument('--lookaheads', default='20')
    ap.add_argument('--rrs', default='1.2')
    ap.add_argument('--minlegs', default='4')
    ap.add_argument('--risk-pcts', default='0.01')
    ap.add_argument('--use-external', action='store_true', help='Use real strategy logic (get_legs, swings, fibo)')
    ap.add_argument('--fib-entry-mins', default='0.705', help='Comma list (only if --use-external)')
    ap.add_argument('--fib-entry-maxs', default='0.9', help='Comma list (only if --use-external)')
    ap.add_argument('--initial-balance', type=float, default=10000.0)
    ap.add_argument('--price-scale', type=int, default=100000)
    ap.add_argument('--sample', type=int, default=0, help='Random sample size (0 = use full grid)')
    ap.add_argument('--max-configs', type=int, default=0, help='Hard cap on total configs (0 = unlimited)')
    ap.add_argument('--top', type=int, default=20, help='Top N to store in summary JSON')
    ap.add_argument('--outdir', default='backtest/opt_results')
    ap.add_argument('--progress-every', type=int, default=1, help='Update progress display every N configs')
    ap.add_argument('--no-dynamic-progress', action='store_true', help='Disable in-place dynamic progress line')
    ap.add_argument('--quiet', action='store_true', help='Suppress progress lines; only print final outputs')
    args = ap.parse_args()

    # Load data sets
    datasets = []
    for p in args.csv:
        try:
            df = load_ohlc_csv(p)
            datasets.append((p, df))
        except Exception as e:
            print(f"⚠️ Skipping {p}: {e}")
    if not datasets:
        print('No valid datasets loaded. Abort.')
        return

    if not args.quiet:
        print("Loaded datasets:")
        for p, df in datasets:
            tfm = infer_timeframe_minutes(df)
            print(f"- {p}: rows={len(df)} timeframe≈{tfm}m")

    grid = build_param_grid(args)
    total_grid = len(grid)
    if args.sample > 0 and args.sample < total_grid:
        random.seed(42)
        grid = random.sample(grid, args.sample)
    if args.max_configs > 0 and len(grid) > args.max_configs:
        grid = grid[:args.max_configs]

    if not args.quiet:
        print(f"Parameter configs to test: {len(grid)} (original grid size {total_grid})")

    results = []
    start = time.time()
    last_print_len = 0
    for idx, params in enumerate(grid, start=1):
        cfg = BacktestConfig(
            symbol=args.symbol,
            threshold_points=params['threshold_points'],
            window_size=params['window_size'],
            lookahead=params['lookahead'],
            rr=params['rr'],
            min_leg_distance_points=params['min_leg_distance_points'],
            price_scale=args.price_scale,
            initial_balance=args.initial_balance,
            risk_pct=params['risk_pct'],
            use_external_logic=args.use_external,
            fib_entry_min=params.get('fib_entry_min') if args.use_external else 0.705,
            fib_entry_max=params.get('fib_entry_max') if args.use_external else 0.9,
            external_quiet=args.quiet,
        )
        per_file_summaries = []
        for path, df in datasets:
            engine = BacktestEngine(cfg)
            trades, summary = engine.run(df)
            per_file_summaries.append(summary)
        agg = aggregate_results(per_file_summaries)
        agg.update({
            'threshold_points': params['threshold_points'],
            'window_size': params['window_size'],
            'lookahead': params['lookahead'],
            'rr': params['rr'],
            'min_leg_distance_points': params['min_leg_distance_points'],
            'risk_pct': params['risk_pct'],
            'use_external_logic': args.use_external,
        })
        if args.use_external:
            agg['fib_entry_min'] = params.get('fib_entry_min')
            agg['fib_entry_max'] = params.get('fib_entry_max')
        # استخراج میانگین تایم‌فریم برای گزارش
        tf_vals = [s.get('timeframe_minutes') for s in per_file_summaries if s.get('timeframe_minutes')]
        if tf_vals:
            agg['timeframe_minutes_mean'] = sum(tf_vals)/len(tf_vals)
            agg['lookahead_minutes_mean'] = agg['timeframe_minutes_mean'] * params['lookahead']
        agg['score'] = score_rule(agg)
        results.append(agg)
        # Progress / ETA
        if not args.quiet:
            if idx % args.progress_every == 0 or idx == len(grid):
                elapsed = time.time() - start
                avg_per = elapsed / idx
                remaining = (len(grid) - idx) * avg_per
                pct = idx / len(grid) * 100.0
                line = (f"[{idx}/{len(grid)}] {pct:5.1f}% | elapsed {elapsed:7.1f}s | "
                        f"ETA {remaining:7.1f}s | avg {avg_per*1000:6.1f} ms/conf | last score {agg['score']:.3f}")
                if not args.no_dynamic_progress:
                    pad = max(0, last_print_len - len(line))
                    print('\r' + line + ' ' * pad, end='', flush=True)
                    last_print_len = len(line)
                else:
                    print(line)

    if not args.quiet and not args.no_dynamic_progress:
        print()  # newline after dynamic progress

    if not results:
        print('No results produced.')
        return

    # Sort by composite score desc
    results.sort(key=lambda r: r.get('score', -math.inf), reverse=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    full_csv = outdir / 'optimization_results.csv'
    pd.DataFrame(results).to_csv(full_csv, index=False)

    top_n = results[:args.top]
    summary_json = outdir / 'top_results.json'
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(top_n, f, ensure_ascii=False, indent=2)

    if not args.quiet:
        print(f"\n✅ Saved full results -> {full_csv}")
        print(f"✅ Saved top {len(top_n)} -> {summary_json}")
        print("Best config:")
        best = top_n[0]
        for k in ['threshold_points','window_size','lookahead','rr','min_leg_distance_points','risk_pct','fib_entry_min','fib_entry_max','win_rate_pct','net_R','profit_factor','max_drawdown_R','score']:
            if k in best:
                print(f"- {k}: {best[k]}")
    else:
        # Quiet mode concise output (single line summary)
        best = top_n[0]
        print(f"BEST threshold={best.get('threshold_points')} win_rate={best.get('win_rate_pct')} net_R={best.get('net_R')} PF={best.get('profit_factor')} DD_R={best.get('max_drawdown_R')} score={round(best.get('score',0),3)} -> {summary_json}")


if __name__ == '__main__':
    main()
