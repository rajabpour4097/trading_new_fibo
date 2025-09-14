#!/usr/bin/env python3
"""CLI to run the lightweight backtester over an OHLC CSV and write trades CSV + summary."""
import argparse
from pathlib import Path
import json
import pandas as pd
from datetime import datetime, timedelta, timezone

from engine import BacktestEngine, BacktestConfig, load_ohlc_csv
from advanced_engine import AdvancedBacktester, AdvancedConfig
from metatrader5_config import MT5_CONFIG, TRADING_CONFIG, DYNAMIC_RISK_CONFIG
from mt5_data_fetch import fetch_last_month_m1, fetch_ticks_between, fetch_m1_range
from mt5_data_fetch import fetch_last_month_m1, fetch_ticks_between, fetch_m1_range, fetch_symbol_specs
from advanced_engine import AdvancedBacktester, AdvancedConfig
from advanced_tick_engine import AdvancedTickBacktester, FullConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="Path to OHLC CSV; omit to fetch last month M1 from MT5")
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--threshold", type=int, default=6, help="leg threshold in points")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--lookahead", type=int, default=20)
    ap.add_argument("--rr", type=float, default=1.2)
    ap.add_argument("--minleg", type=int, default=4, help="minimum leg distance in points")
    ap.add_argument("--scale", type=int, default=100000, help="price scale for points (5-digit FX=100000)")
    ap.add_argument("--outdir", default="backtest/results")
    ap.add_argument("--initial-balance", type=float, default=10000.0, help="starting balance for equity simulation")
    ap.add_argument("--risk-pct", type=float, default=0.01, help="risk per trade as fraction of equity (e.g. 0.01 = 1%)")
    ap.add_argument("--use-external", action="store_true", help="Use real strategy logic (get_legs, swings, fibo)")
    ap.add_argument("--fib-entry-min", type=float, default=0.705, help="Min fib retracement ratio for entry (external logic)")
    ap.add_argument("--fib-entry-max", type=float, default=0.9, help="Max fib retracement ratio for entry (external logic)")
    ap.add_argument("--quiet-ext", action="store_true", help="Suppress prints from external strategy functions")
    ap.add_argument("--mt5-month", action="store_true", help="Fetch last month M1 data from MT5 instead of CSV")
    ap.add_argument("--with-ticks", action="store_true", help="Also fetch tick data for the same period for intra-bar simulation")
    ap.add_argument("--no-tick-path", action="store_true", help="Disable tick path even if ticks fetched")
    ap.add_argument("--advanced", action="store_true", help="Use advanced engine replicating live bot logic (fib + dynamic risk)")
    ap.add_argument("--full-logic", action="store_true", help="Use tick-accurate full logic engine (two-touch, stages, commissions)")
    ap.add_argument("--debug", action="store_true", help="Enable verbose debug for full logic engine")
    ap.add_argument("--single-touch", action="store_true", help="Full logic: allow single 0.705 touch (no second confirmation) to arm entry")
    ap.add_argument("--sell-only", action="store_true", help="Full logic: only allow SELL positions (for weekly test)")
    # Date range (UTC) for MT5 M1 fetch
    ap.add_argument("--from", dest="date_from", help="Start date YYYY-MM-DD (UTC) for M1 fetch")
    ap.add_argument("--to", dest="date_to", help="End date YYYY-MM-DD (UTC, inclusive end day) for M1 fetch")
    # Advanced overrides
    ap.add_argument("--adv-initial-balance", type=float, default=10000.0, help="Advanced: initial balance")
    ap.add_argument("--adv-risk-pct", type=float, default=0.01, help="Advanced: risk percent per trade (0.01=1%)")
    ap.add_argument("--adv-threshold", type=int, default=None, help="Advanced: leg threshold override")
    ap.add_argument("--adv-win-ratio", type=float, default=None, help="Advanced: base RR (win ratio)")
    ap.add_argument("--adv-min-sl-pips", type=float, default=None, help="Advanced: minimum SL distance in pips")
    ap.add_argument("--adv-commission-per-lot", type=float, default=None, help="Advanced: commission per lot override")
    ap.add_argument("--adv-window-size", type=int, default=None, help="Full logic: override window_size (default from TRADING_CONFIG)")
    args = ap.parse_args()

    cfg = BacktestConfig(
        symbol=args.symbol,
        threshold_points=args.threshold,
        window_size=args.window,
        lookahead=args.lookahead,
        rr=args.rr,
        min_leg_distance_points=args.minleg,
        price_scale=args.scale,
    initial_balance=args.initial_balance,
    risk_pct=args.risk_pct,
    use_external_logic=args.use_external,
    fib_entry_min=args.fib_entry_min,
    fib_entry_max=args.fib_entry_max,
    external_quiet=args.quiet_ext,
    )

    if args.mt5_month or args.date_from or args.date_to or not args.csv:
        if args.date_from:
            from datetime import datetime, timezone, timedelta
            start = datetime.strptime(args.date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if args.date_to:
                end = datetime.strptime(args.date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
            else:
                end = start + timedelta(days=30)
            print(f"Fetching M1 data {start.date()} -> {(end - timedelta(seconds=1)).date()} from MT5...")
            df = fetch_m1_range(args.symbol, start, end)
        else:
            print("Fetching last month M1 data from MT5...")
            df = fetch_last_month_m1(args.symbol)
    else:
        df = load_ohlc_csv(args.csv)

    ticks_df = None
    if args.with_ticks or args.full_logic:
        print("Fetching tick data (may take time)...")
        start = df.index[0].to_pydatetime()
        end = df.index[-1].to_pydatetime()
        ticks_df = fetch_ticks_between(args.symbol, start, end)
        print(f"Ticks fetched: {len(ticks_df)}")
    if args.full_logic:
        specs = fetch_symbol_specs(args.symbol)
        full_cfg = FullConfig(
            symbol=args.symbol,
            threshold=(args.adv_threshold if args.adv_threshold is not None else TRADING_CONFIG.get('threshold',7)),
            window_size=(args.adv_window_size if args.adv_window_size is not None else TRADING_CONFIG.get('window_size',100)),
            min_sl_pips=(args.adv_min_sl_pips if args.adv_min_sl_pips is not None else TRADING_CONFIG.get('min_sl_pips',2.0)),
            win_ratio=(args.adv_win_ratio if args.adv_win_ratio is not None else MT5_CONFIG.get('win_ratio',1.2)),
            risk_pct=args.adv_risk_pct,
            initial_balance=args.adv_initial_balance,
            commission_per_lot=(args.adv_commission_per_lot if args.adv_commission_per_lot is not None else DYNAMIC_RISK_CONFIG.get('commission_per_lot',4.5)),
            debug=args.debug,
            require_second_touch=(not args.single_touch),
            sell_only=args.sell_only or TRADING_CONFIG.get('sell_only', False)
        )
        bt = AdvancedTickBacktester(full_cfg, specs, ticks_df)
        trades, summary, events = bt.run(df, progress=True)
        trades_df = bt.trades_df(trades)
        events_df = bt.events_df(events)
    elif args.advanced:
        adv_cfg = AdvancedConfig(symbol=args.symbol,
                                 initial_balance=args.adv_initial_balance,
                                 risk_pct=args.adv_risk_pct,
                                 threshold=(args.adv_threshold if args.adv_threshold is not None else TRADING_CONFIG.get('threshold',7)),
                                 win_ratio=(args.adv_win_ratio if args.adv_win_ratio is not None else MT5_CONFIG.get('win_ratio',1.2)),
                                 min_sl_pips=(args.adv_min_sl_pips if args.adv_min_sl_pips is not None else TRADING_CONFIG.get('min_sl_pips',2.0)),
                                 commission_per_lot=(args.adv_commission_per_lot if args.adv_commission_per_lot is not None else DYNAMIC_RISK_CONFIG.get('commission_per_lot',4.5)))
        adv_engine = AdvancedBacktester(adv_cfg)
        trades, summary = adv_engine.run(df)
        trades_df = adv_engine.trades_dataframe(trades)
    else:
        cfg.use_tick_path = (not args.no_tick_path) and args.with_ticks
        engine = BacktestEngine(cfg)
        trades, summary = engine.run(df, ticks=ticks_df)
        trades_df = engine.to_dataframe(trades)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.csv).stem if args.csv else f"{args.symbol}_M1_month"
    trades_path = outdir / f"{stem}_trades.csv"
    summary_path = outdir / f"{stem}_summary.json"

    trades_df.to_csv(trades_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.full_logic:
        events_path = outdir / f"{stem}_events.csv"
        events_df.to_csv(events_path, index=False)
        print(f"Saved events  -> {events_path}")

    print("Backtest summary:")
    for k, v in summary.items():
        print(f"- {k}: {v}")
    print(f"\nSaved trades -> {trades_path}")
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
