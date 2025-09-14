"""
Lightweight backtesting engine for OHLC data.

- Input: pandas DataFrame with columns ['open','high','low','close'] indexed by timestamp
- Strategy: simple swing/leg detector with threshold (points), RR target, fixed lookahead
- Output: list of trades, per-trade R result, summary metrics

Note: This engine is self-contained and does NOT depend on your incomplete get_legs/swing
implementations. You can later plug in your own signal generator.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Union
import math
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np


@dataclass
class BacktestConfig:
    symbol: str = "EURUSD"
    threshold_points: int = 6            # leg threshold in points (assumes 5-digit -> 0.00010)
    window_size: int = 100               # candles used to detect a setup
    lookahead: int = 20                  # candles to evaluate outcome
    rr: float = 1.2                      # risk:reward target
    min_leg_distance_points: int = 4     # minimum leg size in points
    price_scale: int = 100000            # 5-digit FX scaling for points
    initial_balance: float = 10_000.0    # starting balance for equity simulation
    risk_pct: float = 0.01               # fraction of equity risked per trade (1% default)
    use_external_logic: bool = False     # if True use real bot logic (get_legs, swings, fibo)
    fib_entry_min: float = 0.705         # min fib retracement (only for external logic)
    fib_entry_max: float = 0.9           # max fib retracement
    external_quiet: bool = False         # suppress stdout from external strategy functions
    use_tick_path: bool = False          # if True and ticks provided, simulate intra-bar path via ticks


@dataclass
class Trade:
    ts_entry: pd.Timestamp
    ts_exit: Optional[pd.Timestamp]
    direction: str  # 'bullish' or 'bearish'
    entry: float
    stop: float
    target: float
    rr: float
    outcome: str          # 'win' | 'loss' | 'timeout'
    bars_to_outcome: Optional[int]
    mfe: float            # max favorable excursion (in R)
    mae: float            # max adverse excursion (in R)
    r_result: float       # realized R multiple
    balance_before: Optional[float] = None
    balance_after: Optional[float] = None
    cash_result: Optional[float] = None


class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.cfg = config
        # Lazy import external strategy components; keep engine standalone if unavailable
        self._external_available = False
        if self.cfg.use_external_logic:
            try:
                # Root path insertion so scripts run from backtest/ still find top-level modules
                import sys, os
                root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                if root_dir not in sys.path:
                    sys.path.insert(0, root_dir)
                from get_legs import get_legs as external_get_legs  # type: ignore
                from swing import get_swing_points  # type: ignore
                from fibo_calculate import fibonacci_retracement  # type: ignore
                self._external_get_legs = external_get_legs
                self._external_get_swing_points = get_swing_points
                self._external_fibo = fibonacci_retracement
                self._external_available = True
            except Exception as e:  # pragma: no cover - defensive
                # Fallback silently to internal logic
                self._external_available = False
                self._external_import_error = str(e)

    @staticmethod
    def _ensure_status(df: pd.DataFrame) -> pd.DataFrame:
        if 'status' not in df.columns:
            status = np.where(df['close'] >= df['open'], 'bullish', 'bearish')
            df = df.copy()
            df['status'] = status
        return df

    def _detect_legs(self, data: pd.DataFrame) -> List[Dict]:
        """Simple leg detector similar to backtest/import sys_small.py.
        Threshold measured in points using cfg.price_scale.
        """
        if self.cfg.use_external_logic and self._external_available:
            return self._detect_legs_external(data)
        if data.empty:
            return []
        legs: List[Dict] = []
        start_idx = data.index[0]
        j = 0
        i = 1
        scale = self.cfg.price_scale
        th = self.cfg.threshold_points

        while i < len(data):
            close_i = float(data['close'].iloc[i])
            if i == 1:
                legs.append({
                    'start': start_idx,
                    'start_value': float(data['close'].iloc[0]),
                    'direction': 'up' if close_i >= float(data['close'].iloc[0]) else 'down',
                    'end': data.index[i],
                    'end_value': close_i,
                })
                j += 1
            else:
                last_end = float(legs[j-1]['end_value'])
                price_diff_pts = abs(close_i - last_end) * scale
                if price_diff_pts >= th:
                    current_dir = 'up' if close_i > last_end else 'down'
                    if legs[j-1]['direction'] != current_dir:
                        legs.append({
                            'start': legs[j-1]['end'],
                            'start_value': legs[j-1]['end_value'],
                            'direction': current_dir,
                            'end': data.index[i],
                            'end_value': close_i,
                        })
                        j += 1
                    else:
                        legs[j-1]['end'] = data.index[i]
                        legs[j-1]['end_value'] = close_i
            i += 1
        return legs

    def _build_signal(self, window: pd.DataFrame, legs: List[Dict]) -> Optional[Dict]:
        """Very simple swing signal: require 3 last legs and a direction change.
        Entry = last close; Stop = previous leg end; Target = entry +/- rr*|entry-stop|.
        """
        if self.cfg.use_external_logic and self._external_available:
            return self._build_signal_external(window, legs)
        if len(legs) < 3:
            return None
        last3 = legs[-3:]
        # Heuristic: if leg[-2] reversed vs leg[-1] start and magnitude >= min_leg_distance
        prev_end = float(last3[-2]['end_value'])
        entry = float(window['close'].iloc[-1])
        dist_pts = abs(entry - prev_end) * self.cfg.price_scale
        if dist_pts < self.cfg.min_leg_distance_points:
            return None

        # Direction: if entry > prev_end -> bullish, else bearish
        direction = 'bullish' if entry > prev_end else 'bearish'
        if direction == 'bullish':
            stop = prev_end
            target = entry + self.cfg.rr * abs(entry - stop)
        else:
            stop = prev_end
            target = entry - self.cfg.rr * abs(entry - stop)

        return {
            'direction': direction,
            'entry': entry,
            'stop': stop,
            'target': target,
            'rr': self.cfg.rr,
        }

    def _simulate_trade(self,
                        window_end_ts: pd.Timestamp,
                        future: pd.DataFrame,
                        signal: Dict,
                        ticks: Optional[pd.DataFrame] = None) -> Trade:
        entry = float(signal['entry'])
        stop = float(signal['stop'])
        target = float(signal['target'])
        rr = float(signal['rr'])
        direction = signal['direction']

        risk = abs(entry - stop)
        mfe_r = 0.0
        mae_r = 0.0
        outcome = 'timeout'
        exit_ts: Optional[pd.Timestamp] = None
        bars_to = None

        # If tick path provided and enabled, iterate ticks chronologically for precise sequence
        if ticks is not None and self.cfg.use_tick_path and not ticks.empty:
            # Expected tick columns: bid, ask (fallback to 'last' if missing)
            # For MFE/MAE we track on basis of mid price approximation
            price_col_bid = 'bid' if 'bid' in ticks.columns else None
            price_col_ask = 'ask' if 'ask' in ticks.columns else None
            price_col_last = 'last' if 'last' in ticks.columns else None
            for idx, (ts, row) in enumerate(ticks.iterrows()):
                bid = float(row[price_col_bid]) if price_col_bid else float(row.get(price_col_last, entry))
                ask = float(row[price_col_ask]) if price_col_ask else float(row.get(price_col_last, entry))
                mid = (bid + ask) / 2.0
                if direction == 'bullish':
                    # For long: stop triggers on bid <= stop, target on bid >= target (conservative fill criteria)
                    mfe_r = max(mfe_r, (bid - entry) / risk)
                    mae_r = min(mae_r, (bid - entry) / risk)
                    if bid <= stop:
                        outcome = 'loss'
                        exit_ts = ts
                        bars_to = None  # tick-level exit; bar count not meaningful
                        break
                    if bid >= target:
                        outcome = 'win'
                        exit_ts = ts
                        bars_to = None
                        break
                else:  # bearish
                    # For short: stop triggers on ask >= stop, target on ask <= target
                    mfe_r = max(mfe_r, (entry - ask) / risk)
                    mae_r = min(mae_r, (entry - ask) / risk)
                    if ask >= stop:
                        outcome = 'loss'
                        exit_ts = ts
                        bars_to = None
                        break
                    if ask <= target:
                        outcome = 'win'
                        exit_ts = ts
                        bars_to = None
                        break
        else:
            for idx, row in enumerate(future.itertuples(index=True)):
                high = float(row.high)
                low = float(row.low)
                # excursions
                if direction == 'bullish':
                    mfe_r = max(mfe_r, (high - entry) / risk)
                    mae_r = min(mae_r, (low - entry) / risk)
                    if low <= stop:
                        outcome = 'loss'
                        exit_ts = row.Index
                        bars_to = idx + 1
                        break
                    if high >= target:
                        outcome = 'win'
                        exit_ts = row.Index
                        bars_to = idx + 1
                        break
                else:
                    mfe_r = max(mfe_r, (entry - low) / risk)
                    mae_r = min(mae_r, (entry - high) / risk)
                    if high >= stop:
                        outcome = 'loss'
                        exit_ts = row.Index
                        bars_to = idx + 1
                        break
                    if low <= target:
                        outcome = 'win'
                        exit_ts = row.Index
                        bars_to = idx + 1
                        break

        r_result = rr if outcome == 'win' else (-1.0 if outcome == 'loss' else 0.0)

        return Trade(
            ts_entry=window_end_ts,
            ts_exit=exit_ts,
            direction=direction,
            entry=entry,
            stop=stop,
            target=target,
            rr=rr,
            outcome=outcome,
            bars_to_outcome=bars_to,
            mfe=mfe_r,
            mae=mae_r,
            r_result=r_result,
        )

    def run(self, df: pd.DataFrame, ticks: Optional[pd.DataFrame] = None) -> Tuple[List[Trade], Dict]:
        df = self._ensure_status(df)
        trades: List[Trade] = []
        n = len(df)
        step = max(1, self.cfg.window_size // 10)
        show_progress = True
        last_pct_print = -5

        i = 0
        while i + self.cfg.window_size + 1 < n:
            window = df.iloc[i:i + self.cfg.window_size]
            legs = self._detect_legs(window)
            signal = self._build_signal(window, legs)
            if signal:
                future = df.iloc[i + self.cfg.window_size: i + self.cfg.window_size + self.cfg.lookahead]
                ticks_slice: Optional[pd.DataFrame] = None
                if ticks is not None and self.cfg.use_tick_path:
                    entry_ts = window.index[-1]
                    if not future.empty:
                        end_ts = future.index[-1]
                        ticks_slice = ticks.loc[(ticks.index > entry_ts) & (ticks.index <= end_ts)] if not ticks.empty else None
                trade = self._simulate_trade(window.index[-1], future, signal, ticks_slice)
                trades.append(trade)
            i += step
            if show_progress:
                pct = i / max(1, (n - self.cfg.window_size)) * 100
                if pct - last_pct_print >= 1 or pct >= 100:
                    bar_len = 30
                    filled = int(bar_len * min(100, pct) / 100)
                    bar = '█' * filled + '-' * (bar_len - filled)
                    print(f"\rProgress |{bar}| {pct:5.1f}% Trades:{len(trades)}", end='', flush=True)
                    last_pct_print = pct

        self._apply_equity(trades)
        summary = self._summarize(trades)
        if show_progress:
            print()
        return trades, summary

    # ------------------- External strategy integration ------------------- #
    def _detect_legs_external(self, data: pd.DataFrame) -> List[Dict]:
        """Use user's real get_legs() implementation.
        Note: Original get_legs scales difference by *10000; we pass threshold_points directly.
        Returns list of legs with keys start, end, start_value, end_value, direction (up/down).
        """
        try:
            if self.cfg.external_quiet:
                import io, contextlib, sys  # local import to avoid overhead when not used
                dummy = io.StringIO()
                with contextlib.redirect_stdout(dummy):
                    legs = self._external_get_legs(data, custom_threshold=self.cfg.threshold_points)
            else:
                legs = self._external_get_legs(data, custom_threshold=self.cfg.threshold_points)
            return legs if isinstance(legs, list) else []
        except Exception:  # pragma: no cover
            return []

    def _build_signal_external(self, window: pd.DataFrame, legs: List[Dict]) -> Optional[Dict]:
        """Build signal using swing & fibonacci logic.

        Process:
        - Take last 3 legs -> test swing (bullish/bearish) using get_swing_points
        - Compute fib retracement on impulse leg (leg[0] of the 3) start->end
        - If current price within [fib_entry_min,fib_entry_max] retracement zone, create trade
        Entry = current close; Stop = impulse start (bullish) or impulse start (bearish)
        Target = entry +/- rr*(|entry-stop|)
        """
        if len(legs) < 3:
            return None
        last3 = legs[-3:]
        try:
            swing_type, is_swing = self._external_get_swing_points(window, last3)
        except Exception:
            return None
        if not is_swing:
            return None
        # Impulse leg assumed first of the pattern per original logic
        impulse = last3[0]
        start_price = float(impulse['start_value'])
        end_price = float(impulse['end_value'])
        # Ensure ordering for retracement math (bullish: end > start)
        if swing_type == 'bullish' and end_price < start_price:
            start_price, end_price = end_price, start_price
        if swing_type == 'bearish' and end_price > start_price:
            start_price, end_price = end_price, start_price
        try:
            fib_levels = self._external_fibo(start_price, end_price)
        except Exception:
            return None
        price = float(window['close'].iloc[-1])
        # Determine retracement ratio current price lies at
        total_range = abs(end_price - start_price) or 1e-9
        retr_ratio = (price - start_price) / total_range if swing_type == 'bullish' else (start_price - price) / total_range
        # Check within desired fib retracement band
        if not (self.cfg.fib_entry_min <= retr_ratio <= self.cfg.fib_entry_max):
            return None
        # Construct trade parameters
        if swing_type == 'bullish':
            entry = price
            stop = start_price
            risk = abs(entry - stop)
            if risk * self.cfg.price_scale < 1:  # avoid zero / tiny risk
                return None
            target = entry + self.cfg.rr * risk
            direction = 'bullish'
        else:  # bearish
            entry = price
            stop = start_price
            risk = abs(entry - stop)
            if risk * self.cfg.price_scale < 1:
                return None
            target = entry - self.cfg.rr * risk
            direction = 'bearish'
        return {
            'direction': direction,
            'entry': entry,
            'stop': stop,
            'target': target,
            'rr': self.cfg.rr,
        }

    def _apply_equity(self, trades: List[Trade]):
        """Populate balance_before/after and cash_result given config risk model."""
        balance = self.cfg.initial_balance
        for t in trades:
            t.balance_before = balance
            risk_amount = balance * self.cfg.risk_pct  # fixed fraction risk model
            t.cash_result = t.r_result * risk_amount   # R * risk
            balance = balance + t.cash_result
            t.balance_after = balance

    @staticmethod
    def _streaks(outcomes: List[str]) -> Tuple[int, int]:
        max_win = max_loss = cur_win = cur_loss = 0
        for o in outcomes:
            if o == 'win':
                cur_win += 1
                cur_loss = 0
            elif o == 'loss':
                cur_loss += 1
                cur_win = 0
            else:  # timeout breaks both streaks
                cur_win = cur_loss = 0
            max_win = max(max_win, cur_win)
            max_loss = max(max_loss, cur_loss)
        return max_win, max_loss

    @staticmethod
    def _summarize(trades: List[Trade]) -> Dict:
        wins = sum(1 for t in trades if t.outcome == 'win')
        losses = sum(1 for t in trades if t.outcome == 'loss')
        timeouts = sum(1 for t in trades if t.outcome == 'timeout')
        total = len(trades)
        win_rate = (wins / total * 100.0) if total else 0.0
        sum_pos = sum(t.r_result for t in trades if t.r_result > 0)
        sum_neg = -sum(t.r_result for t in trades if t.r_result < 0)
        profit_factor = (sum_pos / sum_neg) if sum_neg > 0 else math.inf if sum_pos > 0 else 0.0
        avg_r = (sum(t.r_result for t in trades) / total) if total else 0.0
        # R equity curve
        r_eq = np.cumsum([t.r_result for t in trades]) if trades else np.array([])
        max_dd_r = 0.0
        if r_eq.size:
            peak = -1e9
            for v in r_eq:
                peak = max(peak, v)
                max_dd_r = min(max_dd_r, v - peak)
        # Cash equity curve
        cash_eq = np.array([t.balance_after for t in trades if t.balance_after is not None])
        max_dd_cash = 0.0
        if cash_eq.size:
            peak = -1e18
            for v in cash_eq:
                peak = max(peak, v)
                max_dd_cash = min(max_dd_cash, v - peak)
        # Expectancy (average R per trade) & payoff ratio
        wins_r = [t.r_result for t in trades if t.r_result > 0]
        losses_r = [-t.r_result for t in trades if t.r_result < 0]
        avg_win_r = np.mean(wins_r) if wins_r else 0.0
        avg_loss_r = np.mean(losses_r) if losses_r else 0.0
        payoff = (avg_win_r / avg_loss_r) if avg_loss_r > 0 else math.inf if avg_win_r > 0 else 0.0
        expectancy_r = avg_r
        outcomes = [t.outcome for t in trades]
        max_win_streak, max_loss_streak = BacktestEngine._streaks(outcomes)
        net_r = sum(t.r_result for t in trades)
        net_cash = (trades[-1].balance_after - trades[0].balance_before) if trades else 0.0
        ret_pct = (net_cash / trades[0].balance_before * 100.0) if trades else 0.0
        return {
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'timeouts': timeouts,
            'win_rate_pct': round(win_rate, 2),
            'net_R': round(net_r, 3),
            'avg_R': round(avg_r, 3),
            'expectancy_R': round(expectancy_r, 3),
            'avg_win_R': round(avg_win_r, 3),
            'avg_loss_R': round(avg_loss_r, 3),
            'payoff_ratio': round(payoff, 3),
            'profit_factor': round(profit_factor, 3),
            'max_drawdown_R': round(max_dd_r, 3),
            'net_profit_cash': round(net_cash, 2),
            'return_pct': round(ret_pct, 2),
            'max_drawdown_cash': round(max_dd_cash, 2),
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
        }

    @staticmethod
    def to_dataframe(trades: List[Trade]) -> pd.DataFrame:
        if not trades:
            return pd.DataFrame(columns=[f.name for f in Trade.__dataclass_fields__.values()])
        return pd.DataFrame([asdict(t) for t in trades])


def load_ohlc_csv(path: str) -> pd.DataFrame:
    """Load OHLC data supporting two formats:
    1) Standard CSV with header containing at least timestamp, open, high, low, close
    2) Headerless MT5 export (tab-separated): timestamp\topen\thigh\tlow\tclose\tvolume
    """
    path_obj = Path(path)
    # First attempt: assume standard CSV with header
    try:
        df = pd.read_csv(path_obj, parse_dates=['timestamp'], index_col='timestamp')
        if {'open','high','low','close'}.issubset(df.columns):
            return df.sort_index()
    except Exception:
        pass

    # Fallback: attempt headerless tab-separated (MT5 style)
    # Peek first line to decide
    with open(path_obj, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline().strip()
    # Heuristic: if first token matches YYYY-MM-DD and has 6 fields when split by tab
    import re
    parts_tab = first_line.split('\t')
    ts_match = re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', parts_tab[0]) if parts_tab else None
    if ts_match and len(parts_tab) >= 5:
        names = ['timestamp','open','high','low','close','volume']
        df = pd.read_csv(path_obj, sep='\t', header=None, names=names, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df[['open','high','low','close']].sort_index()

    raise ValueError(f"Unsupported OHLC format for file: {path}")
