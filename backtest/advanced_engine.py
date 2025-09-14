"""Advanced backtest engine attempting to replicate live bot logic (main_metatrader).

Scope / Included:
- Legs detection via existing get_legs()
- Swing detection via get_swing_points()
- Fibonacci state machine similar to runtime (BotState like)
- Entry conditions (two touches of 0.705 zone logic) & SL selection (0.9 vs 1.0 with 2 pip tolerance)
- Dynamic risk stages (locking SL & extending TP) based on DYNAMIC_RISK_CONFIG
- Commission per lot (simplified) & risk-based volume sizing (approx FX majors)
- Equity curve with % risk per trade (risk_pct) and R multiples after commission

Simplifications / Not Included Yet:
- Exact broker min stop distance (approximation only)
- Spread impact sequence vs SL/TP (uses bar high/low ordering assumption; optional tick path could refine)
- Partial fills, slippage, spread widening
- Commission coverage stage uses simplified profitability check

Use with: run_backtest.py --advanced --use-external (external not required; this file imports strategy components directly)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import math
import pandas as pd
import numpy as np

import sys, os
# Ensure project root (one level up) is on sys.path so we can import config modules
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from metatrader5_config import TRADING_CONFIG, DYNAMIC_RISK_CONFIG, MT5_CONFIG
from fibo_calculate import fibonacci_retracement
from get_legs import get_legs
from swing import get_swing_points
from utils import BotState


@dataclass
class AdvancedConfig:
    symbol: str = MT5_CONFIG.get('symbol', 'EURUSD')
    window_size: int = TRADING_CONFIG.get('window_size', 100)
    threshold: int = TRADING_CONFIG.get('threshold', 6)
    win_ratio: float = MT5_CONFIG.get('win_ratio', 1.2)
    risk_pct: float = 0.01
    initial_balance: float = 10_000.0
    min_sl_pips: float = TRADING_CONFIG.get('min_sl_pips', 2.0)
    commission_per_lot: float = DYNAMIC_RISK_CONFIG.get('commission_per_lot', 0.0)
    round_trip_commission: bool = DYNAMIC_RISK_CONFIG.get('round_trip', False)
    dynamic_enabled: bool = DYNAMIC_RISK_CONFIG.get('enable', True)
    base_tp_R: float = DYNAMIC_RISK_CONFIG.get('base_tp_R', 1.2)
    stages: List[Dict] = tuple(DYNAMIC_RISK_CONFIG.get('stages', []))  # immutable-ish
    pip_value_per_lot: float = 10.0  # approximation for major FX on USD account
    digits: int = 5  # fallback digits for pip calculation
    use_tick_path: bool = False  # reserved
    two_pip_tolerance: float = 2.0  # pips


@dataclass
class Position:
    direction: str  # 'buy'|'sell'
    entry_ts: pd.Timestamp
    entry_price: float
    stop: float
    target: float
    volume: float
    risk_price: float  # abs(entry-stop)
    open_commission: float
    stages_done: List[str]
    fib_snapshot: Dict


@dataclass
class AdvancedTrade:
    direction: str
    ts_entry: pd.Timestamp
    ts_exit: Optional[pd.Timestamp]
    entry: float
    stop_initial: float
    stop_exit: float
    target_initial: float
    target_final: float
    volume: float
    r_result: float
    cash_result: float
    commission: float
    outcome: str  # win|loss|timeout
    stages_applied: List[str]
    fib_levels: Dict


class AdvancedBacktester:
    def __init__(self, cfg: AdvancedConfig):
        self.cfg = cfg
        self.state = BotState()
        self.balance = cfg.initial_balance
        self.trades: List[AdvancedTrade] = []
        self.open_positions: List[Position] = []

    # ---- Utility pricing helpers ---- #
    def _pip_size(self) -> float:
        # For 5/3 digit FX: 1 pip = 10 * point; we approximate with digits
        return 10 ** (-(self.cfg.digits - 1)) if self.cfg.digits in (3, 5) else 10 ** (-self.cfg.digits)

    def _min_stop_distance(self) -> float:
        # approximate 3 points (0.3 pip for 5-digit) or min_sl_pips
        pip = self._pip_size()
        return max(self.cfg.min_sl_pips * pip, 3 * (pip / 10.0))

    def _risk_volume(self, entry: float, stop: float) -> float:
        pip = self._pip_size()
        risk_pips = abs(entry - stop) / pip
        if risk_pips <= 0:
            return 0.0
        risk_money = self.balance * self.cfg.risk_pct
        risk_per_lot = risk_pips * self.cfg.pip_value_per_lot
        vol = risk_money / risk_per_lot if risk_per_lot > 0 else 0.0
        # normalize to 0.01 lot steps
        return round(max(0.01, vol), 2)

    # ---- Core loop ---- #
    def run(self, df: pd.DataFrame) -> Tuple[List[AdvancedTrade], Dict]:
        df = df.copy()
        if 'status' not in df.columns:
            df['status'] = np.where(df['close'] >= df['open'], 'bullish', 'bearish')
        window_size = self.cfg.window_size

        for i in range(window_size, len(df)):
            window = df.iloc[i - window_size: i + 1]
            current_bar = df.iloc[i]
            self._update_strategy_state(window)
            # Check entry conditions after state update
            self._maybe_open_trade(window, current_bar.name)
            # Progress open positions (exit or stage modifications)
            self._update_open_positions(current_bar, current_bar.name)

        summary = self._summarize()
        return self.trades, summary

    # ---- Strategy State & Entry Logic (derived from main_metatrader) ---- #
    def _update_strategy_state(self, window: pd.DataFrame):
        # legs detection with threshold override
        try:
            legs_all = get_legs(window, custom_threshold=self.cfg.threshold)
        except Exception:
            legs_all = []
        if len(legs_all) >= 3:
            legs = legs_all[-3:]
        else:
            legs = legs_all

        swing_type = None
        is_swing = False
        if len(legs) == 3:
            try:
                swing_type, is_swing = get_swing_points(window, legs)
            except Exception:
                swing_type, is_swing = None, False

        # Re-implement a reduced subset of fib logic: whenever new swing & no fib -> set fib using impulse
        # For bullish: start=current high, end=previous leg end; For bearish analogous
        last_close = window['close'].iloc[-1]
        last_high = window['high'].iloc[-1]
        last_low = window['low'].iloc[-1]

        if is_swing and self.state.fib_levels is None:
            if swing_type == 'bullish':
                # ensure last close pierced first leg end
                impulse_end = legs[1]['end_value']
                if last_close >= legs[0]['end_value']:
                    start_price = last_high
                    end_price = impulse_end
                    if last_high >= impulse_end:
                        self.state.fib_levels = fibonacci_retracement(start_price=start_price, end_price=end_price)
            elif swing_type == 'bearish':
                impulse_end = legs[1]['end_value']
                if last_close <= legs[0]['end_value']:
                    start_price = last_low
                    end_price = impulse_end
                    if last_low <= impulse_end:
                        self.state.fib_levels = fibonacci_retracement(start_price=start_price, end_price=end_price)

        # Update existing fib / touches
        if self.state.fib_levels is not None:
            fib = self.state.fib_levels
            pip = self._pip_size()
            # Touch 0.705 logic to arm true_position after second status change
            if last_low <= fib['0.705'] and last_high >= fib['0.705']:
                # price traversed zone
                if last_close >= fib['0.705']:
                    # possible bullish side
                    if self.state.last_touched_705_point_up is None:
                        self.state.last_touched_705_point_up = window.iloc[-1]
                    elif window.iloc[-1]['status'] != self.state.last_touched_705_point_up['status']:
                        self.state.true_position = True
                elif last_close <= fib['0.705']:
                    if self.state.last_touched_705_point_down is None:
                        self.state.last_touched_705_point_down = window.iloc[-1]
                    elif window.iloc[-1]['status'] != self.state.last_touched_705_point_down['status']:
                        self.state.true_position = True

            # Reset if violated 1.0 boundary strongly (fail fib)
            if last_low < fib['1.0'] - 2 * pip and last_high > fib['1.0'] + 2 * pip:
                # wide violation
                self.state.reset()

    def _maybe_open_trade(self, window: pd.DataFrame, ts):
        if not self.state.true_position or self.state.fib_levels is None:
            return
        fib = self.state.fib_levels
        close_price = window['close'].iloc[-1]
        pip = self._pip_size()
        two_pips = self.cfg.two_pip_tolerance * pip
        # Determine direction from relative price to fib anchor range
        direction = 'buy' if close_price >= fib['0.705'] else 'sell'
        # Candidate SL selection
        if direction == 'buy':
            candidate_sl = fib['1.0'] if abs(fib['0.9'] - close_price) <= two_pips else fib['0.9']
            if candidate_sl >= close_price:
                candidate_sl = fib['1.0']
            if (close_price - candidate_sl) < self._min_stop_distance():
                candidate_sl = close_price - self._min_stop_distance()
            if candidate_sl >= close_price:
                self.state.reset()
                return
            risk_price = abs(close_price - candidate_sl)
            volume = self._risk_volume(close_price, candidate_sl)
            if volume <= 0:
                self.state.reset()
                return
            target = close_price + risk_price * self.cfg.win_ratio
        else:  # sell
            candidate_sl = fib['1.0'] if abs(fib['0.9'] - close_price) <= two_pips else fib['0.9']
            if candidate_sl <= close_price:
                candidate_sl = fib['1.0']
            if (candidate_sl - close_price) < self._min_stop_distance():
                candidate_sl = close_price + self._min_stop_distance()
            if candidate_sl <= close_price:
                self.state.reset()
                return
            risk_price = abs(close_price - candidate_sl)
            volume = self._risk_volume(close_price, candidate_sl)
            if volume <= 0:
                self.state.reset()
                return
            target = close_price - risk_price * self.cfg.win_ratio

        # Commission (entry side); if round_trip, half now, half at exit
        commission = self.cfg.commission_per_lot * volume
        entry_cash_risk = self.balance * self.cfg.risk_pct
        # Deduct entry commission immediately
        self.balance -= commission if not self.cfg.round_trip_commission else commission / 2.0
        pos = Position(
            direction=direction,
            entry_ts=ts,
            entry_price=close_price,
            stop=candidate_sl,
            target=target,
            volume=volume,
            risk_price=risk_price,
            open_commission=commission,
            stages_done=[],
            fib_snapshot=dict(fib),
        )
        self.open_positions.append(pos)
        # reset state after opening
        self.state.reset()

    def _update_open_positions(self, bar, ts):
        remaining: List[Position] = []
        for pos in self.open_positions:
            high = bar['high']
            low = bar['low']
            exited = False
            outcome = 'timeout'
            stop_hit = False
            target_hit = False
            # Evaluate hits (sequence ambiguity: assume worst-case for trader -> stop before target if both in same bar)
            if pos.direction == 'buy':
                if low <= pos.stop:
                    stop_hit = True
                elif high >= pos.target:
                    target_hit = True
            else:
                if high >= pos.stop:
                    stop_hit = True
                elif low <= pos.target:
                    target_hit = True

            # Dynamic stages if still open
            if not stop_hit and not target_hit and self.cfg.dynamic_enabled:
                profit_price = (bar['close'] - pos.entry_price) if pos.direction == 'buy' else (pos.entry_price - bar['close'])
                profit_R = profit_price / pos.risk_price if pos.risk_price else 0.0
                for stage in self.cfg.stages:
                    sid = stage.get('id')
                    if sid in pos.stages_done:
                        continue
                    # Commission cover stage
                    if stage.get('type') == 'commission' and pos.open_commission > 0:
                        # approximate profit cash = profit_R * (risk_cash) ; risk_cash = balance * risk_pct at entry (approx)
                        # Here risk_cash unknown after commission; we approximate with balance*risk_pct
                        risk_cash = self.balance * self.cfg.risk_pct
                        profit_cash = profit_R * risk_cash
                        if profit_cash >= pos.open_commission:
                            # lock SL to entry + commission cash in price terms (approx commission / (pip_value_per_lot*volume) pips)
                            pips_gain = pos.open_commission / (self.cfg.pip_value_per_lot * pos.volume)
                            price_gain = pips_gain * self._pip_size()
                            if pos.direction == 'buy':
                                pos.stop = pos.entry_price + price_gain
                            else:
                                pos.stop = pos.entry_price - price_gain
                            pos.stages_done.append(sid)
                    else:
                        trig = stage.get('trigger_R')
                        if trig is not None and profit_R >= trig:
                            sl_lock_R = stage.get('sl_lock_R', trig)
                            tp_R = stage.get('tp_R')
                            if pos.direction == 'buy':
                                pos.stop = pos.entry_price + sl_lock_R * pos.risk_price
                                if tp_R:
                                    pos.target = pos.entry_price + tp_R * pos.risk_price
                            else:
                                pos.stop = pos.entry_price - sl_lock_R * pos.risk_price
                                if tp_R:
                                    pos.target = pos.entry_price - tp_R * pos.risk_price
                            pos.stages_done.append(sid)

            if stop_hit or target_hit:
                exited = True
                if stop_hit:
                    outcome = 'loss'
                    r_result = -1.0
                elif target_hit:
                    # compute R relative to current target distance over initial risk
                    # assumption: target set at entry as RR (may have moved) -> profit_R = (abs(target-entry)/risk_price)
                    r_result = abs(pos.target - pos.entry_price) / pos.risk_price
                    outcome = 'win'
                # Commission at exit if round_trip
                exit_commission = 0.0
                if self.cfg.round_trip_commission:
                    exit_commission = pos.open_commission / 2.0
                # Cash result
                risk_cash = self.balance * self.cfg.risk_pct  # approximate risk cash (could track per-position)
                cash_result_gross = r_result * risk_cash
                total_commission = pos.open_commission if not self.cfg.round_trip_commission else pos.open_commission
                cash_result = cash_result_gross - (exit_commission if outcome == 'win' else 0.0)
                # Deduct commission (already deducted entry side earlier)
                self.balance += cash_result
                trade = AdvancedTrade(
                    direction=pos.direction,
                    ts_entry=pos.entry_ts,
                    ts_exit=ts,
                    entry=pos.entry_price,
                    stop_initial=pos.entry_price - pos.risk_price if pos.direction == 'buy' else pos.entry_price + pos.risk_price,
                    stop_exit=pos.stop,
                    target_initial=pos.target if not pos.stages_done else pos.entry_price + self.cfg.base_tp_R * pos.risk_price * (1 if pos.direction=='buy' else -1),
                    target_final=pos.target,
                    volume=pos.volume,
                    r_result=r_result,
                    cash_result=cash_result,
                    commission=pos.open_commission,
                    outcome=outcome,
                    stages_applied=list(pos.stages_done),
                    fib_levels=pos.fib_snapshot,
                )
                self.trades.append(trade)
            else:
                remaining.append(pos)

        self.open_positions = remaining

    # ---- Summary ---- #
    def _summarize(self) -> Dict:
        wins = sum(1 for t in self.trades if t.outcome == 'win')
        losses = sum(1 for t in self.trades if t.outcome == 'loss')
        total = len(self.trades)
        net_r = sum(t.r_result for t in self.trades)
        avg_r = net_r / total if total else 0.0
        win_rate = wins / total * 100 if total else 0.0
        sum_pos = sum(t.r_result for t in self.trades if t.r_result > 0)
        sum_neg = -sum(t.r_result for t in self.trades if t.r_result < 0)
        pf = (sum_pos / sum_neg) if sum_neg > 0 else math.inf if sum_pos > 0 else 0.0
        return {
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate_pct': round(win_rate, 2),
            'net_R': round(net_r, 3),
            'avg_R': round(avg_r, 3),
            'profit_factor': round(pf, 3),
            'ending_balance': round(self.balance, 2),
        }

    @staticmethod
    def trades_dataframe(trades: List[AdvancedTrade]) -> pd.DataFrame:
        if not trades:
            return pd.DataFrame(columns=[f.name for f in AdvancedTrade.__dataclass_fields__.values()])
        return pd.DataFrame([asdict(t) for t in trades])


__all__ = [
    'AdvancedConfig',
    'AdvancedBacktester',
]
