# (File added in patch earlier but not created due to apply order). See earlier patch content placeholder.
# Minimal stub to satisfy import; full implementation should have been created.
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import sys, os, math
import pandas as pd
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from metatrader5_config import TRADING_CONFIG, DYNAMIC_RISK_CONFIG, MT5_CONFIG
from fibo_calculate import fibonacci_retracement
from get_legs import get_legs
from swing import get_swing_points
from utils import BotState

# Analytics integration
try:
    from analytics.hooks import log_signal, log_position_event
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    def log_signal(*args, **kwargs): pass
    def log_position_event(*args, **kwargs): pass

@dataclass
class FullConfig:
    symbol: str = MT5_CONFIG.get('symbol', 'EURUSD')
    threshold: int = TRADING_CONFIG.get('threshold', 7)  # updated default from main_metatrader
    window_size: int = TRADING_CONFIG.get('window_size', 100)
    min_sl_pips: float = TRADING_CONFIG.get('min_sl_pips', 2.0)
    win_ratio: float = MT5_CONFIG.get('win_ratio', 1.2)
    risk_pct: float = MT5_CONFIG.get('risk_pct', 0.01)
    initial_balance: float = 100_000.0
    commission_per_lot: float = DYNAMIC_RISK_CONFIG.get('commission_per_lot', 4.5)  # updated default
    commission_round_trip: bool = DYNAMIC_RISK_CONFIG.get('round_trip', False)
    dynamic_enabled: bool = DYNAMIC_RISK_CONFIG.get('enable', True)
    stages: List[Dict] = tuple(DYNAMIC_RISK_CONFIG.get('stages', []))
    two_pip_tolerance: float = 2.0
    progress_log_step_pct: float = 5.0
    max_trade_minutes: int = 8 * 60
    debug: bool = False
    require_second_touch: bool = True  # مطابق main_metatrader.py دو-تماس اجباری
    sell_only: bool = TRADING_CONFIG.get('sell_only', False)  # فیلتر برای تست یک هفته‌ای

@dataclass
class FullTrade:
    direction: str
    ts_entry: pd.Timestamp
    ts_exit: pd.Timestamp
    entry: float
    stop_initial: float
    stop_final: float
    target_initial: float
    target_final: float
    r_initial: float
    r_result: float
    volume: float
    commission: float
    cash_result: float
    outcome: str
    stages: List[str]
    touches: int

class AdvancedTickBacktester:
    def __init__(self, cfg: FullConfig, symbol_specs: Dict, ticks: Optional[pd.DataFrame]):
        self.cfg = cfg
        self.specs = symbol_specs
        self.ticks = ticks if ticks is not None else pd.DataFrame(columns=['bid','ask'])
        self.state = BotState()
        self.balance = cfg.initial_balance
        self.trades: List[FullTrade] = []
        self.events: List[Dict] = []
        self.open_positions: List[Dict] = []
        self.debug = cfg.debug
        self.skip_reasons = {
            'no_fib':0,
            'not_true_position':0,
            'invalid_sl':0,
            'min_stop_distance':0,
            'vol_zero':0,
            'sell_only_filter':0
        }

    def _pip_size(self) -> float:
        digits = self.specs.get('digits', 5)
        point = self.specs.get('point', 0.00001)
        return point * 10 if digits in (3,5) else point

    def _min_stop_distance(self) -> float:
        point = self.specs.get('point', 0.00001)
        stops_level = self.specs.get('trade_stops_level', 0)
        broker_min = stops_level * point
        min_user = self.cfg.min_sl_pips * self._pip_size()
        return max(broker_min, min_user, 3 * point)

    def _log_event(self, kind: str, ts, **fields):
        row = {'event': kind, 'ts': ts}
        row.update(fields)
        self.events.append(row)

    def _risk_volume(self, entry: float, sl: float, spread: float) -> float:
        tick_size = self.specs.get('tick_size') or self.specs.get('point')
        tick_value = self.specs.get('tick_value')
        if (not tick_value or tick_value==0) and self.specs.get('contract_size') and tick_size:
            tick_value = self.specs['contract_size'] * tick_size
            if self.debug:
                self._log_event('debug_tick_value_fallback', None, tick_value=tick_value)
        if not tick_size or not tick_value:
            return 0.0
        risk_price = abs(entry - sl)
        risk_ticks = risk_price / tick_size
        price_risk_per_lot = risk_ticks * tick_value
        spread_ticks = spread / tick_size
        spread_cost_per_lot = spread_ticks * tick_value
        commission_per_lot = self.cfg.commission_per_lot
        total_per_lot = price_risk_per_lot + spread_cost_per_lot + commission_per_lot
        if total_per_lot <= 0:
            return 0.0
        risk_money = self.balance * self.cfg.risk_pct
        vol = risk_money / total_per_lot
        step = self.specs.get('volume_step', 0.01) or 0.01
        vmin = self.specs.get('volume_min', step)
        steps = round(vol / step)
        vol = steps * step
        return max(vmin, round(vol, 2))

    def _update_state(self, window: pd.DataFrame):
        try:
            legs_all = get_legs(window, custom_threshold=self.cfg.threshold, verbose=self.debug)
        except Exception:
            legs_all = []
        legs = legs_all[-3:] if len(legs_all) >= 3 else legs_all
        swing_type = None
        is_swing = False
        if len(legs) == 3:
            try:
                swing_type, is_swing = get_swing_points(window, legs)
            except Exception:
                pass
        
        last_close = window['close'].iloc[-1]
        last_high = window['high'].iloc[-1]
        last_low = window['low'].iloc[-1]
        status = window['status'].iloc[-1]
        
        # فاز 1: تشخیص اولیه swing - مطابق main_metatrader.py
        if is_swing and self.state.fib_levels is None:
            if swing_type == 'bullish':
                if last_close >= legs[0]['end_value']:
                    start_price = last_high
                    end_price = legs[1]['end_value']
                    if last_high >= legs[1]['end_value']:
                        fib = fibonacci_retracement(end_price=end_price, start_price=start_price)
                        self.state.fib_levels = fib
                        self._log_event('fib_init', window.index[-1], direction='bullish', fib=fib)
                    elif self.state.fib_levels and last_low < self.state.fib_levels['1.0']:
                        self.state.reset()
            elif swing_type == 'bearish':
                if last_close <= legs[0]['end_value']:
                    start_price = last_low
                    end_price = legs[1]['end_value']
                    if last_low <= legs[1]['end_value']:
                        fib = fibonacci_retracement(start_price=start_price, end_price=end_price)
                        self.state.fib_levels = fib
                        self._log_event('fib_init', window.index[-1], direction='bearish', fib=fib)
                    elif self.state.fib_levels and last_high > self.state.fib_levels['1.0']:
                        self.state.reset()
        
        # فاز 2: به‌روزرسانی در swing مشابه - مطابق main_metatrader.py
        elif is_swing and self.state.fib_levels and hasattr(self.state, 'last_swing_type') and self.state.last_swing_type == swing_type:
            if swing_type == 'bullish':
                if last_high >= legs[1]['end_value']:
                    start_price = last_high
                    end_price = legs[1]['end_value']
                    fib = fibonacci_retracement(start_price=start_price, end_price=end_price)
                    self.state.fib_levels = fib
                    self._log_event('fib_update', window.index[-1], direction='bullish', fib=fib)
                elif self.state.fib_levels and last_low < self.state.fib_levels['1.0']:
                    self.state.reset()
            elif swing_type == 'bearish':
                if last_low <= legs[1]['end_value']:
                    start_price = last_low
                    end_price = legs[1]['end_value']
                    fib = fibonacci_retracement(start_price=start_price, end_price=end_price)
                    self.state.fib_levels = fib
                    self._log_event('fib_update', window.index[-1], direction='bearish', fib=fib)
                elif self.state.fib_levels and last_high > self.state.fib_levels['1.0']:
                    self.state.reset()
        
        # بروزرسانی فیبوناچی برای کمتر از 3 leg - مطابق main_metatrader.py
        elif len(legs) < 3:
            if self.state.fib_levels:
                if hasattr(self.state, 'last_swing_type'):
                    if self.state.last_swing_type == 'bullish' or swing_type == 'bullish':
                        if self.state.fib_levels['0.0'] < last_high:
                            start_price = last_high
                            fib = fibonacci_retracement(start_price=start_price, end_price=self.state.fib_levels['1.0'])
                            self.state.fib_levels = fib
                            self._log_event('fib_update_ext', window.index[-1], direction='bullish', fib=fib)
                    if self.state.last_swing_type == 'bearish' or swing_type == 'bearish':
                        if self.state.fib_levels['0.0'] > last_low:
                            start_price = last_low
                            fib = fibonacci_retracement(start_price=start_price, end_price=self.state.fib_levels['1.0'])
                            self.state.fib_levels = fib
                            self._log_event('fib_update_ext', window.index[-1], direction='bearish', fib=fib)
        
        # ذخیره نوع swing
        if swing_type:
            self.state.last_swing_type = swing_type
        
        # بررسی تماس با سطح 0.705 - مطابق main_metatrader.py
        fib = self.state.fib_levels
        if fib:
            fib705 = fib['0.705']
            if last_low <= fib705 <= last_high:
                # بررسی تماس با سطح 0.705 برای تایید معامله - مطابق main_metatrader.py
                if status == 'bullish':
                    if self.state.last_touched_705_point_up is None:
                        self.state.last_touched_705_point_up = window.iloc[-1]
                        self.state._touch_count = 1
                        self._log_event('touch_705_up_first', window.index[-1])
                        if not self.cfg.require_second_touch:
                            self.state.true_position = True
                            self._log_event('touch_705_up_single_ok', window.index[-1])
                    elif window.iloc[-1]['status'] != self.state.last_touched_705_point_up['status']:
                        self.state.true_position = True
                        self.state._touch_count = 2
                        self._log_event('touch_705_up_second', window.index[-1])
                else:  # bearish
                    if self.state.last_touched_705_point_down is None:
                        self.state.last_touched_705_point_down = window.iloc[-1]
                        self.state._touch_count = 1
                        self._log_event('touch_705_down_first', window.index[-1])
                        if not self.cfg.require_second_touch:
                            self.state.true_position = True
                            self._log_event('touch_705_down_single_ok', window.index[-1])
                    elif window.iloc[-1]['status'] != self.state.last_touched_705_point_down['status']:
                        self.state.true_position = True
                        self.state._touch_count = 2
                        self._log_event('touch_705_down_second', window.index[-1])
            
            # بررسی نقض سطح 1.0 - مطابق main_metatrader.py
            pip = self._pip_size()
            if (hasattr(self.state, 'last_swing_type') and self.state.last_swing_type == 'bullish' and 
                last_low < fib['1.0']):
                self._log_event('fib_reset', window.index[-1], reason='violated_1.0_bullish')
                self.state.reset()
            elif (hasattr(self.state, 'last_swing_type') and self.state.last_swing_type == 'bearish' and 
                  last_high > fib['1.0']):
                self._log_event('fib_reset', window.index[-1], reason='violated_1.0_bearish')
                self.state.reset()

    def _maybe_open(self, bar_ts: pd.Timestamp, bar_row):
        if self.state.fib_levels is None:
            self.skip_reasons['no_fib'] += 1
            return
        if not self.state.true_position:
            self.skip_reasons['not_true_position'] += 1
            return
            
        fib = self.state.fib_levels
        close_price = bar_row['close']
        pip = self._pip_size()
        two_pips = self.cfg.two_pip_tolerance * pip
        min_dist = self._min_stop_distance()
        
        # تشخیص جهت معامله - بر اساس نوع فیبوناچی که شکل گرفته
        fib = self.state.fib_levels
        close_price = bar_row['close']
        
        # تشخیص جهت بر اساس موقعیت فعلی نسبت به فیبوناچی
        if hasattr(self.state, 'last_swing_type') and self.state.last_swing_type:
            direction = self.state.last_swing_type
        else:
            # fallback: تشخیص بر اساس موقعیت قیمت نسبت به فیبوناچی
            if close_price > fib['0.705']:
                direction = 'buy'  # قیمت بالای 0.705 - احتمال bullish setup
            else:
                direction = 'sell'  # قیمت پایین 0.705 - احتمال bearish setup
        
        # فیلتر sell_only برای تست یک هفته‌ای
        if self.cfg.sell_only and direction != 'sell':
            self.skip_reasons['sell_only_filter'] = self.skip_reasons.get('sell_only_filter', 0) + 1
            self.state.reset()
            return
        
        # محاسبه SL بر اساس منطق main_metatrader.py
        if direction == 'buy':
            # منطق انتخاب SL مطابق main_metatrader.py
            is_close_to_09 = abs(fib['0.9'] - close_price) <= two_pips
            candidate_sl = fib['1.0'] if is_close_to_09 else fib['0.9']
            
            # گارد جهت
            if candidate_sl >= close_price:
                candidate_sl = float(fib['1.0'])
            
            # اطمینان از فاصله حداقل
            min_pip_dist = 2  # حداقل 2 پیپ واقعی
            min_abs_dist = max(min_pip_dist * pip, min_dist)
            
            if (close_price - candidate_sl) < min_abs_dist:
                adj = close_price - min_abs_dist
                if adj <= 0:
                    self.skip_reasons['invalid_sl'] += 1
                    self.state.reset()
                    return
                candidate_sl = float(adj)
            
            if candidate_sl >= close_price:
                self.skip_reasons['invalid_sl'] += 1
                self.state.reset()
                return
                
        else:  # sell
            # منطق انتخاب SL مطابق main_metatrader.py  
            is_close_to_09 = abs(fib['0.9'] - close_price) <= two_pips
            candidate_sl = fib['1.0'] if is_close_to_09 else fib['0.9']
            
            if candidate_sl <= close_price:
                candidate_sl = float(fib['1.0'])
            
            min_pip_dist = 2.0
            min_abs_dist = max(min_pip_dist * pip, min_dist)
            
            if (candidate_sl - close_price) < min_abs_dist:
                adj = close_price + min_abs_dist
                candidate_sl = float(adj)
            
            if candidate_sl <= close_price:
                self.skip_reasons['invalid_sl'] += 1
                self.state.reset()
                return
        
        # محاسبه حجم
        spread = 0.0
        if not self.ticks.empty:
            window_ticks = self.ticks.loc[(self.ticks.index > bar_ts - pd.Timedelta(minutes=1)) & (self.ticks.index <= bar_ts)]
            if not window_ticks.empty:
                spread = (window_ticks['ask'] - window_ticks['bid']).tail(50).mean()
        
        vol = self._risk_volume(close_price, candidate_sl, spread)
        if vol <= 0:
            self.skip_reasons['vol_zero'] += 1
            self.state.reset()
            return
        
        # محاسبه TP
        risk_price = abs(close_price - candidate_sl)
        target = close_price + risk_price * self.cfg.win_ratio if direction == 'buy' else close_price - risk_price * self.cfg.win_ratio
        
        # لاگ سیگنال (مطابق main_metatrader.py)
        if ANALYTICS_AVAILABLE:
            try:
                log_signal(
                    symbol=self.cfg.symbol,
                    strategy="swing_fib_v1_backtest",
                    direction=direction,
                    rr=self.cfg.win_ratio,
                    entry=close_price,
                    sl=candidate_sl,
                    tp=target,
                    fib=fib,
                    confidence=None,
                    features_json=None,
                    note="triggered_by_pullback_bt"
                )
            except Exception:
                pass
        
        # کسر کمیسیون
        commission_total = self.cfg.commission_per_lot * vol
        self.balance -= commission_total if not self.cfg.commission_round_trip else commission_total / 2.0
        
        # ایجاد پوزیشن
        pos = {
            'direction': direction,
            'entry_ts': bar_ts,
            'entry': close_price,
            'stop': candidate_sl,
            'target': target,
            'initial_stop': candidate_sl,
            'initial_target': target,
            'risk_price': risk_price,
            'volume': vol,
            'commission': commission_total,
            'stages_done': [],
            'touches': getattr(self.state, '_touch_count', 2),  # شمارش تماس‌ها
            'open_balance': self.balance,
        }
        self.open_positions.append(pos)
        self._log_event('trade_open', bar_ts, direction=direction, entry=close_price, stop=candidate_sl, target=target, volume=vol)
        
        # Analytics logging for position open
        if ANALYTICS_AVAILABLE:
            try:
                log_position_event(
                    symbol=self.cfg.symbol,
                    ticket=id(pos),  # استفاده از id به عنوان ticket موقت
                    event='open',
                    direction=direction,
                    entry=close_price,
                    current_price=close_price,
                    sl=candidate_sl,
                    tp=target,
                    profit_R=0.0,
                    stage=0,
                    risk_abs=risk_price,
                    locked_R=None,
                    volume=vol,
                    note='backtest_position_open'
                )
            except Exception:
                pass
        
        self.state.reset()

    def _apply_stages(self, pos: Dict, bid: float, ask: float, ts):
        if not self.cfg.dynamic_enabled:
            return
            
        price = bid if pos['direction'] == 'buy' else ask
        profit_price = (price - pos['entry']) if pos['direction'] == 'buy' else (pos['entry'] - price)
        profit_R = profit_price / pos['risk_price'] if pos['risk_price'] else 0.0
        
        for stage in self.cfg.stages:
            sid = stage.get('id')
            if sid in pos['stages_done']:
                continue
                
            # مرحله پوشش کمیسیون - مطابق main_metatrader.py
            if stage.get('type') == 'commission' and pos['commission'] > 0:
                # محاسبه ارزش پولی 1R تقریبی
                tick_size = self.specs.get('tick_size') or self.specs.get('point')
                tick_value = self.specs.get('tick_value')
                
                if not tick_value and self.specs.get('contract_size') and tick_size:
                    tick_value = self.specs['contract_size'] * tick_size
                
                if tick_value and tick_size and pos['volume'] > 0:
                    # تبدیل کمیسیون به R
                    price_offset = pos['commission'] / pos['volume']
                    
                    # اگر سود قیمتی * حجم >= کمیسیون
                    if profit_price * pos['volume'] >= pos['commission']:
                        if pos['direction'] == 'buy':
                            new_sl = pos['entry'] + price_offset
                        else:
                            new_sl = pos['entry'] - price_offset
                        
                        # تطبیق SL فقط در صورت بهبود
                        if ((pos['direction'] == 'buy' and new_sl > pos['stop']) or 
                            (pos['direction'] == 'sell' and new_sl < pos['stop'])):
                            pos['stop'] = new_sl
                            pos['stages_done'].append(sid)
                            self._log_event('stage_commission', ts, ticket_id=id(pos), new_stop=pos['stop'])
                            
                            # Analytics logging for commission stage
                            if ANALYTICS_AVAILABLE:
                                try:
                                    log_position_event(
                                        symbol=self.cfg.symbol,
                                        ticket=id(pos),
                                        event='commission_cover',
                                        direction=pos['direction'],
                                        entry=pos['entry'],
                                        current_price=price,
                                        sl=new_sl,
                                        tp=pos['target'],
                                        profit_R=profit_R,
                                        stage=None,
                                        risk_abs=pos['risk_price'],
                                        locked_R=None,
                                        volume=pos['volume'],
                                        note='commission stage trigger'
                                    )
                                except Exception:
                                    pass
            else:
                # مراحل R-based - مطابق main_metatrader.py
                trig = stage.get('trigger_R')
                if trig is not None and profit_R >= trig:
                    sl_lock_R = stage.get('sl_lock_R', trig)
                    tp_R = stage.get('tp_R')
                    
                    if pos['direction'] == 'buy':
                        new_sl = pos['entry'] + sl_lock_R * pos['risk_price']
                        if new_sl > pos['stop']:  # فقط در صورت بهبود
                            pos['stop'] = new_sl
                        if tp_R:
                            pos['target'] = pos['entry'] + tp_R * pos['risk_price']
                    else:
                        new_sl = pos['entry'] - sl_lock_R * pos['risk_price'] 
                        if new_sl < pos['stop']:  # فقط در صورت بهبود
                            pos['stop'] = new_sl
                        if tp_R:
                            pos['target'] = pos['entry'] - tp_R * pos['risk_price']
                    
                    pos['stages_done'].append(sid)
                    self._log_event('stage_R', ts, stage=sid, new_stop=pos['stop'], new_target=pos['target'])
                    
                    # Analytics logging for R-based stage
                    if ANALYTICS_AVAILABLE:
                        try:
                            log_position_event(
                                symbol=self.cfg.symbol,
                                ticket=id(pos),
                                event=sid,
                                direction=pos['direction'],
                                entry=pos['entry'],
                                current_price=price,
                                sl=pos['stop'],
                                tp=pos['target'],
                                profit_R=profit_R,
                                stage=None,
                                risk_abs=pos['risk_price'],
                                locked_R=sl_lock_R,
                                volume=pos['volume'],
                                note=f'stage {sid} trigger'
                            )
                        except Exception:
                            pass

    def _process_ticks_until_next_bar(self, bar_start: pd.Timestamp, bar_end: pd.Timestamp):
        if not self.open_positions or self.ticks.empty:
            return
        ticks = self.ticks.loc[(self.ticks.index > bar_start) & (self.ticks.index <= bar_end)]
        if ticks.empty:
            return
        remaining = []
        for pos in self.open_positions:
            closed = False
            for ts, row in ticks.iterrows():
                bid = row.get('bid', np.nan)
                ask = row.get('ask', np.nan)
                if np.isnan(bid) or np.isnan(ask):
                    continue
                self._apply_stages(pos, bid, ask, ts)
                if pos['direction'] == 'buy':
                    if bid <= pos['stop']:
                        self._close_position(pos, ts, 'loss'); closed = True; break
                    if bid >= pos['target']:
                        self._close_position(pos, ts, 'win'); closed = True; break
                else:
                    if ask >= pos['stop']:
                        self._close_position(pos, ts, 'loss'); closed = True; break
                    if ask <= pos['target']:
                        self._close_position(pos, ts, 'win'); closed = True; break
            if not closed:
                remaining.append(pos)
        self.open_positions = remaining

    def _close_position(self, pos: Dict, ts, outcome: str):
        if outcome == 'loss':
            r_result = -1.0
        elif outcome == 'win':
            r_result = abs(pos['target'] - pos['entry']) / pos['risk_price']
        else:
            r_result = 0.0
        risk_cash = pos['open_balance'] * self.cfg.risk_pct
        gross = r_result * risk_cash
        if self.cfg.commission_round_trip:
            self.balance -= pos['commission'] / 2.0
        self.balance += gross
        trade = FullTrade(
            direction=pos['direction'],
            ts_entry=pos['entry_ts'],
            ts_exit=ts,
            entry=pos['entry'],
            stop_initial=pos['initial_stop'],
            stop_final=pos['stop'],
            target_initial=pos['initial_target'],
            target_final=pos['target'],
            r_initial=self.cfg.win_ratio,
            r_result=r_result,
            volume=pos['volume'],
            commission=pos['commission'],
            cash_result=gross,
            outcome=outcome,
            stages=list(pos['stages_done']),
            touches=pos['touches'],
        )
        self.trades.append(trade)
        self._log_event('trade_close', ts, outcome=outcome, r=r_result, balance=self.balance)
        
        # Analytics logging for position close
        if ANALYTICS_AVAILABLE:
            try:
                current_price = pos.get('last_price', pos['entry'])  # fallback if no last_price
                log_position_event(
                    symbol=self.cfg.symbol,
                    ticket=id(pos),
                    event='close',
                    direction=pos['direction'],
                    entry=pos['entry'],
                    current_price=current_price,
                    sl=pos['stop'],
                    tp=pos['target'],
                    profit_R=r_result,
                    stage=None,
                    risk_abs=pos['risk_price'],
                    locked_R=None,
                    volume=pos['volume'],
                    note=f'closed_{outcome}'
                )
            except Exception:
                pass

    def run(self, ohlc: pd.DataFrame, progress: bool = True):
        df = ohlc.copy()
        if 'status' not in df.columns:
            df['status'] = np.where(df['close'] >= df['open'], 'bullish', 'bearish')
        n = len(df)
        ws = self.cfg.window_size
        last_pct = -self.cfg.progress_log_step_pct
        for i in range(ws, n):
            window = df.iloc[i-ws:i+1]
            bar_ts = window.index[-1]
            bar_row = window.iloc[-1]
            self._update_state(window)
            self._maybe_open(bar_ts, bar_row)
            if i > ws:
                prev_ts = df.index[i-1]
                self._process_ticks_until_next_bar(prev_ts, bar_ts)
            if progress:
                pct = (i-ws)/(n-ws)*100 if n>ws else 100
                if pct - last_pct >= self.cfg.progress_log_step_pct or pct >= 100:
                    self._log_event('progress', bar_ts, percent=round(pct,2))
                    bar_len = 30
                    filled = int(bar_len * pct / 100)
                    bar = '█' * filled + '-' * (bar_len - filled)
                    print(f"\rProgress |{bar}| {pct:5.1f}%  Balance: {self.balance:,.2f}", end='', flush=True)
                    last_pct = pct
        final_ts = df.index[-1]
        for pos in list(self.open_positions):
            self._close_position(pos, final_ts, 'timeout')
        if progress:
            print()  # newline after progress bar
        summary = self._summarize()
        return self.trades, summary, self.events

    def _summarize(self):
        wins = sum(1 for t in self.trades if t.outcome=='win')
        losses = sum(1 for t in self.trades if t.outcome=='loss')
        timeouts = sum(1 for t in self.trades if t.outcome=='timeout')
        total = len(self.trades)
        net_r = sum(t.r_result for t in self.trades)
        avg_r = net_r/total if total else 0.0
        sum_pos = sum(t.r_result for t in self.trades if t.r_result>0)
        sum_neg = -sum(t.r_result for t in self.trades if t.r_result<0)
        pf = (sum_pos/sum_neg) if sum_neg>0 else (math.inf if sum_pos>0 else 0.0)
        return {
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'timeouts': timeouts,
            'win_rate_pct': round((wins/total*100) if total else 0.0,2),
            'net_R': round(net_r,3),
            'avg_R': round(avg_r,3),
            'profit_factor': round(pf,3),
            'ending_balance': round(self.balance,2),
            'skip_reasons': self.skip_reasons,
        }

    @staticmethod
    def trades_df(trades: List[FullTrade]):
        if not trades:
            return pd.DataFrame(columns=[f.name for f in FullTrade.__dataclass_fields__.values()])
        return pd.DataFrame([asdict(t) for t in trades])

    @staticmethod
    def events_df(events: List[Dict]):
        if not events:
            return pd.DataFrame(columns=['event','ts'])
        return pd.DataFrame(events)

__all__ = ['FullConfig','AdvancedTickBacktester']
