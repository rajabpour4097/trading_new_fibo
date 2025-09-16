import MetaTrader5 as mt5
from datetime import datetime
from fibo_calculate import fibonacci_retracement
import numpy as np
import pandas as pd
from time import sleep
from colorama import init, Fore
from get_legs import get_legs
from mt5_connector import MT5Connector
from swing import get_swing_points
from utils import BotState
from save_file import log
from metatrader5_config import MT5_CONFIG, TRADING_CONFIG, DYNAMIC_RISK_CONFIG
from email_notifier import send_trade_email_async
from analytics.hooks import log_signal, log_position_event


def main():
    # راه‌اندازی MT5 و colorama
    init(autoreset=True)
    mt5_conn = MT5Connector()

    if not mt5_conn.initialize():
        print("❌ Failed to connect to MT5")
        return

    # Initial state با تنظیمات - مطابق main_saver_copy2.py
    state = BotState()
    state.reset()

    start_index = 0
    win_ratio = MT5_CONFIG['win_ratio']
    threshold = TRADING_CONFIG['threshold']
    window_size = TRADING_CONFIG['window_size']
    min_swing_size = TRADING_CONFIG['min_swing_size']

    i = 1
    f = 0
    position_open = False
    last_swing_type = None
    fib_index = None
    fib0_point = None
    last_leg1_value = None
    end_price = None
    start_price = None

    print(f"🚀 MT5 Trading Bot Started...")
    print(f"📊 Config: Symbol={MT5_CONFIG['symbol']}, Lot={MT5_CONFIG['lot_size']}, Win Ratio={win_ratio}")
    print(f"⏰ Trading Hours (Iran): {MT5_CONFIG['trading_hours']['start']} - {MT5_CONFIG['trading_hours']['end']}")
    print(f"🇮🇷 Current Iran Time: {mt5_conn.get_iran_time().strftime('%Y-%m-%d %H:%M:%S')}")

    # در ابتدای main loop بعد از initialize
    print("🔍 Checking symbol properties...")
    mt5_conn.check_symbol_properties()
    print("🔍 Testing broker filling modes...")
    mt5_conn.test_filling_modes()
    mt5_conn.check_trading_limits()
    print("🔍 Checking account permissions...")
    mt5_conn.check_account_trading_permissions()
    print("🔍 Checking market state...")
    mt5_conn.check_market_state()
    print("-" * 50)

    # اضافه کردن متغیر برای ذخیره آخرین داده
    last_data_time = None
    wait_count = 0
    max_wait_cycles = 120  # پس از 60 ثانیه (120 * 0.5) اجبار به پردازش
    # نگهداری وضعیت قبلی قابلیت معامله برای ریست در انتهای ساعات ترید
    last_can_trade_state = None

    # بعد از تعریف متغیرها در main()
    def reset_state_and_window():
        nonlocal start_index, f
        state.reset()
        f = 0
        start_index = max(0, len(cache_data) - window_size)
        log(f'Reset state -> new start_index={start_index} (slice len={len(cache_data.iloc[start_index:])})', color='magenta')

    # حالت‌های مدیریت پوزیشن
    position_states = {}  # ticket -> {'entry':..., 'risk':..., 'direction':..., 'done_stages':set(), 'base_tp_R':float, 'commission_locked':False}

    def _digits():
        info = mt5.symbol_info(MT5_CONFIG['symbol'])
        return info.digits if info else 5

    def _round(p):
        return float(f"{p:.{_digits()}f}")

    # --- Touch epsilon (optional tolerance in pips) ---
    def _touch_epsilon_price() -> float:
        """Return price epsilon for touch detection based on TRADING_CONFIG['touch_epsilon_pips'] (default 0)."""
        try:
            pips = float(TRADING_CONFIG.get('touch_epsilon_pips', 0.0) or 0.0)
        except Exception:
            pips = 0.0
        return pips * _pip_size_for(MT5_CONFIG['symbol'])

    # --- Fib helpers: single source of truth for init/update ---
    def _init_fib_from_swing(s_type: str, legs_list, row):
        """Create fib_levels for a detected swing. Returns dict or None if gating not satisfied."""
        if len(legs_list) < 2:
            return None
        # Orientation contract per user:
        # - bullish swing: 0.0 -> current candle HIGH, 1.0 -> leg1 LOW (legs[1].end_value)
        # - bearish swing: 0.0 -> current candle LOW, 1.0 -> leg1 HIGH (legs[1].end_value)
        # Gating (critical): new fib only when last candle CLOSE crosses legs1 high/low (legs[1].start_value)
        try:
            l1_start = legs_list[1]['start_value']
            l1_end = legs_list[1]['end_value']
        except Exception:
            return None
        if s_type == 'bullish':
            # Require close >= legs1 high (start_value of a down leg)
            if row['close'] >= l1_start:
                return fibonacci_retracement(start_price=row['high'], end_price=l1_end)
            else:
                try:
                    log(f"Skip fib init: bullish close {row['close']} < legs1.high {l1_start} at {row.name}", color='lightyellow_ex')
                except Exception:
                    pass
        elif s_type == 'bearish':
            # Require close <= legs1 low (start_value of an up leg)
            if row['close'] <= l1_start:
                return fibonacci_retracement(start_price=row['low'], end_price=l1_end)
            else:
                try:
                    log(f"Skip fib init: bearish close {row['close']} > legs1.low {l1_start} at {row.name}", color='lightyellow_ex')
                except Exception:
                    pass
        return None

    def _update_fib0_if_extends(s_type: str, fib: dict, row, end_price_ref: float | None):
        """If price makes a new extreme in swing direction, update 0.0 and recompute other levels.
        Keeps 1.0 anchored to end_price_ref (leg1 end)."""
        if fib is None or end_price_ref is None:
            return fib
        # Update rules per orientation contract
        if s_type == 'bullish':
            # new higher high extends 0.0 upward (since 0.0 is current HIGH)
            if row['high'] > fib['0.0']:
                return fibonacci_retracement(start_price=row['high'], end_price=end_price_ref)
        elif s_type == 'bearish':
            # new lower low extends 0.0 downward (since 0.0 is current LOW)
            if row['low'] < fib['0.0']:
                return fibonacci_retracement(start_price=row['low'], end_price=end_price_ref)
        return fib

    # Note: fib 1.0 is locked after initialization; it changes only when a new swing (opposite or new same-direction) forms.

    def register_position(pos):
        # محاسبه R (ریسک اولیه)
        risk = abs(pos.price_open - pos.sl) if pos.sl else None
        if not risk or risk == 0:
            return
        position_states[pos.ticket] = {
            'entry': pos.price_open,
            'risk': risk,
            'direction': 'buy' if pos.type == mt5.POSITION_TYPE_BUY else 'sell',
            'done_stages': set(),
            'base_tp_R': DYNAMIC_RISK_CONFIG.get('base_tp_R', 1.2),
            'commission_locked': False
        }
        # رویداد ثبت پوزیشن
        try:
            log_position_event(
                symbol=MT5_CONFIG['symbol'],
                ticket=pos.ticket,
                event='open',
                direction=position_states[pos.ticket]['direction'],
                entry=pos.price_open,
                current_price=pos.price_open,
                sl=pos.sl,
                tp=pos.tp,
                profit_R=0.0,
                stage=0,
                risk_abs=risk,
                locked_R=None,
                volume=pos.volume,
                note='position registered'
            )
        except Exception:
            pass

    def manage_open_positions():
        if not DYNAMIC_RISK_CONFIG.get('enable'):
            return
        positions = mt5_conn.get_positions()
        if not positions:
            return
        tick = mt5.symbol_info_tick(MT5_CONFIG['symbol'])
        if not tick:
            return
        stages_cfg = DYNAMIC_RISK_CONFIG.get('stages', [])
        for pos in positions:
            if pos.ticket not in position_states:
                register_position(pos)
            st = position_states.get(pos.ticket)
            if not st:
                continue
            entry = st['entry']
            risk = st['risk']
            direction = st['direction']
            cur_price = tick.bid if direction == 'buy' else tick.ask
            # profit in price
            if direction == 'buy':
                price_profit = cur_price - entry
            else:
                price_profit = entry - cur_price
            profit_R = price_profit / risk if risk else 0.0
            modified_any = False

            # محاسبه ارزش پولی 1R تقریبی (بدون اسپرد) برای تبدیل کامیشن به R:
            # risk_abs_price = risk (فاصله قیمتی) * volume * contract ارزش واقعی - ساده‌سازی: فقط نسبت بر اساس فاصله قیمتی.
            # برای دقت بیشتر باید tick_value استفاده شود؛ اینجا ساده نگه می‌داریم.

            # عبور از مراحل R-based
            for stage_cfg in stages_cfg:
                sid = stage_cfg.get('id')
                if sid in st['done_stages']:
                    continue
                new_sl = None
                new_tp = None
                event_name = None
                locked_R = None

                # R-based stage
                trigger_R = stage_cfg.get('trigger_R')
                if trigger_R is not None and profit_R >= trigger_R:
                    sl_lock_R = stage_cfg.get('sl_lock_R', trigger_R)
                    tp_R = stage_cfg.get('tp_R')
                    # SL placement
                    if direction == 'buy':
                        new_sl = entry + sl_lock_R * risk
                        if tp_R:
                            new_tp = entry + tp_R * risk
                    else:
                        new_sl = entry - sl_lock_R * risk
                        if tp_R:
                            new_tp = entry - tp_R * risk
                    event_name = sid
                    locked_R = sl_lock_R

                if new_sl is not None:
                    # Round
                    new_sl_r = _round(new_sl)
                    new_tp_r = _round(new_tp) if new_tp is not None else pos.tp
                    # Apply only if improves
                    apply = False
                    if direction == 'buy' and new_sl_r > pos.sl:
                        apply = True
                    if direction == 'sell' and new_sl_r < pos.sl:
                        apply = True
                    if apply:
                        res = mt5_conn.modify_sl_tp(pos.ticket, new_sl=new_sl_r, new_tp=new_tp_r)
                        if res and getattr(res, 'retcode', None) == 10009:
                            st['done_stages'].add(sid)
                            modified_any = True
                            log(f'⚙️ Dynamic Risk Stage {sid} applied: ticket={pos.ticket} | Profit: {profit_R:.2f}R | SL: {new_sl_r} | TP: {new_tp_r}', color='cyan')
                            try:
                                log_position_event(
                                    symbol=MT5_CONFIG['symbol'],
                                    ticket=pos.ticket,
                                    event=event_name or sid,
                                    direction=direction,
                                    entry=entry,
                                    current_price=cur_price,
                                    sl=new_sl_r,
                                    tp=new_tp_r,
                                    profit_R=profit_R,
                                    stage=None,
                                    risk_abs=risk,
                                    locked_R=locked_R,
                                    volume=pos.volume,
                                    note=f'stage {sid} trigger'
                                )
                            except Exception:
                                pass
            if modified_any:
                position_states[pos.ticket] = st

    while True:
        try:
            # بررسی ساعات معاملاتی
            can_trade, trade_message = mt5_conn.can_trade()
            # اگر از حالت قابل معامله به غیرقابل معامله تغییر کرد => ریست کامل BotState
            try:
                if last_can_trade_state is True and not can_trade:
                    log("🧹 Trading hours ended -> resetting BotState to avoid stale context", color='magenta')
                    state.reset()
            except Exception:
                pass
            finally:
                last_can_trade_state = can_trade
            
            if not can_trade:
                log(f"⏰ {trade_message}", color='yellow', save_to_file=False)
                sleep(60)
                continue
            
            # دریافت داده از MT5
            cache_data = mt5_conn.get_historical_data(count=window_size * 2)
            
            if cache_data is None:
                log("❌ Failed to get data from MT5", color='red')
                sleep(5)
                continue
                
            cache_data['status'] = np.where(cache_data['open'] > cache_data['close'], 'bearish', 'bullish')
            
            # بررسی تغییر داده - مشابه main_saver_copy2.py
            current_time = cache_data.index[-1]
            if last_data_time is None:
                log(f"🔄 First run - processing data from {current_time}", color='cyan')
                last_data_time = current_time
                process_data = True
                wait_count = 0
            elif current_time != last_data_time:
                log(f"📊 New data received: {current_time} (previous: {last_data_time})", color='cyan')
                last_data_time = current_time
                process_data = True
                wait_count = 0
            else:
                wait_count += 1
                if wait_count % 20 == 0:  # هر 10 ثانیه یک بار لاگ
                    log(f"⏳ Waiting for new data... Current: {current_time} (wait cycles: {wait_count})", color='yellow')
                
                # اگر خیلی زیاد انتظار کشیدیم، اجبار به پردازش (در صورت تست)
                if wait_count >= max_wait_cycles:
                    log(f"⚠️ Force processing after {wait_count} cycles without new data", color='magenta')
                    process_data = True
                    wait_count = 0
                else:
                    process_data = False
            
            if process_data:
                log((' ' * 80 + '\n') * 3)
                log(f'Log number {i}:', color='lightred_ex')
                log(f' ' * 80)
                i += 1
                
                legs = get_legs(cache_data.iloc[start_index:])
                log(f'First len legs: {len(legs)}', color='green')
                log(f' ' * 80)

                if len(legs) > 2:
                    legs = legs[-3:]
                    swing_type, is_swing = get_swing_points(data=cache_data, legs=legs)

                    if is_swing == False and state.fib_levels is None:
                        log(f'No swing or fib levels and legs>2', color='blue')
                        log(f"{cache_data.loc[legs[0]['start']].name} {cache_data.loc[legs[0]['end']].name} "
                            f"{cache_data.loc[legs[1]['start']].name} {cache_data.loc[legs[1]['end']].name} "
                            f"{cache_data.loc[legs[2]['start']].name} {cache_data.loc[legs[2]['end']].name}", color='yellow')

                    if is_swing or state.fib_levels:
                        log(f'1- is_swing or fib_levels is not None code:411112', color='blue')
                        log(f"{swing_type} | {cache_data.loc[legs[0]['start']].name} {cache_data.loc[legs[0]['end']].name} "
                            f"{cache_data.loc[legs[1]['start']].name} {cache_data.loc[legs[1]['end']].name} "
                            f"{cache_data.loc[legs[2]['start']].name} {cache_data.loc[legs[2]['end']].name}", color='yellow')

                        log(f' ' * 80)
                        
                        # فاز 1: تشخیص اولیه/تعویض swing و ساخت fib از swing جدید
                        if is_swing and (state.fib_levels is None or last_swing_type != swing_type):
                            row = cache_data.iloc[-1]
                            end_price_ref = legs[1]['end_value'] if len(legs) >= 2 else None
                            # هر swing جدید => fib قبلی حذف و از نو ساخته شود (در صورت برآورده شدن گیت)
                            new_fib = _init_fib_from_swing(swing_type, legs, row)
                            if new_fib:
                                last_swing_type = swing_type
                                state.fib_levels = new_fib
                                # reset first-touch state on (re)build
                                state.last_touched_705_point_up = None
                                state.last_touched_705_point_down = None
                                state.true_position = False
                                # track fib timings/anchors
                                state.fib_built_time = row.name
                                state.fib0_last_update_time = row.name
                                # Lock fib 1.0 anchor on creation
                                state.fib1_time = legs[1]['end'] if len(legs) >= 2 else None
                                state.fib1_price = legs[1]['end_value'] if len(legs) >= 2 else None
                                fib0_point = cache_data.index.tolist().index(row.name)
                                fib_index = row.name
                                last_leg1_value = cache_data.index.tolist().index(legs[1]['end']) if len(legs) >= 2 else None
                                # save swing signature to detect new same-direction swings
                                try:
                                    state.swing_signature = (
                                        legs[0]['start'], legs[0]['end'],
                                        legs[1]['start'], legs[1]['end'],
                                        legs[2]['start'], legs[2]['end'],
                                    ) if len(legs) >= 3 else None
                                except Exception:
                                    state.swing_signature = None
                                legs = legs[-2:]
                                # Reset fib update counter on (re)initialization
                                f = 0
                                log(f'Fib initialized/reset for new swing -> {swing_type} | {row.name}', color='green')
                                log(f'fib_levels: {state.fib_levels}', color='yellow')
                                log(f'fib_index: {fib_index}', color='yellow')

                        # فاز 2: در swing مشابه - آپدیت 0.0 در صورت ثبت سقف/کف جدید + لاجیک لمس‌ها
                        elif is_swing and state.fib_levels and last_swing_type == swing_type:
                            log(f'is_swing and state.fib_levels and last_swing_type == swing_type code:4213312', color='yellow')
                            row = cache_data.iloc[-1]
                            # اگر سوئینگ همان جهت است اما ساختار سوئینگ جدید شده باشد (signature متفاوت)، فیبو را از نو بساز
                            try:
                                cur_sig = (
                                    legs[0]['start'], legs[0]['end'],
                                    legs[1]['start'], legs[1]['end'],
                                    legs[2]['start'], legs[2]['end'],
                                ) if len(legs) >= 3 else None
                            except Exception:
                                cur_sig = None
                            if cur_sig is not None and getattr(state, 'swing_signature', None) is not None and cur_sig != state.swing_signature:
                                new_fib = _init_fib_from_swing(swing_type, legs, row)
                                if new_fib:
                                    state.fib_levels = new_fib
                                    state.last_touched_705_point_up = None
                                    state.last_touched_705_point_down = None
                                    state.true_position = False
                                    state.fib_built_time = row.name
                                    state.fib0_last_update_time = row.name
                                    state.fib1_time = legs[1]['end'] if len(legs) >= 2 else state.fib1_time
                                    state.fib1_price = legs[1]['end_value'] if len(legs) >= 2 else state.fib1_price
                                    fib0_point = cache_data.index.tolist().index(row.name)
                                    fib_index = row.name
                                    last_leg1_value = cache_data.index.tolist().index(legs[1]['end']) if len(legs) >= 2 else last_leg1_value
                                    state.swing_signature = cur_sig
                                    f = 0
                                    log(f'Fib re-initialized for new same-direction swing -> {swing_type} | {row.name}', color='green')
                                    log(f'fib_levels: {state.fib_levels}', color='yellow')
                                    log(f'fib_index: {fib_index}', color='yellow')
                                    # پس از بازسازی، ادامهٔ این حلقه با فیب جدید
                            # برای آپدیت 0.0 از لنگر قفل‌شدهٔ 1.0 استفاده می‌کنیم (نه leg1 فعلی)
                            end_price_ref = state.fib1_price
                            # بروزرسانی 0.0 با سقف/کف جدید
                            new_fib = _update_fib0_if_extends(swing_type, state.fib_levels, row, end_price_ref)
                            if new_fib is not state.fib_levels:
                                state.fib_levels = new_fib
                                # reset first-touch state on 0.0 update
                                state.last_touched_705_point_up = None
                                state.last_touched_705_point_down = None
                                state.true_position = False
                                state.fib0_last_update_time = row.name
                                fib0_point = cache_data.index.tolist().index(row.name)
                                fib_index = row.name
                                # لنگر 1.0 قفل است؛ تغییر نمی‌کند
                                legs = legs[-2:]
                                # Count only extensions within the same swing
                                f += 1
                                log(f'Fib 0.0 updated (extend) at {row.name}', color='green')
                            # لمس 0.705 و گارد 1.0
                            if swing_type == 'bullish':
                                thr_705 = state.fib_levels['0.705']
                                eps = _touch_epsilon_price()
                                low_touch = row['low'] <= (thr_705 + eps)
                                high_touch = row['high'] >= (thr_705 - eps)
                                if low_touch:
                                    if state.last_touched_705_point_up is None:
                                        log(f'first touch 705 point code:7318455', color='green')
                                        state.last_touched_705_point_up = row
                                    elif row['status'] != state.last_touched_705_point_up['status']:
                                        log(f'Second touch 705 point code:7218455 {row.name}', color='green')
                                        state.true_position = True
                                        # track second touch
                                        state.last_second_touch_705_point_up = row
                                    else:
                                        try:
                                            if high_touch and not getattr(state, 'last_second_touch_705_point_up', None):
                                                log(f'BOTH-TOUCH: registering 2nd touch (bullish) at {row.name} via opposite side. low={row["low"]} high={row["high"]} thr={thr_705} eps={eps}', color='green')
                                                state.true_position = True
                                                state.last_second_touch_705_point_up = row
                                            else:
                                                delta = (row['low'] - thr_705)
                                                log(f'705 touched again but same status (no 2nd touch). prev={state.last_touched_705_point_up["status"]} cur={row["status"]} low={row["low"]} thr={thr_705} eps={eps} delta={delta}', color='yellow')
                                        except Exception:
                                            pass
                                elif state.fib_levels and row['low'] < state.fib_levels['1.0']:
                                    reset_state_and_window()
                                    legs = []
                            elif swing_type == 'bearish':
                                thr_705 = state.fib_levels['0.705']
                                eps = _touch_epsilon_price()
                                high_touch = row['high'] >= (thr_705 - eps)
                                low_touch = row['low'] <= (thr_705 + eps)
                                if high_touch:
                                    if state.last_touched_705_point_down is None:
                                        log(f'first touch 705 point code:6328455', color='red')
                                        state.last_touched_705_point_down = row
                                    elif row['status'] != state.last_touched_705_point_down['status']:
                                        log(f'Second touch 705 point code:6228455 {row.name}', color='green')
                                        state.true_position = True
                                        state.last_second_touch_705_point_down = row
                                    else:
                                        try:
                                            if low_touch and not getattr(state, 'last_second_touch_705_point_down', None):
                                                log(f'BOTH-TOUCH: registering 2nd touch (bearish) at {row.name} via opposite side. low={row["low"]} high={row["high"]} thr={thr_705} eps={eps}', color='green')
                                                state.true_position = True
                                                state.last_second_touch_705_point_down = row
                                            else:
                                                delta = (row['high'] - thr_705)
                                                log(f'705 touched again but same status (no 2nd touch). prev={state.last_touched_705_point_down["status"]} cur={row["status"]} high={row["high"]} thr={thr_705} eps={eps} delta={delta}', color='yellow')
                                        except Exception:
                                            pass
                                elif state.fib_levels and row['high'] > state.fib_levels['1.0']:
                                    reset_state_and_window()
                                    legs = []
                            # Back-check previous closed candle to avoid timing misses for second touch
                            try:
                                prev_row = cache_data.iloc[-2] if len(cache_data) >= 2 else None
                                if prev_row is not None and state.fib_levels:
                                    thr_705_bc = state.fib_levels.get('0.705')
                                    eps_bc = _touch_epsilon_price()
                                    # Bullish first touch (low side). For legacy second touch, status flip with low-touch is enough.
                                    # For opposite-side acceptance without status flip, require the candle SPANS 0.705 (both low and high cross the threshold with epsilon).
                                    if (state.last_touched_705_point_up is not None and
                                        prev_row['low'] <= (thr_705_bc + eps_bc) and
                                        prev_row['status'] != state.last_touched_705_point_up['status'] and
                                        not getattr(state, 'last_second_touch_705_point_up', None)):
                                        log(f'BACKCHECK Second touch 705 (bullish) at {prev_row.name} price={prev_row["low"]} thr={thr_705_bc} eps={eps_bc}', color='green')
                                        state.true_position = True
                                        state.last_second_touch_705_point_up = prev_row
                                    if (state.last_touched_705_point_down is not None and
                                        prev_row['high'] >= (thr_705_bc - eps_bc) and
                                        prev_row['status'] != state.last_touched_705_point_down['status'] and
                                        not getattr(state, 'last_second_touch_705_point_down', None)):
                                        log(f'BACKCHECK Second touch 705 (bearish) at {prev_row.name} price={prev_row["high"]} thr={thr_705_bc} eps={eps_bc}', color='green')
                                        state.true_position = True
                                        state.last_second_touch_705_point_down = prev_row
                                    # Opposite-side acceptance: require both sides touched (span) to ensure 0.705 truly crossed within the candle
                                    if state.last_touched_705_point_up is not None and not getattr(state, 'last_second_touch_705_point_up', None):
                                        span_low = prev_row['low'] <= (thr_705_bc + eps_bc)
                                        span_high = prev_row['high'] >= (thr_705_bc - eps_bc)
                                        if span_low and span_high:
                                            log(f'BACKCHECK BOTH-TOUCH second (bullish) via span at {prev_row.name} low={prev_row["low"]} high={prev_row["high"]} thr={thr_705_bc} eps={eps_bc}', color='green')
                                            state.true_position = True
                                            state.last_second_touch_705_point_up = prev_row
                                    if state.last_touched_705_point_down is not None and not getattr(state, 'last_second_touch_705_point_down', None):
                                        span_low = prev_row['low'] <= (thr_705_bc + eps_bc)
                                        span_high = prev_row['high'] >= (thr_705_bc - eps_bc)
                                        if span_low and span_high:
                                            log(f'BACKCHECK BOTH-TOUCH second (bearish) via span at {prev_row.name} low={prev_row["low"]} high={prev_row["high"]} thr={thr_705_bc} eps={eps_bc}', color='green')
                                            state.true_position = True
                                            state.last_second_touch_705_point_down = prev_row
                            except Exception:
                                pass

                        # فاز جدید: اگر جهت swing عوض شد، فیب قبلی حذف و تلاش برای ساخت فیب جدید (بدون انتظار برای نقض 1.0)
                        elif is_swing and state.fib_levels and last_swing_type != swing_type:
                            row = cache_data.iloc[-1]
                            log(f'Opposite swing detected -> reset fib to new swing if gated', color='orange')
                            state.reset()
                            # تلاش برای ساخت فیب جدید از swing تازه
                            new_fib = _init_fib_from_swing(swing_type, legs, row)
                            if new_fib:
                                last_swing_type = swing_type
                                state.fib_levels = new_fib
                                # touches already cleared by state.reset(); ensure position flag reset
                                state.true_position = False
                                fib0_point = cache_data.index.tolist().index(row.name)
                                fib_index = row.name
                                last_leg1_value = cache_data.index.tolist().index(legs[1]['end']) if len(legs) >= 2 else None
                                legs = legs[-2:]
                                # Reset counter on re-initialization due to opposite swing
                                f = 0
                                log(f'Fib re-initialized for opposite swing -> {swing_type} | {row.name}', color='green')
                                log(f'fib_levels: {state.fib_levels}', color='yellow')
                                log(f'fib_index: {fib_index}', color='yellow')

                        elif is_swing == False and state.fib_levels:
                            # خارج از حالت swing هم در صورت ثبت سقف/کف جدید 0.0 آپدیت شود (اگر leg1 موجود باشد)
                            row = cache_data.iloc[-1]
                            # برای آپدیت 0.0 از لنگر قفل‌شدهٔ 1.0 استفاده می‌کنیم
                            end_price_ref = state.fib1_price
                            maybe_new = _update_fib0_if_extends(last_swing_type or swing_type, state.fib_levels, row, end_price_ref)
                            if maybe_new is not state.fib_levels:
                                state.fib_levels = maybe_new
                                # reset first-touch state on 0.0 update
                                state.last_touched_705_point_up = None
                                state.last_touched_705_point_down = None
                                state.true_position = False
                                state.fib0_last_update_time = row.name
                                # Count extension outside swing as well (same swing context)
                                f += 1
                                try:
                                    log(f'Fib 0.0 updated (extend, out-of-swing) at {row.name}', color='green')
                                except Exception:
                                    pass
                            # لمس‌ها و گارد 1.0 مانند بالا
                            if last_swing_type == 'bullish' or swing_type == 'bullish':
                                thr_705 = state.fib_levels.get('0.705', float('inf'))
                                eps = _touch_epsilon_price()
                                low_touch = row['low'] <= (thr_705 + eps)
                                high_touch = row['high'] >= (thr_705 - eps)
                                if low_touch:
                                    if state.last_touched_705_point_up is None:
                                        log(f'first touch 705 point at {row.name} price={row["low"]}', color='green')
                                        state.last_touched_705_point_up = row
                                    elif row['status'] != state.last_touched_705_point_up['status']:
                                        log(f'Second touch 705 point at {row.name} price={row["low"]}', color='green')
                                        state.true_position = True
                                        state.last_second_touch_705_point_up = row
                                    else:
                                        try:
                                            if high_touch and not getattr(state, 'last_second_touch_705_point_up', None):
                                                log(f'BOTH-TOUCH: registering 2nd touch (bullish) at {row.name} via opposite side. low={row["low"]} high={row["high"]} thr={thr_705} eps={eps}', color='green')
                                                state.true_position = True
                                                state.last_second_touch_705_point_up = row
                                            else:
                                                delta = (row['low'] - thr_705)
                                                log(f'705 touched again but same status (no 2nd touch). prev={state.last_touched_705_point_up["status"]} cur={row["status"]} low={row["low"]} thr={thr_705} eps={eps} delta={delta}', color='yellow')
                                        except Exception:
                                            pass
                                elif state.fib_levels and row['low'] < state.fib_levels.get('1.0', -float('inf')):
                                    reset_state_and_window()
                                    legs = []
                            if last_swing_type == 'bearish' or swing_type == 'bearish':
                                thr_705 = state.fib_levels.get('0.705', -float('inf'))
                                eps = _touch_epsilon_price()
                                high_touch = row['high'] >= (thr_705 - eps)
                                low_touch = row['low'] <= (thr_705 + eps)
                                if high_touch:
                                    if state.last_touched_705_point_down is None:
                                        log(f'first touch 705 point code:6328455', color='red')
                                        state.last_touched_705_point_down = row
                                    elif row['status'] != state.last_touched_705_point_down['status']:
                                        log(f'Second touch 705 point code:6228455 {row.name}', color='green')
                                        state.true_position = True
                                        state.last_second_touch_705_point_down = row
                                    else:
                                        try:
                                            if low_touch and not getattr(state, 'last_second_touch_705_point_down', None):
                                                log(f'BOTH-TOUCH: registering 2nd touch (bearish) at {row.name} via opposite side. low={row["low"]} high={row["high"]} thr={thr_705} eps={eps}', color='green')
                                                state.true_position = True
                                                state.last_second_touch_705_point_down = row
                                            else:
                                                delta = (row['high'] - thr_705)
                                                log(f'705 touched again but same status (no 2nd touch). prev={state.last_touched_705_point_down["status"]} cur={row["status"]} high={row["high"]} thr={thr_705} eps={eps} delta={delta}', color='yellow')
                                        except Exception:
                                            pass
                                elif state.fib_levels and row['high'] > state.fib_levels.get('1.0', float('inf')):
                                    reset_state_and_window()
                                    legs = []
                            # Back-check previous closed candle
                            try:
                                prev_row = cache_data.iloc[-2] if len(cache_data) >= 2 else None
                                if prev_row is not None and state.fib_levels:
                                    thr_705_bc = state.fib_levels.get('0.705')
                                    eps_bc = _touch_epsilon_price()
                                    # Bullish first touch (low side): accept second touch if prev bar hit either:
                                    # - low side with status flip (legacy), OR
                                    # - high side (opposite side) only if the candle spans across 0.705 (both low and high touch)
                                    if state.last_touched_705_point_up is not None and not getattr(state, 'last_second_touch_705_point_up', None):
                                        has_low = prev_row['low'] <= (thr_705_bc + eps_bc)
                                        has_high = prev_row['high'] >= (thr_705_bc - eps_bc)
                                        if has_low and prev_row['status'] != state.last_touched_705_point_up['status']:
                                            log(f'BACKCHECK Second touch 705 (bullish) at {prev_row.name} price={prev_row["low"]} thr={thr_705_bc} eps={eps_bc}', color='green')
                                            state.true_position = True
                                            state.last_second_touch_705_point_up = prev_row
                                        elif has_high and has_low:
                                            log(f'BACKCHECK BOTH-TOUCH second (bullish) via span at {prev_row.name} low={prev_row["low"]} high={prev_row["high"]} thr={thr_705_bc} eps={eps_bc}', color='green')
                                            state.true_position = True
                                            state.last_second_touch_705_point_up = prev_row
                                    # Bearish first touch (high side): accept second touch if prev bar hit either:
                                    # - high side with status flip (legacy), OR
                                    # - low side (opposite side) only if the candle spans across 0.705 (both low and high touch)
                                    if state.last_touched_705_point_down is not None and not getattr(state, 'last_second_touch_705_point_down', None):
                                        has_high = prev_row['high'] >= (thr_705_bc - eps_bc)
                                        has_low = prev_row['low'] <= (thr_705_bc + eps_bc)
                                        if has_high and prev_row['status'] != state.last_touched_705_point_down['status']:
                                            log(f'BACKCHECK Second touch 705 (bearish) at {prev_row.name} price={prev_row["high"]} thr={thr_705_bc} eps={eps_bc}', color='green')
                                            state.true_position = True
                                            state.last_second_touch_705_point_down = prev_row
                                        elif has_low and has_high:
                                            log(f'BACKCHECK BOTH-TOUCH second (bearish) via span at {prev_row.name} low={prev_row["low"]} high={prev_row["high"]} thr={thr_705_bc} eps={eps_bc}', color='green')
                                            state.true_position = True
                                            state.last_second_touch_705_point_down = prev_row
                            except Exception:
                                pass

                elif len(legs) < 3:
                    # این بخش قبلاً از end_price نامشخص استفاده می‌کرد؛
                    # اکنون صرفاً لاگ ساختار و لمس‌ها را نگه می‌داریم و آپدیت 0.0 را به مسیرهای بالا سپردیم.
                    if state.fib_levels:
                        row = cache_data.iloc[-1]
                        if last_swing_type == 'bullish' or swing_type == 'bullish':
                            thr_705 = state.fib_levels.get('0.705', float('inf'))
                            eps = _touch_epsilon_price()
                            low_touch = row['low'] <= (thr_705 + eps)
                            high_touch = row['high'] >= (thr_705 - eps)
                            if low_touch:
                                if state.last_touched_705_point_up is None:
                                    log(f'first touch 705 point at {row.name} price={row["low"]}', color='green')
                                    state.last_touched_705_point_up = row
                                elif row['status'] != state.last_touched_705_point_up['status']:
                                    log(f'Second touch 705 point at {row.name} price={row["low"]}', color='green')
                                    state.true_position = True
                                else:
                                    try:
                                        if high_touch and not getattr(state, 'last_second_touch_705_point_up', None):
                                            log(f'BOTH-TOUCH: registering 2nd touch (bullish) at {row.name} via opposite side. low={row["low"]} high={row["high"]} thr={thr_705} eps={eps}', color='green')
                                            state.true_position = True
                                            state.last_second_touch_705_point_up = row
                                        else:
                                            delta = (row['low'] - thr_705)
                                            log(f'705 touched again but same status (no 2nd touch). prev={state.last_touched_705_point_up["status"]} cur={row["status"]} low={row["low"]} thr={thr_705} eps={eps} delta={delta}', color='yellow')
                                    except Exception:
                                        pass
                        if last_swing_type == 'bearish' or swing_type == 'bearish':
                            thr_705 = state.fib_levels.get('0.705', -float('inf'))
                            eps = _touch_epsilon_price()
                            high_touch = row['high'] >= (thr_705 - eps)
                            low_touch = row['low'] <= (thr_705 + eps)
                            if high_touch:
                                if state.last_touched_705_point_down is None:
                                    log(f'first touch 705 point', color='red')
                                    state.last_touched_705_point_down = row
                                elif row['status'] != state.last_touched_705_point_down['status']:
                                    log(f'Second touch 705 point code:5128455 {row.name}', color='green')
                                    state.true_position = True
                                else:
                                    try:
                                        if low_touch and not getattr(state, 'last_second_touch_705_point_down', None):
                                            log(f'BOTH-TOUCH: registering 2nd touch (bearish) at {row.name} via opposite side. low={row["low"]} high={row["high"]} thr={thr_705} eps={eps}', color='green')
                                            state.true_position = True
                                            state.last_second_touch_705_point_down = row
                                        else:
                                            delta = (row['high'] - thr_705)
                                            log(f'705 touched again but same status (no 2nd touch). prev={state.last_touched_705_point_down["status"]} cur={row["status"]} high={row["high"]} thr={thr_705} eps={eps} delta={delta}', color='yellow')
                                    except Exception:
                                        pass
                        # Back-check previous closed candle
                        try:
                            prev_row = cache_data.iloc[-2] if len(cache_data) >= 2 else None
                            if prev_row is not None and state.fib_levels:
                                thr_705_bc = state.fib_levels.get('0.705')
                                eps_bc = _touch_epsilon_price()
                                if state.last_touched_705_point_up is not None and not getattr(state, 'last_second_touch_705_point_up', None):
                                    has_low = prev_row['low'] <= (thr_705_bc + eps_bc)
                                    has_high = prev_row['high'] >= (thr_705_bc - eps_bc)
                                    if has_low and prev_row['status'] != state.last_touched_705_point_up['status']:
                                        log(f'BACKCHECK Second touch 705 (bullish) at {prev_row.name} price={prev_row["low"]} thr={thr_705_bc} eps={eps_bc}', color='green')
                                        state.true_position = True
                                        state.last_second_touch_705_point_up = prev_row
                                    elif has_high and has_low:
                                        log(f'BACKCHECK BOTH-TOUCH second (bullish) via span at {prev_row.name} low={prev_row["low"]} high={prev_row["high"]} thr={thr_705_bc} eps={eps_bc}', color='green')
                                        state.true_position = True
                                        state.last_second_touch_705_point_up = prev_row
                                if state.last_touched_705_point_down is not None and not getattr(state, 'last_second_touch_705_point_down', None):
                                    has_high = prev_row['high'] >= (thr_705_bc - eps_bc)
                                    has_low = prev_row['low'] <= (thr_705_bc + eps_bc)
                                    if has_high and prev_row['status'] != state.last_touched_705_point_down['status']:
                                        log(f'BACKCHECK Second touch 705 (bearish) at {prev_row.name} price={prev_row["high"]} thr={thr_705_bc} eps={eps_bc}', color='green')
                                        state.true_position = True
                                        state.last_second_touch_705_point_down = prev_row
                                    elif has_low and has_high:
                                        log(f'BACKCHECK BOTH-TOUCH second (bearish) via span at {prev_row.name} low={prev_row["low"]} high={prev_row["high"]} thr={thr_705_bc} eps={eps_bc}', color='green')
                                        state.true_position = True
                                        state.last_second_touch_705_point_down = prev_row
                        except Exception:
                            pass
                    if len(legs) == 2:
                        log(f'leg0: {legs[0]["start"]}, {legs[0]["end"]}, leg1: {legs[1]["start"]}, {legs[1]["end"]}', color='lightcyan_ex')
                    if len(legs) == 1:
                        log(f'leg0: {legs[0]["start"]}, {legs[0]["end"]}', color='lightcyan_ex')
                
                # بخش معاملات - buy statement (مطابق منطق main_saver_copy2.py)
                if state.true_position and (last_swing_type == 'bullish' or swing_type == 'bullish'):
                    last_tick = mt5.symbol_info_tick(MT5_CONFIG['symbol'])
                    buy_entry_price = last_tick.ask
                    # لاگ سیگنال (قبل از ارسال سفارش)
                    try:
                        log_signal(
                            symbol=MT5_CONFIG['symbol'],
                            strategy="swing_fib_v1",
                            direction="buy",
                            rr=win_ratio,
                            entry=buy_entry_price,
                            sl=float(state.fib_levels['1.0'] if abs(state.fib_levels['0.9']-buy_entry_price) <= _pip_size_for(MT5_CONFIG['symbol'])*2 else state.fib_levels['0.9']),
                            tp=None,
                            fib=state.fib_levels,
                            confidence=None,
                            features_json=None,
                            note="triggered_by_pullback"
                        )
                    except Exception:
                        pass
                    # دریافت قیمت لحظه‌ای بازار از MT5
                    # current_open_point = cache_data.iloc[-1]['close']
                    log(f'Start long position income {cache_data.iloc[-1].name}', color='blue')
                    log(f'current_open_point (market ask): {buy_entry_price}', color='blue')
                    # ENTRY CONTEXT (BUY): fib snapshot + touches
                    try:
                        fib = state.fib_levels or {}
                        fib0_p = fib.get('0.0')
                        fib1_p = fib.get('1.0')
                        log(
                            f"ENTRY_CTX_BUY | fib0={fib0_p} t0={state.fib0_last_update_time} | fib705={fib.get('0.705')} | fib09={fib.get('0.9')} | fib1={fib1_p} t1={state.fib1_time}",
                            color='cyan'
                        )
                        if state.last_touched_705_point_up is not None:
                            lt = state.last_touched_705_point_up
                            log(f"ENTRY_CTX_BUY | first_touch_705_up t={lt.name} price={lt['low']}", color='cyan')
                        if hasattr(state, 'last_second_touch_705_point_up') and state.last_second_touch_705_point_up is not None:
                            stp = state.last_second_touch_705_point_up
                            log(f"ENTRY_CTX_BUY | second_touch_705_up t={stp.name} price={stp['low']}", color='cyan')
                    except Exception:
                        pass

                    pip_size = _pip_size_for(MT5_CONFIG['symbol'])
                    two_pips = 2.0 * pip_size
                    min_dist = _min_stop_distance(MT5_CONFIG['symbol'])

                    # معیار درستِ 2 پیپ
                    is_close_to_09 = abs(state.fib_levels['0.9'] - buy_entry_price) <= two_pips

                    candidate_sl = state.fib_levels['1.0'] if is_close_to_09 else state.fib_levels['0.9']

                    min_pip_dist = 2  # حداقل 2 پیپ واقعی
                    pip_size = _pip_size_for(MT5_CONFIG['symbol'])
                    min_abs_dist = max(min_pip_dist * pip_size, min_dist)

                    # گارد جهت
                    if candidate_sl >= buy_entry_price:
                        # برگرداندن به 1.0 اگر 0.9 بالاتر بود
                        candidate_sl = float(state.fib_levels['1.0'])
                    # اطمینان از فاصله
                    if (buy_entry_price - candidate_sl) < min_abs_dist:
                        # اگر فاصله خیلی کم است، یا SL را جابه‌جا کن یا معامله را لغو کن
                        adj = buy_entry_price - min_abs_dist
                        if adj <= 0:
                            log("🚫 Skip BUY: invalid SL distance", color='red')
                            state.reset()
                            reset_state_and_window()
                            continue
                        candidate_sl = float(adj)

                    stop = float(candidate_sl)
                    if stop >= buy_entry_price:
                        log("🚫 Skip BUY: SL still >= entry after adjust", color='red')
                        state.reset()
                        reset_state_and_window()
                        continue

                    stop_distance = abs(buy_entry_price - stop)
                    reward_end = buy_entry_price + (stop_distance * win_ratio)
                    log(f'stop = {stop}', color='green')
                    log(f'reward_end = {reward_end}', color='green')

                    # ارسال سفارش BUY با هر stop و reward
                    result = mt5_conn.open_buy_position(
                        tick=last_tick,
                        sl=stop,
                        tp=reward_end,
                        comment=f"Bullish Swing {swing_type}",
                        risk_pct=0.01  # مثلا 1% ریسک
                    )
                    # ارسال ایمیل غیرمسدودکننده
                    try:
                        send_trade_email_async(
                            subject=f"NEW BUY ORDER {MT5_CONFIG['symbol']}",
                            body=(
                                f"Time: {datetime.now()}\n"
                                f"Symbol: {MT5_CONFIG['symbol']}\n"
                                f"Type: BUY (Bullish Swing)\n"
                                f"Entry: {buy_entry_price}\n"
                                f"SL: {stop}\n"
                                f"TP: {reward_end}\n"
                            )
                        )
                    except Exception as _e:
                        log(f'Email dispatch failed: {_e}', color='red')

                    if result and getattr(result, 'retcode', None) == 10009:
                        log(f'✅ BUY order executed successfully', color='green')
                        log(f'📊 Ticket={result.order} Price={result.price} Volume={result.volume}', color='cyan')
                        # ارسال ایمیل غیرمسدودکننده
                        try:
                            send_trade_email_async(
                                subject = f"Last order result",
                                body=(
                                    f"Ticket={result.order}\n"
                                    f"Price={result.price}\n"
                                    f"Volume={result.volume}\n"
                                )
                            )
                        except Exception as _e:
                            log(f'Email dispatch failed: {_e}', color='red')
                    else:
                        if result:
                            log(f'❌ BUY failed retcode={result.retcode} comment={result.comment}', color='red')
                        else:
                            log(f'❌ BUY failed (no result object)', color='red')
                    state.reset()

                    reset_state_and_window()
                    legs = []

                # بخش معاملات - sell statement (مطابق منطق main_saver_copy2.py)
                if state.true_position and (last_swing_type == 'bearish' or swing_type == 'bearish'):
                    last_tick = mt5.symbol_info_tick(MT5_CONFIG['symbol'])
                    sell_entry_price = last_tick.bid
                    try:
                        log_signal(
                            symbol=MT5_CONFIG['symbol'],
                            strategy="swing_fib_v1",
                            direction="sell",
                            rr=win_ratio,
                            entry=sell_entry_price,
                            sl=float(state.fib_levels['1.0'] if abs(state.fib_levels['0.9']-sell_entry_price) <= _pip_size_for(MT5_CONFIG['symbol'])*2 else state.fib_levels['0.9']),
                            tp=None,
                            fib=state.fib_levels,
                            confidence=None,
                            features_json=None,
                            note="triggered_by_pullback"
                        )
                    except Exception:
                        pass
                    log(f'Start short position income {cache_data.iloc[-1].name}', color='red')
                    log(f'current_open_point (market bid): {sell_entry_price}', color='red')
                    # ENTRY CONTEXT (SELL): fib snapshot + touches
                    try:
                        fib = state.fib_levels or {}
                        fib0_p = fib.get('0.0')
                        fib1_p = fib.get('1.0')
                        log(
                            f"ENTRY_CTX_SELL | fib0={fib0_p} t0={state.fib0_last_update_time} | fib705={fib.get('0.705')} | fib09={fib.get('0.9')} | fib1={fib1_p} t1={state.fib1_time}",
                            color='cyan'
                        )
                        if state.last_touched_705_point_down is not None:
                            lt = state.last_touched_705_point_down
                            log(f"ENTRY_CTX_SELL | first_touch_705_down t={lt.name} price={lt['high']}", color='cyan')
                        if hasattr(state, 'last_second_touch_705_point_down') and state.last_second_touch_705_point_down is not None:
                            stp = state.last_second_touch_705_point_down
                            log(f"ENTRY_CTX_SELL | second_touch_705_down t={stp.name} price={stp['high']}", color='cyan')
                    except Exception:
                        pass

                    pip_size = _pip_size_for(MT5_CONFIG['symbol'])
                    two_pips = 2.0 * pip_size
                    min_dist = _min_stop_distance(MT5_CONFIG['symbol'])

                    is_close_to_09 = abs(state.fib_levels['0.9'] - sell_entry_price) <= two_pips
                    candidate_sl = state.fib_levels['1.0'] if is_close_to_09 else state.fib_levels['0.9']

                    min_pip_dist = 2.0
                    pip_size = _pip_size_for(MT5_CONFIG['symbol'])
                    min_abs_dist = max(min_pip_dist * pip_size, min_dist)

                    if candidate_sl <= sell_entry_price:
                        candidate_sl = float(state.fib_levels['1.0'])
                    if (candidate_sl - sell_entry_price) < min_abs_dist:
                        adj = sell_entry_price + min_abs_dist
                        candidate_sl = float(adj)

                    stop = float(candidate_sl)
                    if stop <= sell_entry_price:
                        log("🚫 Skip SELL: SL still <= entry after adjust", color='red')
                        state.reset()
                        reset_state_and_window()
                        continue

                    stop_distance = abs(sell_entry_price - stop)
                    reward_end = sell_entry_price - (stop_distance * win_ratio)
                    log(f'stop = {stop}', color='red')
                    log(f'reward_end = {reward_end}', color='red')

                    # ارسال سفارش SELL با هر stop و reward
                    result = mt5_conn.open_sell_position(
                        tick=last_tick,
                        sl=stop,
                        tp=reward_end,
                        comment=f"Bearish Swing {swing_type}",
                        risk_pct=0.01  # مثلا 1% ریسک
                    )
                    
                    # ارسال ایمیل غیرمسدودکننده
                    try:
                        send_trade_email_async(
                            subject=f"NEW SELL ORDER {MT5_CONFIG['symbol']}",
                            body=(
                                f"Time: {datetime.now()}\n"
                                f"Symbol: {MT5_CONFIG['symbol']}\n"
                                f"Type: SELL (Bearish Swing)\n"
                                f"Entry: {sell_entry_price}\n"
                                f"SL: {stop}\n"
                                f"TP: {reward_end}\n"
                            )
                        )
                    except Exception as _e:
                        log(f'Email dispatch failed: {_e}', color='red')
                    
                    if result and getattr(result, 'retcode', None) == 10009:
                        log(f'✅ SELL order executed successfully', color='green')
                        log(f'📊 Ticket={result.order} Price={result.price} Volume={result.volume}', color='cyan')
                        # ارسال ایمیل غیرمسدودکننده
                        try:
                            send_trade_email_async(
                                subject = f"Last order result",
                                body=(
                                    f"Ticket={result.order}\n"
                                    f"Price={result.price}\n"
                                    f"Volume={result.volume}\n"
                                )
                            )
                        except Exception as _e:
                            log(f'Email dispatch failed: {_e}', color='red')
                    else:
                        if result:
                            log(f'❌ SELL failed retcode={result.retcode} comment={result.comment}', color='red')
                        else:
                            log(f'❌ SELL failed (no result object)', color='red')
                    state.reset()

                    reset_state_and_window()
                    legs = []
                
                log(f'cache_data.iloc[-1].name: {cache_data.iloc[-1].name}', color='lightblue_ex')
                log(f'len(legs): {len(legs)} | start_index: {start_index} | {cache_data.iloc[start_index].name}', color='lightred_ex')
                log(f' ' * 80)
                log(f'-'* 80)
                log(f' ' * 80)

                # ذخیره آخرین زمان داده
                # last_data_time = cache_data.index[-1]  # این خط حذف شد چون بالا انجام شد

            # بررسی وضعیت پوزیشن‌های باز
            positions = mt5_conn.get_positions()
            if positions is None or len(positions) == 0:
                if position_open:
                    log("🏁 Position closed", color='yellow')
                    position_open = False

            manage_open_positions()

            sleep(0.5)  # مطابق main_saver_copy2.py

        except KeyboardInterrupt:
            log("🛑 Bot stopped by user", color='yellow')
            mt5_conn.close_all_positions()
            break
        except Exception as e:
            log(f' ' * 80)
            log(f"❌ Error: {e}", color='red')
            sleep(5)

    mt5_conn.shutdown()
    print("🔌 MT5 connection closed")

def _pip_size_for(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    if not info:
        return 0.0001
    # برای 5/3 رقمی: 1 pip = 10 * point
    return info.point * (10.0 if info.digits in (3, 5) else 1.0)

def _min_stop_distance(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    if not info:
        return 0.0003
    point = info.point
    # حداقل فاصله مجاز بروکر (stops_level) یا 3 پوینت به‌عنوان فfallback
    return max((getattr(info, 'trade_stops_level', 0) or 0) * point, 3 * point)

if __name__ == "__main__":
    main()
