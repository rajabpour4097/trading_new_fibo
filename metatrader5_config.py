# ساعات مختلف بازار فارکس بر اساس ساعت ایران

# جلسه سیدنی (05:30 - 14:30 ایران)
SYDNEY_HOURS_IRAN = {
    'start': '05:30',
    'end': '14:30'
}

# جلسه توکیو (07:30 - 16:30 ایران)  
TOKYO_HOURS_IRAN = {
    'start': '07:30',
    'end': '16:30'
}

# جلسه لندن (12:30 - 21:30 ایران)
LONDON_HOURS_IRAN = {
    'start': '12:30',
    'end': '21:30'
}

# جلسه نیویورک (17:30 - 02:30 ایران)
NEWYORK_HOURS_IRAN = {
    'start': '17:30',
    'end': '02:30'  # روز بعد
}

# همپوشانی لندن-نیویورک (17:30 - 21:30 ایران) - بهترین زمان
OVERLAP_LONDON_NY_IRAN = {
    'start': '17:30',
    'end': '21:30'
}

# ساعات فعال ایرانی (09:00 - 21:00)
IRAN_ACTIVE_HOURS = {
    'start': '09:00',
    'end': '21:00'
}

# 24 ساعته
FULL_TIME_IRAN = {
    'start': '00:00',
    'end': '23:59'
}

# تنظیمات MT5
MT5_CONFIG = {
    'symbol': 'EURUSD',
    'lot_size': 0.01,
    'win_ratio': 1.2,
    'magic_number': 234000,
    'deviation': 20,
    'max_spread': 3.0,
    'min_balance': 1,
    'max_daily_trades': 10,
    'trading_hours': FULL_TIME_IRAN,
}

# تنظیمات استراتژی
TRADING_CONFIG = {
    'threshold': 6,
    'fib_705': 0.705,
    'fib_90': 0.9,
    'window_size': 100,
    'min_swing_size': 4,
    'entry_tolerance': 2.0,
    'lookback_period': 20,
}

# مدیریت پویا چند مرحله‌ای جدید - 12 مرحله
# مراحل بر اساس درخواست:
# 1) 1.0R: SL روی +1.0R، TP به 1.5R
# 2) 1.5R: SL روی +1.5R، TP به 2.0R
# 3) 2.0R: SL روی +2.0R، TP به 2.5R
# 4) 2.5R: SL روی +2.5R، TP به 3.0R
# 5) 3.0R: SL روی +3.0R، TP به 3.5R
# 6) 3.5R: SL روی +3.5R، TP به 4.0R
# 7) 4.0R: SL روی +4.0R، TP به 4.5R
# 8) 4.5R: SL روی +4.5R، TP به 5.0R
# 9) 5.0R: SL روی +5.0R، TP به 5.5R
# 10) 5.5R: SL روی +5.5R، TP به 6.0R
# 11) 6.0R: SL روی +6.0R، TP به 6.5R
# 12) 6.5R: SL روی +6.5R، TP به 7.0R
DYNAMIC_RISK_CONFIG = {
    'enable': True,
    'commission_per_lot': 4.5,          # کمیسیون کل (رفت و برگشت یا فقط رفت؟ طبق بروکر - قابل تنظیم)
    'commission_mode': 'per_lot',       # per_lot (کل)، per_side (نیمی از رفت و برگشت) در صورت نیاز توسعه
    'round_trip': False,                # اگر True و per_side باشد دو برابر می‌کند
    'base_tp_R': 1.2,                   # TP اولیه تنظیم‌شده هنگام ورود (برای مرجع)
    'stages': [
        {  # 1.0R stage
            'id': 'stage_1_0R',
            'trigger_R': 1.0,
            'sl_lock_R': 1.0,
            'tp_R': 1.5
        },
        {  # 1.5R stage
            'id': 'stage_1_5R',
            'trigger_R': 1.5,
            'sl_lock_R': 1.5,
            'tp_R': 2.0
        },
        {  # 2.0R stage
            'id': 'stage_2_0R',
            'trigger_R': 2.0,
            'sl_lock_R': 2.0,
            'tp_R': 2.5
        },
        {  # 2.5R stage
            'id': 'stage_2_5R',
            'trigger_R': 2.5,
            'sl_lock_R': 2.5,
            'tp_R': 3.0
        },
        {  # 3.0R stage
            'id': 'stage_3_0R',
            'trigger_R': 3.0,
            'sl_lock_R': 3.0,
            'tp_R': 3.5
        },
        {  # 3.5R stage
            'id': 'stage_3_5R',
            'trigger_R': 3.5,
            'sl_lock_R': 3.5,
            'tp_R': 4.0
        },
        {  # 4.0R stage
            'id': 'stage_4_0R',
            'trigger_R': 4.0,
            'sl_lock_R': 4.0,
            'tp_R': 4.5
        },
        {  # 4.5R stage
            'id': 'stage_4_5R',
            'trigger_R': 4.5,
            'sl_lock_R': 4.5,
            'tp_R': 5.0
        },
        {  # 5.0R stage
            'id': 'stage_5_0R',
            'trigger_R': 5.0,
            'sl_lock_R': 5.0,
            'tp_R': 5.5
        },
        {  # 5.5R stage
            'id': 'stage_5_5R',
            'trigger_R': 5.5,
            'sl_lock_R': 5.5,
            'tp_R': 6.0
        },
        {  # 6.0R stage
            'id': 'stage_6_0R',
            'trigger_R': 6.0,
            'sl_lock_R': 6.0,
            'tp_R': 6.5
        },
        {  # 6.5R stage
            'id': 'stage_6_5R',
            'trigger_R': 6.5,
            'sl_lock_R': 6.5,
            'tp_R': 7.0
        }
    ]
}

# تنظیمات لاگ
LOG_CONFIG = {
    'log_level': 'INFO',        # DEBUG, INFO, WARNING, ERROR
    'save_to_file': True,       # ذخیره در فایل
    'max_log_size': 10,         # حداکثر حجم فایل لاگ (MB)
}