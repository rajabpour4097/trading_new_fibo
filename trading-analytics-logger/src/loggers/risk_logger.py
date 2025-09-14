import pandas as pd
from datetime import datetime

# لیست برای ذخیره اطلاعات
trade_data = []

def log_trade(trade_info):
    trade_data.append(trade_info)

def save_to_csv():
    df = pd.DataFrame(trade_data)
    df.to_csv('trading_data.csv', index=False)

# در طول اجرای ربات، هر بار که معامله‌ای انجام می‌شود
trade_info = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'trade_type': 'buy',  # یا 'sell'
    'volume': 0.01,
    'entry_price': 1.2000,
    'exit_price': 1.2050,
    'profit_loss': 50.0,  # مقدار سود یا زیان
    'market_conditions': 'bullish',  # شرایط بازار
    'signal': 'Fibonacci Level',  # سیگنال ورودی
}

log_trade(trade_info)

# در انتهای برنامه یا در زمان مناسب
save_to_csv()