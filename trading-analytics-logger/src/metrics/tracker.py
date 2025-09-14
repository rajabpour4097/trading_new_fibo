import pandas as pd
from datetime import datetime

# تابعی برای ذخیره‌سازی داده‌ها
def save_trade_data(trade_data):
    # تبدیل داده‌ها به DataFrame
    df = pd.DataFrame(trade_data)
    
    # ذخیره‌سازی در فایل CSV
    df.to_csv('trade_data.csv', mode='a', header=False, index=False)

# نمونه‌ای از داده‌های معاملاتی
trade_data = {
    'timestamp': datetime.now(),
    'symbol': 'EURUSD',
    'action': 'buy',
    'price': 1.12345,
    'volume': 0.01,
    'profit': 10.0,
    'market_conditions': 'bullish',
}

# ذخیره‌سازی داده‌ها
save_trade_data(trade_data)