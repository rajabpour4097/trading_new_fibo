import pandas as pd
from datetime import datetime

# تابع برای ذخیره‌سازی داده‌ها
def save_trade_data(trade_data):
    df = pd.DataFrame(trade_data)
    df.to_csv('trade_data.csv', mode='a', header=False, index=False)

# نمونه‌ای از داده‌ها
trade_data = {
    'timestamp': datetime.now(),
    'symbol': 'EURUSD',
    'action': 'buy',
    'price': 1.12345,
    'volume': 0.01,
    'profit': 10.0,
    'market_conditions': 'bullish'
}

# ذخیره‌سازی داده‌ها
save_trade_data(trade_data)