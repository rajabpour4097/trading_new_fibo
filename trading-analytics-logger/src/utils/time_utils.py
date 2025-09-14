import pandas as pd
from datetime import datetime

# تابع برای ذخیره‌سازی داده‌ها
def save_trade_data(trade_data):
    # تبدیل داده‌ها به DataFrame
    df = pd.DataFrame(trade_data)
    
    # ذخیره‌سازی در فایل CSV
    df.to_csv('trade_data.csv', mode='a', header=False, index=False)

# مثال از داده‌های جمع‌آوری شده
trade_data = {
    'timestamp': datetime.now(),
    'symbol': 'EURUSD',
    'action': 'buy',
    'entry_price': 1.2000,
    'exit_price': 1.2050,
    'volume': 0.01,
    'profit': 5.0,
    'market_conditions': 'bullish'
}

# ذخیره‌سازی داده‌ها
save_trade_data(trade_data)