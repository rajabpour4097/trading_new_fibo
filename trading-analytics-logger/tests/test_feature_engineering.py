import pandas as pd
from datetime import datetime

# ایجاد یک DataFrame برای ذخیره داده‌ها
data = {
    'timestamp': [],
    'trade_type': [],
    'entry_price': [],
    'exit_price': [],
    'volume': [],
    'profit': [],
    'market_conditions': [],
}

# تابع برای ذخیره داده‌ها
def log_trade(trade_type, entry_price, exit_price, volume, profit, market_conditions):
    global data
    data['timestamp'].append(datetime.now())
    data['trade_type'].append(trade_type)
    data['entry_price'].append(entry_price)
    data['exit_price'].append(exit_price)
    data['volume'].append(volume)
    data['profit'].append(profit)
    data['market_conditions'].append(market_conditions)

    # ذخیره داده‌ها در فایل CSV
    df = pd.DataFrame(data)
    df.to_csv('trading_data.csv', index=False)

# مثال از نحوه استفاده از تابع log_trade
log_trade('buy', 1.2000, 1.2050, 0.01, 50, 'bullish')