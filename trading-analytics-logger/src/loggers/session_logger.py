import pandas as pd
from datetime import datetime

# ایجاد یک DataFrame خالی
columns = ['timestamp', 'trade_type', 'entry_price', 'exit_price', 'volume', 'profit', 'market_conditions']
trades_data = pd.DataFrame(columns=columns)

def log_trade(trade_type, entry_price, exit_price, volume, profit, market_conditions):
    global trades_data
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_trade = {
        'timestamp': timestamp,
        'trade_type': trade_type,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'volume': volume,
        'profit': profit,
        'market_conditions': market_conditions
    }
    trades_data = trades_data.append(new_trade, ignore_index=True)

def save_to_csv(filename='trades_data.csv'):
    trades_data.to_csv(filename, index=False)

# مثال از نحوه استفاده
log_trade('buy', 1.2000, 1.2050, 0.01, 50, 'bullish')
log_trade('sell', 1.2100, 1.2050, 0.01, -50, 'bearish')

# ذخیره‌سازی داده‌ها در فایل CSV
save_to_csv()