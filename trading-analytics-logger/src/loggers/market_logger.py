import pandas as pd
from datetime import datetime

# لیست برای ذخیره داده‌ها
data = []

def log_trade(symbol, volume, trade_type, entry_price, exit_price, profit):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data.append({
        'timestamp': timestamp,
        'symbol': symbol,
        'volume': volume,
        'trade_type': trade_type,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'profit': profit
    })

# در انتهای برنامه یا در زمان مناسب، داده‌ها را در فایل CSV ذخیره کنید
def save_to_csv():
    df = pd.DataFrame(data)
    df.to_csv('trading_data.csv', index=False)

# مثال استفاده از تابع log_trade
log_trade('EURUSD', 0.01, 'buy', 1.2000, 1.2050, 50)

# ذخیره‌سازی داده‌ها در فایل CSV
save_to_csv()