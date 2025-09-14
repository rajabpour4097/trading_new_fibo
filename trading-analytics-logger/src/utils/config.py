import pandas as pd
from datetime import datetime

# لیست برای ذخیره داده‌ها
data = []

# تابعی برای ذخیره‌سازی داده‌ها
def log_trade_data(symbol, entry_price, exit_price, profit, trade_time):
    data.append({
        'timestamp': trade_time,
        'symbol': symbol,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'profit': profit
    })

# در زمان باز کردن یا بستن معامله، تابع log_trade_data را فراخوانی کنید
# به عنوان مثال:
log_trade_data('EURUSD', 1.2000, 1.2050, 50, datetime.now())

# در پایان روز یا در زمان مناسب، داده‌ها را در فایل CSV ذخیره کنید
df = pd.DataFrame(data)
df.to_csv('trading_data.csv', index=False)