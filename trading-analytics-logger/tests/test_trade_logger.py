import pandas as pd
from datetime import datetime

# لیست برای ذخیره داده‌ها
data = []

def log_trade(price, volume, trade_type, profit_loss, timestamp):
    data.append({
        'timestamp': timestamp,
        'price': price,
        'volume': volume,
        'trade_type': trade_type,
        'profit_loss': profit_loss
    })

# در هر بار خرید یا فروش
def execute_trade(price, volume, trade_type):
    # کد برای اجرای معامله
    profit_loss = calculate_profit_loss()  # تابعی برای محاسبه سود و زیان
    timestamp = datetime.now()
    log_trade(price, volume, trade_type, profit_loss, timestamp)

# ذخیره‌سازی داده‌ها در فایل CSV
def save_to_csv():
    df = pd.DataFrame(data)
    df.to_csv('trading_data.csv', index=False)

# در پایان برنامه یا در زمان‌های مشخص
save_to_csv()