import pandas as pd
from datetime import datetime

# لیست برای ذخیره داده‌ها
data = []

def log_trade(action, price, volume, profit, timestamp):
    """ثبت اطلاعات معامله"""
    data.append({
        'timestamp': timestamp,
        'action': action,
        'price': price,
        'volume': volume,
        'profit': profit
    })

def save_to_csv(filename='trades.csv'):
    """ذخیره داده‌ها در فایل CSV"""
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

# مثال استفاده
log_trade('buy', 1.2000, 0.01, 10.0, datetime.now())
log_trade('sell', 1.2050, 0.01, -5.0, datetime.now())

# ذخیره‌سازی داده‌ها در فایل
save_to_csv()