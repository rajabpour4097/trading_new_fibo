import pandas as pd
from datetime import datetime

# ایجاد یک DataFrame برای ذخیره داده‌ها
data_columns = ['timestamp', 'open', 'high', 'low', 'close', 'profit', 'trade_type']
data = pd.DataFrame(columns=data_columns)

# در حلقه اصلی ربات، داده‌ها را جمع‌آوری کنید
while True:
    # جمع‌آوری داده‌های بازار
    live_data = get_live_data()  # فرض کنید این تابع داده‌های زنده را برمی‌گرداند
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # فرض کنید شما یک معامله انجام داده‌اید و سود آن را محاسبه کرده‌اید
    profit = calculate_profit()  # تابعی برای محاسبه سود
    trade_type = 'BUY'  # یا 'SELL' بسته به نوع معامله

    # اضافه کردن داده‌ها به DataFrame
    data = data.append({
        'timestamp': current_time,
        'open': live_data['open'],
        'high': live_data['high'],
        'low': live_data['low'],
        'close': live_data['close'],
        'profit': profit,
        'trade_type': trade_type
    }, ignore_index=True)

    # ذخیره داده‌ها در فایل CSV
    data.to_csv('trading_data.csv', index=False)