import pandas as pd

# فرض کنید data_list یک لیست از دیکشنری‌ها است که داده‌های شما را ذخیره می‌کند
data_list = []

# در هر بار که معامله‌ای انجام می‌شود، داده‌ها را به data_list اضافه کنید
data_list.append({
    'timestamp': datetime.now(),
    'open_price': open_price,
    'close_price': close_price,
    'high_price': high_price,
    'low_price': low_price,
    'profit': profit,
    'trade_signal': trade_signal,
    # سایر داده‌های مورد نظر
})

# در پایان روز یا در زمان مشخص، داده‌ها را در یک فایل CSV ذخیره کنید
df = pd.DataFrame(data_list)
df.to_csv('trading_data.csv', index=False)