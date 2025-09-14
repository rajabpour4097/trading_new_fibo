import pandas as pd

# فرض کنید data_list یک لیست از دیکشنری‌ها است که داده‌های شما را شامل می‌شود
data_list = []

# در هر بار معامله یا جمع‌آوری داده، داده‌ها را به data_list اضافه کنید
data_list.append({
    'timestamp': timestamp,
    'open': open_price,
    'high': high_price,
    'low': low_price,
    'close': close_price,
    'profit': profit,
    'trade_signal': trade_signal,
    # سایر داده‌های مورد نیاز
})

# در پایان روز یا در زمان‌های مشخص، داده‌ها را در فایل CSV ذخیره کنید
df = pd.DataFrame(data_list)
df.to_csv('trading_data.csv', index=False)