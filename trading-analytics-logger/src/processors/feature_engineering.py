import pandas as pd

# فرض کنید data_list یک لیست از دیکشنری‌ها است که داده‌های شما را ذخیره می‌کند
data_list = []

# در طول زمان داده‌ها را جمع‌آوری کنید
data_list.append({
    'timestamp': timestamp,
    'open': open_price,
    'high': high_price,
    'low': low_price,
    'close': close_price,
    'trade_type': trade_type,
    'profit': profit,
    'fibonacci_level': fibonacci_level,
    # سایر داده‌های مورد نیاز
})

# در پایان هر دوره، داده‌ها را به CSV اضافه کنید
df = pd.DataFrame(data_list)
df.to_csv('trading_data.csv', index=False)