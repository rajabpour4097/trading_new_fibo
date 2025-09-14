import pandas as pd

# فرض کنید data_list یک لیست از دیکشنری‌ها است که داده‌ها را ذخیره می‌کند
data_list = []

# در هر بار جمع‌آوری داده، یک دیکشنری جدید به data_list اضافه کنید
data_list.append({
    'timestamp': timestamp,
    'open': open_price,
    'high': high_price,
    'low': low_price,
    'close': close_price,
    'profit': profit,
    'signal': signal_type,
    # سایر داده‌های مورد نیاز
})

# در پایان، داده‌ها را به یک DataFrame تبدیل کرده و در فایل CSV ذخیره کنید
df = pd.DataFrame(data_list)
df.to_csv('trading_data.csv', index=False)