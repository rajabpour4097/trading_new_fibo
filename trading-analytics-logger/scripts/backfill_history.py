import pandas as pd

# فرض کنید داده‌ها در یک لیست از دیکشنری‌ها ذخیره شده‌اند
data = []

# در طول زمان، داده‌ها را جمع‌آوری کنید
data.append({
    'timestamp': '2023-10-01 10:00:00',
    'action': 'buy',
    'price': 1.2000,
    'volume': 0.01,
    'profit': 10.0,
    'market_conditions': 'bullish',
    'signal': 'buy_signal'
})

# در پایان، داده‌ها را به یک DataFrame تبدیل کنید
df = pd.DataFrame(data)

# ذخیره‌سازی در فایل CSV
df.to_csv('trading_data.csv', index=False)