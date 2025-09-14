import pandas as pd

# فرض کنید که شما یک لیست از داده‌ها دارید
data = []

# در هر بار که یک معامله انجام می‌شود، اطلاعات مربوطه را به لیست اضافه کنید
data.append({
    'timestamp': datetime.now(),
    'symbol': 'EURUSD',
    'action': 'buy',
    'price': 1.12345,
    'volume': 0.01,
    'profit': 10.0,
    'market_condition': 'bullish'
})

# در پایان روز یا در زمان‌های مشخص، داده‌ها را در یک فایل CSV ذخیره کنید
df = pd.DataFrame(data)
df.to_csv('trading_data.csv', index=False)