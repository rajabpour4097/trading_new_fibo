import pandas as pd

# فرض کنید که شما یک لیست از داده‌ها دارید
data = {
    'timestamp': [],
    'open': [],
    'high': [],
    'low': [],
    'close': [],
    'volume': [],
    'profit': [],
    'trade_signal': [],
}

# در هر بار معامله، داده‌ها را به لیست اضافه کنید
data['timestamp'].append(current_time)
data['open'].append(open_price)
data['high'].append(high_price)
data['low'].append(low_price)
data['close'].append(close_price)
data['volume'].append(volume)
data['profit'].append(profit)
data['trade_signal'].append(signal)

# بعد از جمع‌آوری داده‌ها، آن‌ها را به DataFrame تبدیل کنید و در CSV ذخیره کنید
df = pd.DataFrame(data)
df.to_csv('trading_data.csv', index=False)