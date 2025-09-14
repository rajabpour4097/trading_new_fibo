import pandas as pd

# فرض کنید که این داده‌ها را جمع‌آوری کرده‌اید
data = {
    'timestamp': [],
    'trade_type': [],
    'entry_price': [],
    'exit_price': [],
    'volume': [],
    'profit_loss': [],
    'market_conditions': [],
}

# بعد از هر معامله، داده‌ها را به لیست‌ها اضافه کنید
data['timestamp'].append(datetime.now())
data['trade_type'].append('Buy')  # یا 'Sell'
data['entry_price'].append(entry_price)
data['exit_price'].append(exit_price)
data['volume'].append(volume)
data['profit_loss'].append(profit_loss)
data['market_conditions'].append(market_conditions)

# در نهایت، داده‌ها را به DataFrame تبدیل کرده و در فایل CSV ذخیره کنید
df = pd.DataFrame(data)
df.to_csv('trading_data.csv', index=False)