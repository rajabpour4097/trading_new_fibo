from datetime import datetime, timedelta
import pytz
import MetaTrader5 as mt5

# اتصال
mt5.initialize()

# زمان UTC (از خروجی قبلی)
from_time = datetime(2025, 8, 7, 9, 28, tzinfo=pytz.UTC)
to_time = from_time + timedelta(minutes=2)

# دریافت معاملات
deals = mt5.history_deals_get(from_time, to_time)

if deals:
    print(f"✅ معاملات یافت شده: {len(deals)}")
    for d in deals:
        print(f"{datetime.fromtimestamp(d.time, pytz.UTC)} | Type: {d.type} | Price: {d.price} | Profit: {d.profit}")
else:
    print("❌ هیچ معامله‌ای در بازه زمانی یافت نشد.")
