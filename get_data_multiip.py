import requests
import random
from datetime import datetime
import pytz
import csv
import os
import time
from tradingview_ta import TA_Handler, Interval
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager


class SourceIPAdapter(HTTPAdapter):
    def __init__(self, source_ip, **kwargs):
        self.source_ip = source_ip
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs['source_address'] = (self.source_ip, 0)
        self.poolmanager = PoolManager(*args, **kwargs)


# لیست IPهایی که روی سرور ست شدن
SOURCE_IPS = [
    "91.107.142.98",
    "159.69.101.205",
    "49.12.112.219"
]

tehran_tz = pytz.timezone('Asia/Tehran')

current_ip_index = 0

def get_session_with_next_ip():
    global current_ip_index
    ip = SOURCE_IPS[current_ip_index]
    current_ip_index = (current_ip_index + 1) % len(SOURCE_IPS)
    session = requests.Session()
    adapter = SourceIPAdapter(ip)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session, ip


def get_live_data():
    session, used_ip = get_session_with_next_ip()

    handler = TA_Handler(
        symbol="EURUSD",
        screener="forex",
        exchange="FOREXCOM",
        interval=Interval.INTERVAL_1_MINUTE,
        # proxies=None,
    )

    try:
        now_tehran = datetime.now(tehran_tz)
        timestamp = now_tehran.strftime("%Y-%m-%d %H:%M")  # بدون ثانیه
        analysis = handler.get_analysis()
        indicators = analysis.indicators

        open_price = indicators.get('open')
        high_price = indicators.get('high')
        low_price = indicators.get('low')
        close_price = indicators.get('close')

        print(f"✅ {timestamp} | IP: {used_ip} => O:{open_price}, H:{high_price}, L:{low_price}, C:{close_price}")
        return {'timestamp': timestamp, 'open': open_price, 'high': high_price, 'low': low_price, 'close': close_price, 'ip': used_ip}

    except Exception as e:
        print(f"❌ خطا: {e} | IP: {used_ip}")


def get_1min_data_and_save():
    filename = 'eurusd_prices_multiip.csv'

    if not os.path.isfile(filename):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'ip'])

    last_minute = None
    while True:
        now_tehran = datetime.now(tehran_tz)
        current_minute = now_tehran.strftime("%Y-%m-%d %H:%M")

        if current_minute != last_minute and now_tehran.second == 59:
            last_minute = current_minute
            try:
                data = get_live_data()
                if data:
                    with open(filename, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            data['timestamp'],
                            data['open'],
                            data['high'],
                            data['low'],
                            data['close'],
                            data['ip']
                        ])
            except Exception as e:
                print(f"❌ خطا در ذخیره‌سازی: {e}")
        else:
            time.sleep(0.5)

