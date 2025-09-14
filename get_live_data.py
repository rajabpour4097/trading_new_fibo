import requests
from requests.adapters import HTTPAdapter
from urllib3 import PoolManager
from datetime import datetime
import pytz
from time import sleep

# کلاس مربوط به IP چرخشی
class SourceIPAdapter(HTTPAdapter):
    def __init__(self, source_ip, **kwargs):
        self.source_ip = source_ip
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs['source_address'] = (self.source_ip, 0)
        self.poolmanager = PoolManager(*args, **kwargs)

# لیست IPهای سرور
SOURCE_IPS = [
    "91.107.142.98",
    "159.69.101.205",
    "49.12.112.219"
]

API_KEY = "AEiGyrU2Mo5hbKEuxOOm"  # Tradermade API key
SYMBOL = "EURUSD"
tehran_tz = pytz.timezone('Asia/Tehran')
current_ip_index = 0

def log(msg, color=None):
    print(msg)  # می‌تونی رنگ اضافه کنی با colorama

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
    url = f"https://marketdata.tradermade.com/api/v1/live?currency={SYMBOL}&api_key={API_KEY}"

    try:
        response = session.get(url, timeout=5)
        data = response.json()

        if "quotes" not in data or not data["quotes"]:
            raise Exception("❌ Invalid API response: quotes empty")

        quote = data["quotes"][0]

        mid = quote["mid"]
        bid = quote["bid"]
        ask = quote["ask"]
        timestamp = datetime.now(tehran_tz).strftime("%Y-%m-%d %H:%M:%S")

        print(f"📶 {timestamp} | IP: {used_ip} | bid: {bid}, ask: {ask}, mid: {mid}")
        return {"timestamp": timestamp, "bid": bid, "ask": ask, "mid": mid, "ip": used_ip}

    except Exception as e:
        print(f"❌ Error from IP {used_ip}: {e}")
        return None
