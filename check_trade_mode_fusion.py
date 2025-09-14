import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz
import time

# Initialize MT5
if not mt5.initialize():
    print(f"❌ MT5 initialization failed: {mt5.last_error()}")
    quit()

print("✅ MT5 initialized successfully")

# Check trading hours
iran_tz = pytz.timezone('Asia/Tehran')
utc_tz = pytz.UTC
utc_now = datetime.now(utc_tz)
iran_now = utc_now.astimezone(iran_tz)
print(f"🇮🇷 Current Iran Time: {iran_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"🌐 Current UTC Time: {utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")

# Get account info
account_info = mt5.account_info()
if account_info:
    print("\n=== Account Info ===")
    for prop in dir(account_info):
        if not prop.startswith('_'):
            print(f"{prop}: {getattr(account_info, prop)}")
    
    print(f"\nAccount Type: {'Demo' if account_info.trade_mode == 0 else 'Real'}")
    print(f"Trading Allowed: {account_info.trade_allowed}")
    print(f"Expert Trading Allowed: {account_info.trade_expert}")

# Get USDJPY symbol info
symbol = "USDJPY"
symbol_info = mt5.symbol_info(symbol)
if symbol_info:
    print(f"\n=== {symbol} Symbol Info ===")
    for prop in dir(symbol_info):
        if not prop.startswith('_'):
            print(f"{prop}: {getattr(symbol_info, prop)}")
    
    # Decode trade_mode value
    trade_modes = {
        0: "SYMBOL_TRADE_MODE_DISABLED (Trading disabled)",
        1: "SYMBOL_TRADE_MODE_LONGONLY (Long positions only)",
        2: "SYMBOL_TRADE_MODE_SHORTONLY (Short positions only)",
        3: "SYMBOL_TRADE_MODE_FULL (Full access)",
        4: "SYMBOL_TRADE_MODE_CLOSEONLY (Close only)"
    }
    
    print(f"\nTrade Mode: {trade_modes.get(symbol_info.trade_mode, 'Unknown')}")
    print(f"Trade Execution Mode: {symbol_info.trade_exemode}")
    print(f"Session Deals: {symbol_info.session_deals}")

# Check server settings
terminal_info = mt5.terminal_info()
if terminal_info:
    print("\n=== Terminal Info ===")
    print(f"Community Account: {terminal_info.community_account}")
    print(f"Connected: {terminal_info.connected}")
    print(f"DLLS Allowed: {terminal_info.dlls_allowed}")
    print(f"Trade Allowed: {terminal_info.trade_allowed}")
    print(f"Server Name: {terminal_info.name}")
    print(f"Path: {terminal_info.path}")
    print(f"Connected to: {terminal_info.connected_proxy if terminal_info.connected_proxy else 'Direct'}")

# Try to check if this is a temporary restriction
print("\n=== Market Hours Check ===")
print("Checking if this is a temporary trading restriction...")

# Get current and next day of week (0=Monday, 1=Tuesday, etc.)
current_day = iran_now.weekday()
next_day = (current_day + 1) % 7

# Check the session start and end for today and tomorrow
for day in [current_day, next_day]:
    day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][day]
    sessions = mt5.symbol_info_session_trades(symbol, day)
    
    if sessions:
        print(f"{day_name} Trading Sessions for {symbol}:")
        for session in sessions:
            start = datetime.fromtimestamp(session[0]).strftime('%H:%M:%S')
            end = datetime.fromtimestamp(session[1]).strftime('%H:%M:%S')
            print(f"  {start} - {end}")
    else:
        print(f"{day_name} Trading Sessions for {symbol}: None (Market Closed)")

# Try an order request to get specific rejection reason
print("\n=== Test Order ===")
print("Attempting to send a test order to get specific rejection reason...")

# Prepare a minimal request
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": 0.01,
    "type": mt5.ORDER_TYPE_BUY,
    "price": mt5.symbol_info_tick(symbol).ask,
    "deviation": 20,
    "magic": 234000,
    "comment": "Test Order",
    "type_time": mt5.ORDER_TIME_GTC,
}

# Try all filling modes
for filling_mode in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
    filling_name = {
        mt5.ORDER_FILLING_FOK: "FOK (Fill or Kill)",
        mt5.ORDER_FILLING_IOC: "IOC (Immediate or Cancel)",
        mt5.ORDER_FILLING_RETURN: "Return"
    }[filling_mode]
    
    request["type_filling"] = filling_mode
    result = mt5.order_send(request)
    
    print(f"\nTest with {filling_name} filling mode:")
    if result.retcode == 10009:
        print(f"✅ Order successful (unlikely with CLOSEONLY mode)")
        print(f"Order ticket: {result.order}")
    else:
        print(f"❌ Order failed with code {result.retcode}: {result.comment}")

mt5.shutdown()
print("\nMT5 connection closed")