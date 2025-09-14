"""Utilities to fetch last month 1-minute OHLC and tick data from MetaTrader5.

Provides two main functions:
- fetch_last_month_m1(symbol) -> DataFrame with columns open,high,low,close indexed by UTC timestamp
- fetch_ticks_between(symbol, start, end) -> DataFrame of ticks (bid,ask,last,time)

Note: Requires a running MT5 terminal logged into the target account.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
import MetaTrader5 as mt5
import pandas as pd

UTC = timezone.utc


def _ensure_initialized() -> bool:
    if not mt5.initialize():
        return False
    return True


def fetch_last_month_m1(symbol: str) -> pd.DataFrame:
    if not _ensure_initialized():
        raise RuntimeError("MT5 initialize failed")
    end = datetime.now(UTC)
    start = end - timedelta(days=30)
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start, end)
    if rates is None:
        raise RuntimeError(f"No M1 data returned for {symbol}")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    return df[['open','high','low','close']]


def fetch_ticks_between(symbol: str, start: datetime, end: datetime, max_points: int = 200_000) -> pd.DataFrame:
    """Fetch raw ticks between start and end (UTC datetimes).
    If range large, splits into daily chunks to avoid server limits.
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    if not _ensure_initialized():
        raise RuntimeError("MT5 initialize failed")
    out = []
    cur_start = start
    while cur_start < end:
        cur_end = min(cur_start + timedelta(days=1), end)
        ticks = mt5.copy_ticks_range(symbol, cur_start, cur_end, mt5.COPY_TICKS_ALL)
        if ticks is not None and len(ticks):
            df = pd.DataFrame(ticks)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            df.set_index('time', inplace=True)
            out.append(df[['bid','ask','last']])
        cur_start = cur_end
        if sum(len(o) for o in out) >= max_points:
            break
    if not out:
        return pd.DataFrame(columns=['bid','ask','last'])
    ticks_df = pd.concat(out).sort_index()
    return ticks_df

__all__ = [
    'fetch_last_month_m1',
    'fetch_ticks_between',
]


def fetch_m1_range(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch M1 candles for an arbitrary UTC date range (inclusive start, exclusive end)."""
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    if not _ensure_initialized():
        raise RuntimeError("MT5 initialize failed")
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start, end)
    if rates is None:
        raise RuntimeError(f"No data returned for {symbol} in range {start} - {end}")
    df = pd.DataFrame(rates)
    if df.empty:
        return pd.DataFrame(columns=['open','high','low','close'])
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    return df[['open','high','low','close']]

__all__.append('fetch_m1_range')

def fetch_symbol_specs(symbol: str):
    """Fetch symbol specification (tick_size, tick_value, point, digits).
    Returns dict or raises RuntimeError if unavailable."""
    if not _ensure_initialized():
        raise RuntimeError("MT5 initialize failed")
    info = mt5.symbol_info(symbol)
    if not info:
        raise RuntimeError(f"Symbol info not found for {symbol}")
    return {
        'tick_size': getattr(info, 'trade_tick_size', None) or getattr(info, 'tick_size', None) or info.point,
        'tick_value': getattr(info, 'trade_tick_value', None) or getattr(info, 'tick_value', None),
        'point': info.point,
        'digits': info.digits,
        'trade_stops_level': getattr(info, 'trade_stops_level', 0),
        'volume_min': getattr(info, 'volume_min', 0.01),
        'volume_step': getattr(info, 'volume_step', 0.01),
    'contract_size': getattr(info, 'trade_contract_size', None) or getattr(info, 'contract_size', None),
    }

__all__.append('fetch_symbol_specs')
