#!/usr/bin/env python3
"""
Debug script to check the full-logic backtest step by step
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backtest.advanced_tick_engine import AdvancedTickBacktester, FullConfig
from backtest.engine import load_ohlc_csv
from backtest.mt5_data_fetch import fetch_symbol_specs

def debug_backtest():
    print("🔍 Debugging Full-Logic Backtest")
    print("=" * 50)
    
    # Load data
    df = load_ohlc_csv('backtest/EURUSD_August_2025_M1.csv')
    print(f"📊 Data loaded: {len(df)} bars")
    
    # Create config
    config = FullConfig(
        symbol='EURUSD',
        threshold=4,
        require_second_touch=False,  # Single touch for easier debugging
        debug=True
    )
    
    # Mock symbol specs (since we don't have MT5 connection)
    specs = {
        'digits': 5,
        'point': 0.00001,
        'tick_size': 0.00001,
        'tick_value': 1.0,
        'contract_size': 100000,
        'volume_min': 0.01,
        'volume_step': 0.01,
        'trade_stops_level': 0
    }
    
    # Create backtest engine
    bt = AdvancedTickBacktester(config, specs, None)  # No ticks for simplicity
    
    # Run a mini version to debug
    df_small = df.iloc[:200]  # Test first 200 bars only
    if 'status' not in df_small.columns:
        df_small['status'] = pd.np.where(df_small['close'] >= df_small['open'], 'bullish', 'bearish')
    
    print(f"🔍 Testing first {len(df_small)} bars...")
    print("Looking for fibonacci setups and touch events...")
    
    ws = config.window_size
    fib_found = 0
    touches_found = 0
    true_positions = 0
    
    for i in range(ws, len(df_small)):
        window = df_small.iloc[i-ws:i+1]
        bar_ts = window.index[-1]
        bar_row = window.iloc[-1]
        
        bt._update_state(window)
        
        # Check state
        if bt.state.fib_levels is not None:
            if fib_found < 5:  # Log first few fibs
                fib = bt.state.fib_levels
                print(f"📐 Fibonacci found at {bar_ts}:")
                print(f"   0.705 level: {fib['0.705']:.5f}")
                print(f"   Bar high: {bar_row['high']:.5f}")
                print(f"   Bar low: {bar_row['low']:.5f}")
                print(f"   Touch check: {bar_row['low']} <= {fib['0.705']:.5f} <= {bar_row['high']}")
                if bar_row['low'] <= fib['0.705'] <= bar_row['high']:
                    print(f"   ✅ TOUCH CONDITION MET!")
                else:
                    print(f"   ❌ No touch")
            fib_found += 1
            
        if hasattr(bt.state, 'last_touched_705_point_up') and bt.state.last_touched_705_point_up is not None:
            if touches_found < 3:
                print(f"👆 Touch UP detected at {bar_ts}")
            touches_found += 1
            
        if hasattr(bt.state, 'last_touched_705_point_down') and bt.state.last_touched_705_point_down is not None:
            if touches_found < 3:
                print(f"👇 Touch DOWN detected at {bar_ts}")
            touches_found += 1
            
        if bt.state.true_position:
            if true_positions < 3:
                print(f"✅ TRUE POSITION triggered at {bar_ts}")
                print(f"   - Fib levels: {bt.state.fib_levels}")
                print(f"   - Last swing type: {getattr(bt.state, 'last_swing_type', 'None')}")
                print(f"   - Bar close: {bar_row['close']}")
                # Try to place order
                bt._maybe_open(bar_ts, bar_row)
            true_positions += 1
    
    print("\n📊 Debug Summary:")
    print(f"   - Fibonacci setups found: {fib_found}")
    print(f"   - 0.705 touches detected: {touches_found}")
    print(f"   - True positions triggered: {true_positions}")
    print(f"   - Actual trades placed: {len(bt.trades)}")
    print(f"   - Skip reasons: {bt.skip_reasons}")
    
    return bt

if __name__ == "__main__":
    bt = debug_backtest()
