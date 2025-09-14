
### command:  PYTHONPATH=. python3 backtest/test_trading.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest import TestCase, main
from datetime import datetime
import pandas as pd
import numpy as np
from fibo_calculate import fibonacci_retracement
from get_legs import get_legs
from swing import get_swing_points
from utils import BotState


class TestFibonacciCalculation(TestCase):
    def setUp(self):
        self.start_price = 1.0000
        self.end_price = 1.0100
        
    def test_fibonacci_levels(self):
        # Test basic fibonacci calculation
        fib_levels = fibonacci_retracement(self.start_price, self.end_price)
        
        self.assertEqual(fib_levels['0.0'], self.start_price)
        self.assertEqual(fib_levels['1.0'], self.end_price)
        self.assertAlmostEqual(fib_levels['0.705'], 1.00705)
        self.assertAlmostEqual(fib_levels['0.9'], 1.009)

    def test_reverse_fibonacci(self):
        # Test fibonacci for bearish move
        fib_levels = fibonacci_retracement(self.end_price, self.start_price)
        
        self.assertEqual(fib_levels['0.0'], self.end_price)
        self.assertEqual(fib_levels['1.0'], self.start_price)


class TestTradingStrategy(TestCase):
    def setUp(self):
        # Load historical data
        self.data = pd.read_csv("backtest/eurusd_prices_multiip4.csv", 
                              parse_dates=["timestamp"], 
                              index_col="timestamp")
        self.data['status'] = np.where(self.data['open'] > self.data['close'], 
                                      'bearish', 'bullish')
        self.state = BotState()
        self.trades = []
        
    def test_backtest_strategy(self):
        start_index = 0
        wins = 0
        losses = 0

        while start_index < len(self.data) - 100:
            window_data = self.data[start_index:start_index+50]
            legs = get_legs(window_data)

            if len(legs) >= 3:
                swing_type, is_swing = get_swing_points(window_data, legs[-3:])
                if is_swing:
                    entry = window_data.iloc[-1]['close']
                    stop = legs[-2]['end_value']
                    diff = abs(entry - stop)
                    target = entry + 1.2 * diff if swing_type == 'bullish' else entry - 1.2 * diff

                    future_data = self.data[window_data.index[-1]:].iloc[1:20]
                    for _, row in future_data.iterrows():
                        if swing_type == 'bullish':
                            if row['low'] <= stop:
                                losses += 1
                                break
                            elif row['high'] >= target:
                                wins += 1
                                break
                        else:
                            if row['high'] >= stop:
                                losses += 1
                                break
                            elif row['low'] <= target:
                                wins += 1
                                break

            start_index += 1

        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        print(f"\nBacktest Results:")
        print(f"================")
        print(f"Total trades: {total_trades}")
        print(f"Wins: {wins}")
        print(f"Losses: {losses}")
        print(f"Win rate: {win_rate:.2f}%")

        self.assertGreater(win_rate, 40.0, "Win rate should be above 40%")


if __name__ == '__main__':
    main()