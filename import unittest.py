import unittest
from unittest.mock import patch, MagicMock, call
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import pytest

# Import the necessary modules directly
from main_metatrader import main
from utils import BotState

class TestMainMetatrader(unittest.TestCase):
    
    def setUp(self):
        # Create sample market data for testing
        self.sample_data = self.create_sample_data()
        
        # Set up the patchers - use direct module names instead of full package paths
        self.mt5_conn_patcher = patch('main_metatrader.MT5Connector')
        self.get_legs_patcher = patch('main_metatrader.get_legs')
        self.get_swing_points_patcher = patch('main_metatrader.get_swing_points')
        self.fib_patcher = patch('main_metatrader.fibonacci_retracement')
        self.log_patcher = patch('main_metatrader.log')
        self.sleep_patcher = patch('main_metatrader.sleep')
        
        # Start the patchers
        self.mock_mt5_conn = self.mt5_conn_patcher.start()
        self.mock_get_legs = self.get_legs_patcher.start()
        self.mock_get_swing_points = self.get_swing_points_patcher.start()
        self.mock_fib = self.fib_patcher.start()
        self.mock_log = self.log_patcher.start()
        self.mock_sleep = self.sleep_patcher.start()
        
        # Configure the MT5Connector mock
        self.mt5_instance = MagicMock()
        self.mock_mt5_conn.return_value = self.mt5_instance
        self.mt5_instance.initialize.return_value = True
        self.mt5_instance.can_trade.return_value = (True, "Trading is allowed")
        self.mt5_instance.get_historical_data.return_value = self.sample_data
        
        # Set up legs data
        self.legs_data = [
            {'start': self.sample_data.index[0], 'end': self.sample_data.index[10], 'start_value': 1.2000, 'end_value': 1.2050},
            {'start': self.sample_data.index[10], 'end': self.sample_data.index[20], 'start_value': 1.2050, 'end_value': 1.2010},
            {'start': self.sample_data.index[20], 'end': self.sample_data.index[30], 'start_value': 1.2010, 'end_value': 1.2070}
        ]
        self.mock_get_legs.return_value = self.legs_data
        
        # Set up swing points data
        self.mock_get_swing_points.return_value = ('bullish', True)
        
        # Set up fibonacci retracement levels
        self.mock_fib.return_value = {
            '0.0': 1.2070,
            '0.236': 1.2060,
            '0.382': 1.2050,
            '0.5': 1.2040,
            '0.618': 1.2030,
            '0.705': 1.2025,
            '0.786': 1.2020,
            '0.9': 1.2015,
            '1.0': 1.2010
        }
    
    def tearDown(self):
        # Stop all patchers
        self.mt5_conn_patcher.stop()
        self.get_legs_patcher.stop()
        self.get_swing_points_patcher.stop()
        self.fib_patcher.stop()
        self.log_patcher.stop()
        self.sleep_patcher.stop()
    
    def create_sample_data(self):
        # Create a sample DataFrame that resembles market data
        dates = [datetime.now() - timedelta(minutes=i) for i in range(100, 0, -1)]
        data = {
            'open': np.random.uniform(1.2000, 1.2100, 100),
            'high': np.random.uniform(1.2050, 1.2150, 100),
            'low': np.random.uniform(1.1950, 1.2050, 100),
            'close': np.random.uniform(1.2000, 1.2100, 100),
            'volume': np.random.randint(100, 1000, 100)
        }
        df = pd.DataFrame(data, index=dates)
        df['status'] = np.where(df['open'] > df['close'], 'bearish', 'bullish')
        return df
    
    @patch('main_metatrader.BotState')
    def test_main_initialization(self, mock_bot_state):
        """Test that the main function initializes correctly"""
        # Configure the mock to exit the infinite loop after one iteration
        self.mt5_instance.get_historical_data.side_effect = [
            self.sample_data,  # First call returns data
            KeyboardInterrupt  # Second call raises KeyboardInterrupt to exit loop
        ]
        
        # Run the main function
        with self.assertRaises(KeyboardInterrupt):
            main()
        
        # Assert that initialization was performed
        self.mt5_instance.initialize.assert_called_once()
        self.mt5_instance.check_symbol_properties.assert_called_once()
        self.mt5_instance.test_filling_modes.assert_called_once()
        self.mt5_instance.check_trading_limits.assert_called_once()
    
    @patch('main_metatrader.BotState')
    def test_bullish_swing_detection(self, mock_bot_state):
        """Test that bullish swings are correctly identified and processed"""
        # Configure mocks for a bullish swing scenario
        mock_state = MagicMock()
        mock_state.fib_levels = None
        mock_state.true_position = False
        mock_bot_state.return_value = mock_state
        
        # Configure the mocks to exit the loop after processing a bullish swing
        self.mt5_instance.get_historical_data.side_effect = [
            self.sample_data,  # First call returns data
            KeyboardInterrupt  # Second call raises KeyboardInterrupt to exit loop
        ]
        
        # Run the main function
        with self.assertRaises(KeyboardInterrupt):
            main()
        
        # Assert that the swing detection was called
        self.mock_get_swing_points.assert_called()
        
        # Check that fibonacci levels were calculated (this will happen for a bullish swing)
        self.mock_fib.assert_called()
    
    @patch('main_metatrader.BotState')
    def test_trading_execution_bullish(self, mock_bot_state):
        """Test that trades are executed correctly for bullish swings"""
        # Configure mocks for a trading scenario
        mock_state = MagicMock()
        mock_state.fib_levels = {
            '0.0': 1.2070,
            '0.236': 1.2060,
            '0.382': 1.2050,
            '0.5': 1.2040,
            '0.618': 1.2030,
            '0.705': 1.2025,
            '0.786': 1.2020,
            '0.9': 1.2015,
            '1.0': 1.2010
        }
        mock_state.true_position = True
        mock_state.last_touched_705_point_up = self.sample_data.iloc[50]
        mock_bot_state.return_value = mock_state
        
        # Configure the mocks to exit the loop after executing a trade
        self.mt5_instance.get_historical_data.side_effect = [
            self.sample_data,  # First call returns data
            KeyboardInterrupt  # Second call raises KeyboardInterrupt to exit loop
        ]
        
        # Make open_buy_position return success
        self.mt5_instance.open_buy_position.return_value = True
        
        # Run the main function
        with self.assertRaises(KeyboardInterrupt):
            main()
        
        # Assert that a buy position was opened
        self.mt5_instance.open_buy_position.assert_called()
        
        # Check that state was reset after the trade
        mock_state.reset.assert_called()

if __name__ == '__main__':
    unittest.main()