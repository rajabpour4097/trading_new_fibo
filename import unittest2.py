import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from mt5_connector import MT5Connector
import MetaTrader5 as mt5

# Import the MT5Connector class

class TestCloseAllPositions(unittest.TestCase):
    """Test suite for the close_all_positions method of MT5Connector"""
    
    def setUp(self):
        """Set up a fresh MT5Connector instance for each test"""
        self.connector = MT5Connector()
    
    @patch('mt5_connector.mt5')
    def test_close_all_positions_no_positions(self, mock_mt5):
        """Test when there are no open positions"""
        # Setup: positions_get returns None (no positions)
        mock_mt5.positions_get.return_value = None
        
        # Execute
        self.connector.close_all_positions()
        
        # Verify
        mock_mt5.positions_get.assert_called_once_with(symbol=self.connector.symbol)
        mock_mt5.order_send.assert_not_called()
    
    @patch('mt5_connector.mt5')
    def test_close_all_positions_with_buy_position(self, mock_mt5):
        """Test closing a BUY position"""
        # Setup: Create a mock BUY position
        mock_position = MagicMock()
        mock_position.type = 0  # BUY position
        mock_position.volume = 0.1
        mock_position.ticket = 12345
        mock_mt5.positions_get.return_value = [mock_position]
        
        # Setup mock tick data
        mock_tick = MagicMock()
        mock_tick.bid = 1.12345  # BUY positions close at BID price
        mock_tick.ask = 1.12355
        mock_mt5.symbol_info_tick.return_value = mock_tick
        
        # Constants needed for verification
        mock_mt5.TRADE_ACTION_DEAL = mt5.TRADE_ACTION_DEAL
        mock_mt5.ORDER_TYPE_SELL = mt5.ORDER_TYPE_SELL
        mock_mt5.ORDER_TYPE_BUY = mt5.ORDER_TYPE_BUY
        mock_mt5.ORDER_TIME_GTC = mt5.ORDER_TIME_GTC
        mock_mt5.ORDER_FILLING_IOC = mt5.ORDER_FILLING_IOC
        
        # Execute
        self.connector.close_all_positions()
        
        # Verify
        mock_mt5.positions_get.assert_called_once_with(symbol=self.connector.symbol)
        mock_mt5.symbol_info_tick.assert_called_once_with(self.connector.symbol)
        
        # Verify correct order request
        expected_request = {
            "action": mock_mt5.TRADE_ACTION_DEAL,
            "symbol": self.connector.symbol,
            "volume": mock_position.volume,
            "type": mock_mt5.ORDER_TYPE_SELL,  # Close BUY with SELL
            "position": mock_position.ticket,
            "price": mock_tick.bid,
            "deviation": self.connector.deviation,
            "magic": self.connector.magic,
            "comment": "Close position",
            "type_time": mock_mt5.ORDER_TIME_GTC,
            "type_filling": mock_mt5.ORDER_FILLING_IOC,
        }
        mock_mt5.order_send.assert_called_once_with(expected_request)
    
    @patch('mt5_connector.mt5')
    def test_close_all_positions_with_sell_position(self, mock_mt5):
        """Test closing a SELL position"""
        # Setup: Create a mock SELL position
        mock_position = MagicMock()
        mock_position.type = 1  # SELL position
        mock_position.volume = 0.2
        mock_position.ticket = 67890
        mock_mt5.positions_get.return_value = [mock_position]
        
        # Setup mock tick data
        mock_tick = MagicMock()
        mock_tick.bid = 1.12345
        mock_tick.ask = 1.12355  # SELL positions close at ASK price
        mock_mt5.symbol_info_tick.return_value = mock_tick
        
        # Constants needed for verification
        mock_mt5.TRADE_ACTION_DEAL = mt5.TRADE_ACTION_DEAL
        mock_mt5.ORDER_TYPE_SELL = mt5.ORDER_TYPE_SELL
        mock_mt5.ORDER_TYPE_BUY = mt5.ORDER_TYPE_BUY
        mock_mt5.ORDER_TIME_GTC = mt5.ORDER_TIME_GTC
        mock_mt5.ORDER_FILLING_IOC = mt5.ORDER_FILLING_IOC
        
        # Execute
        self.connector.close_all_positions()
        
        # Verify
        mock_mt5.positions_get.assert_called_once_with(symbol=self.connector.symbol)
        mock_mt5.symbol_info_tick.assert_called_once_with(self.connector.symbol)
        
        # Verify correct order request
        expected_request = {
            "action": mock_mt5.TRADE_ACTION_DEAL,
            "symbol": self.connector.symbol,
            "volume": mock_position.volume,
            "type": mock_mt5.ORDER_TYPE_BUY,  # Close SELL with BUY
            "position": mock_position.ticket,
            "price": mock_tick.ask,
            "deviation": self.connector.deviation,
            "magic": self.connector.magic,
            "comment": "Close position",
            "type_time": mock_mt5.ORDER_TIME_GTC,
            "type_filling": mock_mt5.ORDER_FILLING_IOC,
        }
        mock_mt5.order_send.assert_called_once_with(expected_request)
    
    @patch('mt5_connector.mt5')
    def test_close_all_positions_multiple_positions(self, mock_mt5):
        """Test closing multiple positions of different types"""
        # Setup: Create mock positions of both types
        buy_position = MagicMock()
        buy_position.type = 0  # BUY type
        buy_position.volume = 0.1
        buy_position.ticket = 12345
        
        sell_position = MagicMock()
        sell_position.type = 1  # SELL type
        sell_position.volume = 0.2
        sell_position.ticket = 67890
        
        mock_mt5.positions_get.return_value = [buy_position, sell_position]
        
        # Setup mock tick data
        mock_tick = MagicMock()
        mock_tick.bid = 1.12345
        mock_tick.ask = 1.12355
        mock_mt5.symbol_info_tick.return_value = mock_tick
        
        # Constants needed for verification
        mock_mt5.TRADE_ACTION_DEAL = mt5.TRADE_ACTION_DEAL
        mock_mt5.ORDER_TYPE_SELL = mt5.ORDER_TYPE_SELL
        mock_mt5.ORDER_TYPE_BUY = mt5.ORDER_TYPE_BUY
        mock_mt5.ORDER_TIME_GTC = mt5.ORDER_TIME_GTC
        mock_mt5.ORDER_FILLING_IOC = mt5.ORDER_FILLING_IOC
        
        # Execute
        self.connector.close_all_positions()
        
        # Verify
        mock_mt5.positions_get.assert_called_once_with(symbol=self.connector.symbol)
        self.assertEqual(mock_mt5.symbol_info_tick.call_count, 2)
        self.assertEqual(mock_mt5.order_send.call_count, 2)
        
        # Get all order_send calls
        calls = mock_mt5.order_send.call_args_list
        
        # Verify each position was closed with the correct order type and price
        requests = [call[0][0] for call in calls]  # Extract the request dictionaries
        
        # Find the requests for each position
        buy_close_request = next((r for r in requests if r["position"] == buy_position.ticket), None)
        sell_close_request = next((r for r in requests if r["position"] == sell_position.ticket), None)
        
        self.assertIsNotNone(buy_close_request)
        self.assertIsNotNone(sell_close_request)
        
        # Verify buy position was closed with a SELL order at bid price
        self.assertEqual(buy_close_request["type"], mock_mt5.ORDER_TYPE_SELL)
        self.assertEqual(buy_close_request["price"], mock_tick.bid)
        
        # Verify sell position was closed with a BUY order at ask price
        self.assertEqual(sell_close_request["type"], mock_mt5.ORDER_TYPE_BUY)
        self.assertEqual(sell_close_request["price"], mock_tick.ask)

if __name__ == "__main__":
    unittest.main()