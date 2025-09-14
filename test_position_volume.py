import unittest
from mt5_connector import MT5Connector
from unittest.mock import MagicMock

class TestPositionVolume(unittest.TestCase):
    def setUp(self):
        self.connector = MT5Connector()
        # Mock symbol info and account info for volume calculation
        self.connector.symbol = 'TEST'
        self.connector.lot = 1.0
        self.connector.deviation = 10
        self.connector.magic = 123456
        self.connector.commission_per_lot_side = 0.0
        # Patch mt5 methods
        import sys
        sys.modules['mt5'] = MagicMock()
        mt5 = sys.modules['mt5']
        mt5.account_info.return_value = MagicMock(balance=10000)
        mt5.symbol_info.return_value = MagicMock(
            tick_size=0.01,
            tick_value=1.0,
            volume_step=0.01,
            volume_min=0.01,
            volume_max=100.0
        )
        mt5.ORDER_TYPE_BUY = 0
        mt5.ORDER_TYPE_SELL = 1
        mt5.TRADE_ACTION_DEAL = 1
        mt5.ORDER_TIME_GTC = 0
    def test_open_buy_position_volume(self):
        tick = MagicMock(ask=1.2345, bid=1.2340)
        sl = 1.2300
        tp = 1.2400
        # Test with explicit volume
        result = self.connector.open_buy_position(tick, sl, tp, volume=2.5)
        self.assertEqual(result['volume'], self.connector._normalize_volume(2.5))
        # Test with risk_pct
        result = self.connector.open_buy_position(tick, sl, tp, risk_pct=0.02)
        expected_vol = self.connector.calculate_volume_by_risk(tick.ask, sl, tick, 0.02)
        expected_vol = self.connector.calculate_volume_by_risk(tick.ask, sl, tick, 0.02)
        self.assertEqual(result['volume'], expected_vol)
    def test_open_sell_position_volume(self):
        tick = MagicMock(ask=1.2345, bid=1.2340)
        sl = 1.2380
        tp = 1.2300
        # Test with explicit volume
        result = self.connector.open_sell_position(tick, sl, tp, volume=3.0)
        self.assertEqual(result['volume'], self.connector._normalize_volume(3.0))
        # Test with risk_pct
        result = self.connector.open_sell_position(tick, sl, tp, risk_pct=0.01)
        expected_vol = self.connector.calculate_volume_by_risk(tick.bid, sl, tick, 0.01)
        self.assertEqual(result['volume'], expected_vol)
if __name__ == '__main__':
    unittest.main()
