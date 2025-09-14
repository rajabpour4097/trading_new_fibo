from mt5_connector import MT5Connector
import time

def test_mt5_connection():
    """Simple test to verify MT5 connectivity and AutoTrading status"""
    # Initialize the connector with proper parameters
    # Adjust these parameters to match your MT5Connector's __init__ method
    mt5_conn = MT5Connector()
    
    # Check if MT5 is initialized
    if mt5_conn.initialize():
        print("✅ MT5 initialized successfully")
    else:
        print("❌ MT5 initialization failed")
        return
    
    # Test if AutoTrading is enabled
    can_trade, message = mt5_conn.can_trade()
    print(f"AutoTrading status: {message}")
    
    if can_trade:
        print("✅ AutoTrading is enabled")
    else:
        print("❌ AutoTrading is disabled - Please enable it in MT5!")
        print("   Instructions to enable AutoTrading:")
        print("   1. Open MetaTrader 5")
        print("   2. Click the 'AutoTrading' button in the toolbar (or press Alt+T)")
        print("   3. The button should turn green when enabled")
    
    # Additional tests if needed
    print("\nTesting order permissions:")
    try:
        # Testing connection without actually placing an order
        mt5_conn.test_order_permissions()
    except AttributeError:
        print("Method test_order_permissions() not available")
    
    # Clean up
    try:
        mt5_conn.shutdown()
        print("\nMT5 connection test completed")
    except AttributeError:
        print("\nMT5 shutdown method not available")

if __name__ == "__main__":
    test_mt5_connection()