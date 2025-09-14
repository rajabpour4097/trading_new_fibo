from mt5_connector import MT5Connector
import MetaTrader5 as mt5
import time

def test_buy_position():
    """Test opening a buy position with 1:1.2 risk-reward ratio"""
    # Initialize connector
    connector = MT5Connector()
    if not connector.initialize():
        print("❌ Failed to initialize MT5")
        return
    
    # Check if trading is allowed
    can_trade, message = connector.can_trade()
    print(f"🔍 Trading status: {message}")
    
    # Retrieve symbol information explicitly
    symbol_info = mt5.symbol_info(connector.symbol)
    if symbol_info is None:
        print(f"❌ Failed to get symbol info for {connector.symbol}")
        connector.shutdown()
        return
    
    # Check symbol properties and market state
    connector.check_symbol_properties()
    connector.check_market_state()
    
    # Get current market price
    price_data = connector.get_live_price()
    if not price_data:
        print("❌ Failed to get price data")
        connector.shutdown()
        return
    
    tick = mt5.symbol_info_tick(connector.symbol)
    if not tick:
        print("❌ Failed to get tick data")
        connector.shutdown()
        return
    
    # Calculate SL and TP with 1:1.2 ratio
    entry_price = tick.ask
    sl_distance = 30 * symbol_info.point  # 30 points
    tp_distance = sl_distance * 1.2  # 1.2 times SL
    
    sl_price = entry_price - sl_distance
    tp_price = entry_price + tp_distance
    
    print(f"📊 Test BUY order parameters:")
    print(f"   Symbol: {connector.symbol}")
    print(f"   Entry: {entry_price:.5f}")
    print(f"   SL: {sl_price:.5f} ({sl_distance/symbol_info.point} points)")
    print(f"   TP: {tp_price:.5f} ({tp_distance/symbol_info.point} points)")
    print(f"   Risk-Reward: 1:{tp_distance/sl_distance:.2f}")
    
    # Attempt to open position
    print("\n🔄 Attempting to open BUY position...")
    result = connector.open_buy_position(
        tick=tick,
        sl=sl_price,
        tp=tp_price,
        comment="Test BUY 1:1.2"
    )
    
    # Check result
    if result:
        if hasattr(result, 'retcode') and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ BUY position opened successfully!")
            print(f"   Order ticket: {result.order}")
            print(f"   Deal ticket: {getattr(result, 'deal', 'N/A')}")
            
            # Wait a moment and check the position
            time.sleep(1)
            position = mt5.positions_get(ticket=result.order)
            if position:
                pos = position[0]
                print(f"\n📈 Position details:")
                print(f"   Type: {'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL'}")
                print(f"   Volume: {pos.volume}")
                print(f"   Open price: {pos.price_open}")
                print(f"   Current price: {pos.price_current}")
                print(f"   SL: {pos.sl}")
                print(f"   TP: {pos.tp}")
        else:
            print(f"❌ Failed to open position: {result.retcode} - {result.comment}")
    else:
        print("❌ Order placement returned None")
        last_error = mt5.last_error()
        print(f"   Last error: {last_error}")
    
    # Close all positions
    print("\n🔄 Closing all positions...")
    connector.close_all_positions()
    
    # Shutdown
    connector.shutdown()
    print("✅ Test completed")

if __name__ == "__main__":
    test_buy_position()