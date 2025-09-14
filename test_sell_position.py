from mt5_connector import MT5Connector
import MetaTrader5 as mt5
import time

def test_sell_position():
    """Test opening a sell position with 1:1.2 risk-reward ratio"""
    connector = MT5Connector()
    if not connector.initialize():
        print("❌ MT5 init failed")
        return

    can_trade, message = connector.can_trade()
    print(f"🔍 Trading status: {message}")

    symbol_info = mt5.symbol_info(connector.symbol)
    if symbol_info is None:
        print(f"❌ Symbol info failed: {connector.symbol}")
        connector.shutdown()
        return

    connector.check_symbol_properties()
    connector.check_market_state()

    if not connector.get_live_price():
        print("❌ Price fetch failed")
        connector.shutdown()
        return

    tick = mt5.symbol_info_tick(connector.symbol)
    if not tick:
        print("❌ Tick fetch failed")
        connector.shutdown()
        return

    # For SELL: entry at Bid, SL above, TP below
    entry_price = tick.bid
    sl_distance = 30 * symbol_info.point      # 30 points
    tp_distance = sl_distance * 1.2           # 1.2 RR

    sl_price = entry_price + sl_distance
    tp_price = entry_price - tp_distance

    print("📊 Test SELL order parameters:")
    print(f"   Symbol: {connector.symbol}")
    print(f"   Entry: {entry_price:.5f}")
    print(f"   SL: {sl_price:.5f} ({sl_distance/symbol_info.point} points)")
    print(f"   TP: {tp_price:.5f} ({tp_distance/symbol_info.point} points)")
    print(f"   Risk-Reward: 1:{tp_distance/sl_distance:.2f}")

    print("\n🔄 Attempting to open SELL position...")
    result = connector.open_sell_position(
        tick=tick,
        sl=sl_price,
        tp=tp_price,
        comment="Test SELL 1:1.2"
    )

    if result:
        if hasattr(result, "retcode") and result.retcode == mt5.TRADE_RETCODE_DONE:
            print("✅ SELL position opened")
            print(f"   Order ticket: {result.order}")
            print(f"   Deal ticket: {getattr(result, 'deal', 'N/A')}")
            time.sleep(1)
            # positions_get(ticket=...) expects position ticket, not order; so fetch by symbol
            positions = mt5.positions_get(symbol=connector.symbol)
            if positions:
                for pos in positions:
                    if pos.type == mt5.POSITION_TYPE_SELL:
                        print("\n📈 Position details:")
                        print(f"   Type: SELL")
                        print(f"   Volume: {pos.volume}")
                        print(f"   Open price: {pos.price_open}")
                        print(f"   Current price: {pos.price_current}")
                        print(f"   SL: {pos.sl}")
                        print(f"   TP: {pos.tp}")
                        break
        else:
            print(f"❌ Open failed: {getattr(result,'retcode','?')} - {getattr(result,'comment','')}")
    else:
        print("❌ Order placement returned None")
        print(f"   Last error: {mt5.last_error()}")

    print("\n🔄 Closing all positions...")
    connector.close_all_positions()
    connector.shutdown()
    print("✅ Test completed")

if __name__ == "__main__":
    test_sell_position()