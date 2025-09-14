import pandas as pd
from datetime import datetime
import MetaTrader5 as mt5

# Initialize MT5
mt5.initialize()

# Create a DataFrame to store data
data_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'profit', 'trade_signal']
data = pd.DataFrame(columns=data_columns)

while True:
    # Get live data
    tick = mt5.symbol_info_tick("EURUSD")
    if tick:
        current_time = datetime.now()
        open_price = tick.bid  # or tick.ask depending on your strategy
        high_price = tick.high
        low_price = tick.low
        close_price = tick.last
        volume = 0.01  # Example volume, adjust as needed
        profit = 0  # Calculate profit based on your strategy
        trade_signal = "buy"  # or "sell", based on your strategy

        # Append data to DataFrame
        data = data.append({
            'timestamp': current_time,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'profit': profit,
            'trade_signal': trade_signal
        }, ignore_index=True)

    # Save to CSV every N iterations or at specific intervals
    if len(data) % 100 == 0:  # Adjust the condition as needed
        data.to_csv('trading_data.csv', index=False)

    # Sleep for a while before the next iteration
    time.sleep(60)  # Adjust the sleep time as needed