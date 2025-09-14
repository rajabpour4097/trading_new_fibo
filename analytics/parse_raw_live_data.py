import ast
from pathlib import Path
import pandas as pd
from datetime import datetime

RAW_DIR = Path('analytics/vps-data/raw')
OUT_DIR = Path('analytics/vps-data/converted')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# We expect two files: 'trades' and 'market' containing python example code with log_trade or save_trade_data invocations.

TRADE_CALL_NAMES = {'log_trade','log_trade_data'}
MARKET_CALL_NAMES = {'save_trade_data'}

def extract_calls(file_path: Path):
    try:
        tree = ast.parse(file_path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f'Failed to parse {file_path}: {e}')
        return []
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in TRADE_CALL_NAMES.union(MARKET_CALL_NAMES):
                args = []
                for a in node.args:
                    if isinstance(a, ast.Constant):
                        args.append(a.value)
                    else:
                        args.append(None) # Non literal
                calls.append((name,args))
    return calls

def convert_trades(calls):
    rows = []
    for name,args in calls:
        if name in TRADE_CALL_NAMES:
            # log_trade(price, volume, trade_type, profit_loss, timestamp)
            if len(args) == 5:
                price, volume, trade_type, profit_loss, timestamp = args
            else:
                continue
            rows.append({
                'timestamp': timestamp if isinstance(timestamp,str) else datetime.utcnow().isoformat(),
                'price': price,
                'volume': volume,
                'trade_type': trade_type,
                'profit_loss': profit_loss
            })
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT_DIR / 'trades_converted.csv', index=False)
        print(f'Wrote {len(df)} trades to converted/trades_converted.csv')


def convert_market(calls):
    rows = []
    for name,args in calls:
        if name in MARKET_CALL_NAMES:
            # save_trade_data(timestamp, open_price, close_price, high_price, low_price, volume, profit_loss, trade_signal)
            if len(args) == 8:
                ts, open_p, close_p, high_p, low_p, volume, profit_loss, signal = args
            else:
                continue
            rows.append({
                'timestamp': ts if isinstance(ts,str) else datetime.utcnow().isoformat(),
                'open': open_p,
                'close': close_p,
                'high': high_p,
                'low': low_p,
                'volume': volume,
                'profit_loss': profit_loss,
                'signal': signal
            })
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT_DIR / 'market_converted.csv', index=False)
        print(f'Wrote {len(df)} market rows to converted/market_converted.csv')


def main():
    trade_file = RAW_DIR / 'trades'
    market_file = RAW_DIR / 'market'
    trade_calls = extract_calls(trade_file) if trade_file.exists() else []
    market_calls = extract_calls(market_file) if market_file.exists() else []
    convert_trades(trade_calls)
    convert_market(market_calls)

if __name__ == '__main__':
    main()
