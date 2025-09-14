import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sns.set_theme(style='whitegrid')

# Inputs: converted CSVs produced by parse_raw_live_data.py
# Outputs: PNG charts + summary text in analytics/reports/live

def load_converted(base: Path):
    conv_dir = base / 'vps-data' / 'converted'
    trades = pd.read_csv(conv_dir / 'trades_converted.csv', parse_dates=['timestamp']) if (conv_dir / 'trades_converted.csv').exists() else pd.DataFrame()
    market = pd.read_csv(conv_dir / 'market_converted.csv', parse_dates=['timestamp']) if (conv_dir / 'market_converted.csv').exists() else pd.DataFrame()
    return trades, market


def plot_trades(trades: pd.DataFrame, out: Path):
    if trades.empty:
        return
    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.scatter(trades['timestamp'], trades['price'], c=['green' if t=='buy' else 'red' for t in trades['trade_type']])
    ax.set_title('Executed Trades')
    ax.set_ylabel('Price')
    ax.set_xlabel('Time')
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / 'trades_scatter.png', dpi=140)
    plt.close(fig)

    # Profit/Loss bar
    fig, ax = plt.subplots(figsize=(5,3))
    colors = ['green' if v >=0 else 'red' for v in trades['profit_loss']]
    ax.bar(range(len(trades)), trades['profit_loss'], color=colors)
    ax.set_title('Profit / Loss per Trade (raw)')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('P/L')
    fig.tight_layout()
    fig.savefig(out / 'pl_per_trade.png', dpi=140)
    plt.close(fig)


def plot_market(market: pd.DataFrame, trades: pd.DataFrame, out: Path):
    if market.empty:
        return
    fig, ax = plt.subplots(figsize=(7,3.5))
    ax.plot(market['timestamp'], market['close'], label='Close', linewidth=1)
    # overlay trades
    if not trades.empty:
        merged = pd.merge_asof(trades.sort_values('timestamp'), market.sort_values('timestamp'), on='timestamp')
        buys = trades[trades['trade_type']=='buy']
        sells = trades[trades['trade_type']=='sell']
        ax.scatter(buys['timestamp'], buys['price'], marker='^', color='green', label='Buy', zorder=5)
        ax.scatter(sells['timestamp'], sells['price'], marker='v', color='red', label='Sell', zorder=5)
    ax.set_title('Market Close with Trades')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / 'market_with_trades.png', dpi=140)
    plt.close(fig)


def write_summary(trades: pd.DataFrame, market: pd.DataFrame, out: Path):
    lines = []
    if trades.empty:
        lines.append('No trades parsed.')
    else:
        wins = trades[trades['profit_loss']>0]
        losses = trades[trades['profit_loss']<0]
        wl_ratio = (len(wins)/max(1,len(trades))) * 100
        gross_pl = trades['profit_loss'].sum()
        avg_pl = trades['profit_loss'].mean()
        lines += [
            f'Trades: {len(trades)}',
            f'Win rate (% of >0 P/L): {wl_ratio:.1f}%',
            f'Gross P/L: {gross_pl:.2f}',
            f'Average P/L per trade: {avg_pl:.2f}'
        ]
    if not market.empty:
        lines.append(f'Market rows: {len(market)}')
    out.joinpath('live_summary.txt').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='analytics')
    parser.add_argument('--out', default='analytics/reports/live')
    args = parser.parse_args()

    base = Path(args.base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    trades, market = load_converted(base)
    plot_trades(trades, out)
    plot_market(market, trades, out)
    write_summary(trades, market, out)
    print(f'Live data visuals saved to {out}')

if __name__ == '__main__':
    main()
