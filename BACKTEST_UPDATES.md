# Updated Backtest Engine - Based on main_metatrader.py

## Overview

The backtest engine has been updated to match the latest changes in `main_metatrader.py`, including dynamic risk management, commission-based stages, enhanced logging, and improved trading logic.

## Key Updates Applied

### 1. Trading Logic Improvements

- **Enhanced Fibonacci Setup**: Updated fibonacci retracement logic to match main_metatrader.py
- **Two-Touch Validation**: Requires two opposite-candle touches of 0.705 level before arming trade
- **Improved Stop Loss Placement**: Better logic for choosing between 0.9 and 1.0 levels
- **Direction Detection**: Enhanced swing type detection and validation

### 2. Dynamic Risk Management Stages

The backtest now includes all 4 stages from the live system:

#### Stage 1: Commission Cover
- **Trigger**: When floating profit ≥ total commission cost
- **Action**: Move SL to commission break-even point
- **Purpose**: Lock in commission recovery

#### Stage 2: Half R (0.5R)
- **Trigger**: When profit reaches 0.5R
- **Action**: Move SL to +0.5R, keep TP at 1.2R
- **Purpose**: Secure partial profit

#### Stage 3: One R (1.0R)
- **Trigger**: When profit reaches 1.0R  
- **Action**: Move SL to +1.0R, extend TP to 1.5R
- **Purpose**: Risk-free position with higher target

#### Stage 4: One and Half R (1.5R)
- **Trigger**: When profit reaches 1.5R
- **Action**: Move SL to +1.5R, extend TP to 2.0R
- **Purpose**: Trail profits for maximum gain

### 3. Updated Configuration Defaults

```python
# New defaults matching main_metatrader.py
threshold = 7                    # Stricter noise filtering
commission_per_lot = 4.5        # Realistic commission
require_second_touch = True     # Two-touch validation
sell_only = False              # Can be enabled for testing
```

### 4. Analytics Integration

- **Signal Logging**: Every trade signal is logged for analysis
- **Position Events**: All SL/TP modifications are tracked
- **Stage Tracking**: Monitor which stages are most effective
- **Skip Reasons**: Track why trades were not taken

## Usage Examples

### Basic Full Logic Backtest
```bash
python backtest/run_backtest.py --full-logic --mt5-month
```

### Weekly Test (Sell Only)
```bash
python backtest/run_backtest.py --full-logic --sell-only --mt5-month
```

### Custom Date Range
```bash
python backtest/run_backtest.py --full-logic --from 2024-08-01 --to 2024-08-31
```

### Single Touch Strategy
```bash
python backtest/run_backtest.py --full-logic --single-touch --mt5-month
```

### Debug Mode with Detailed Logs
```bash
python backtest/run_backtest.py --full-logic --debug --mt5-month
```

## Command Line Options

### New Options Added:
- `--sell-only`: Only allow SELL positions (for testing)
- `--single-touch`: Allow single 0.705 touch (less strict)
- `--debug`: Enable detailed debug logging

### Updated Defaults:
- `--adv-threshold`: Default changed from 6 to 7
- `--adv-commission-per-lot`: Default changed from 0.0 to 4.5

## Output Files

### Backtest Results
- `backtest/results/{symbol}_trades.csv` - Individual trade details
- `backtest/results/{symbol}_summary.json` - Performance summary
- `backtest/results/{symbol}_events.csv` - Position management events

### Analytics Logs
- `trading-analytics-logger/data/raw/signals/{symbol}_signals_{date}.csv`
- `trading-analytics-logger/data/raw/events/{symbol}_position_events_{date}.csv`

## Key Metrics to Monitor

### Performance Metrics
- **Total Trades**: Number of positions opened
- **Win Rate**: Percentage of profitable trades  
- **Net R**: Total R gained/lost across all trades
- **Average R**: Mean R per trade
- **Profit Factor**: Gross profit / Gross loss

### Risk Management Metrics
- **Stage Utilization**: How often each stage was triggered
- **Commission Recovery**: Frequency of commission stage hits
- **Trail Effectiveness**: How often extended TPs were hit

### Skip Reasons Analysis
- **no_fib**: No fibonacci levels established
- **not_true_position**: Two-touch validation not met
- **invalid_sl**: Stop loss placement issues
- **vol_zero**: Volume calculation failed
- **sell_only_filter**: Filtered by sell-only mode

## Validation Against Live System

The backtest engine now closely replicates the live `main_metatrader.py` behavior:

1. ✅ Same fibonacci calculation logic
2. ✅ Same two-touch validation requirement
3. ✅ Same stop loss placement rules
4. ✅ Same dynamic risk management stages
5. ✅ Same commission handling
6. ✅ Same entry/exit logic

## Testing Workflow

1. **Run Test Script**: `python test_updated_backtest.py`
2. **Run Example Backtests**: `python run_updated_backtest_examples.py`
3. **Compare with Live Results**: Check if backtest matches recent live performance
4. **Analyze Logs**: Review analytics data for insights

## Troubleshooting

### Common Issues

**Import Errors**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

**MT5 Connection**: For live data fetching, ensure MT5 is running
```bash
# Use CSV data if MT5 unavailable
python backtest/run_backtest.py --full-logic backtest/EURUSD_data.csv
```

**Analytics Warnings**: File/directory conflicts are handled automatically
```
[analytics.hooks] Warning: 'market' exists as a file. Using 'market_dir' for logging.
```

## Performance Optimization

- Use `--quiet-ext` to reduce external function output
- Use specific date ranges instead of `--mt5-month` for faster execution
- Enable `--debug` only when needed (increases log size)
- Use `--no-tick-path` if tick simulation is not required

## Future Enhancements

Potential areas for further development:

1. **ML Filter Integration**: Add machine learning trade filtering
2. **Multi-Symbol Support**: Run backtests across multiple pairs
3. **Portfolio Analysis**: Combine results across strategies
4. **Real-time Validation**: Compare backtest vs live performance automatically
5. **Risk Metrics**: Add more sophisticated risk analysis

---

## Quick Start

For immediate testing of the updated backtest:

```bash
# 1. Test the setup
python test_updated_backtest.py

# 2. Run a quick example
python run_updated_backtest_examples.py

# 3. Select option 1 for "Weekly Test - Sell Only"
```

This will run a complete backtest with all the new features and generate comprehensive results for analysis.
