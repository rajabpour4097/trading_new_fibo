import sys
import os
import pandas as pd
import numpy as np
from itertools import product
import glob
from datetime import datetime

from swing import get_swing_points

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ParameterOptimizer:
    def __init__(self):
        self.results = []
        
    def load_data_files(self):
        """Load all available CSV data files"""
        data_files = []
        backtest_dir = "backtest/"
        
        # Find all CSV files in backtest directory
        csv_files = glob.glob(os.path.join(backtest_dir, "*.csv"))
        
        for file_path in csv_files:
            try:
                data = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")
                if len(data) > 100:  # Only use files with sufficient data
                    data['status'] = np.where(data['open'] > data['close'], 'bearish', 'bullish')
                    data_files.append((file_path, data))
                    print(f"Loaded: {file_path} with {len(data)} rows")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        return data_files
    
    def modified_fibonacci_retracement(self, start_price, end_price, fib_705, fib_90):
        """Modified fibonacci with custom levels"""
        fib_levels = {
            '0.0': start_price,
            '0.705': start_price + fib_705 * (end_price - start_price),
            '0.9': start_price + fib_90 * (end_price - start_price),
            '1.0': end_price
        }
        return fib_levels
    
    def modified_get_legs(self, data, threshold):
        """Modified get_legs with custom threshold"""
        legs = []
        start_index = data.index[0]
        j = 0
        last_start_price = None
        i = 1

        while i < len(data):
            current_is_bullish = data['close'].iloc[i] >= data['open'].iloc[i]
            
            if i == 1:
                legs.append({
                    'start': start_index,
                    'start_value': data['close'].iloc[0],
                    'direction': 'up' if current_is_bullish else 'down',
                    'end': data.index[i],
                    'end_value': data['close'].iloc[i]
                })
                j += 1
            else:
                price_diff = abs(data['close'].iloc[i] - legs[j-1]['end_value']) * 100000
                
                if price_diff >= threshold:
                    current_direction = 'up' if data['close'].iloc[i] > legs[j-1]['end_value'] else 'down'
                    
                    if legs[j-1]['direction'] != current_direction:
                        legs.append({
                            'start': legs[j-1]['end'],
                            'start_value': legs[j-1]['end_value'],
                            'direction': current_direction,
                            'end': data.index[i],
                            'end_value': data['close'].iloc[i]
                        })
                        j += 1
                    else:
                        legs[j-1]['end'] = data.index[i]
                        legs[j-1]['end_value'] = data['close'].iloc[i]
            i += 1
            
        return legs
    
    def backtest_strategy(self, data, params):
        """Run backtest with specific parameters"""
        threshold, fib_705, fib_90, risk_reward, window_size, min_swing_size = params
        
        wins = 0
        losses = 0
        trades = []
        start_index = 0
        
        while start_index < len(data) - window_size - 20:
            window_data = data[start_index:start_index + window_size]
            
            if len(window_data) < 10:
                start_index += 1
                continue
                
            legs = self.modified_get_legs(window_data, threshold)
            
            if len(legs) >= 3:
                swing_result = get_swing_points(window_data, legs[-3:])
                
                if swing_result and len(swing_result) == 2:
                    swing_type, is_swing = swing_result
                    
                    if is_swing and swing_type in ['bullish', 'bearish']:
                        entry = window_data.iloc[-1]['close']
                        stop = legs[-2]['end_value']
                        diff = abs(entry - stop)
                        
                        # Check minimum swing size
                        if diff * 100000 < min_swing_size:
                            start_index += 1
                            continue
                        
                        if swing_type == 'bullish':
                            target = entry + risk_reward * diff
                        else:
                            target = entry - risk_reward * diff
                        
                        # Look for trade outcome in next 20 candles
                        future_data = data[window_data.index[-1]:].iloc[1:21]
                        
                        trade_outcome = None
                        for _, row in future_data.iterrows():
                            if swing_type == 'bullish':
                                if row['low'] <= stop:
                                    trade_outcome = 'loss'
                                    break
                                elif row['high'] >= target:
                                    trade_outcome = 'win'
                                    break
                            else:
                                if row['high'] >= stop:
                                    trade_outcome = 'loss'
                                    break
                                elif row['low'] <= target:
                                    trade_outcome = 'win'
                                    break
                        
                        if trade_outcome:
                            trades.append({
                                'type': swing_type,
                                'entry': entry,
                                'stop': stop,
                                'target': target,
                                'outcome': trade_outcome
                            })
                            
                            if trade_outcome == 'win':
                                wins += 1
                            else:
                                losses += 1
            
            start_index += 1
        
        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'wins': wins,
            'losses': losses,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'trades': trades
        }
    
    def optimize_parameters(self):
        """Test different parameter combinations"""
        print("Loading data files...")
        data_files = self.load_data_files()
        
        if not data_files:
            print("No data files found!")
            return
        
        # Parameter ranges to test
        thresholds = [4, 5, 6, 7, 8, 10]
        fib_705_levels = [0.618, 0.705, 0.786]
        fib_90_levels = [0.85, 0.9, 0.95]
        risk_rewards = [1.0, 1.2, 1.5, 2.0]
        window_sizes = [40, 50, 60, 80]
        min_swing_sizes = [4, 5, 6]
        
        param_combinations = list(product(
            thresholds, fib_705_levels, fib_90_levels, 
            risk_rewards, window_sizes, min_swing_sizes
        ))
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        for i, params in enumerate(param_combinations):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(param_combinations)} ({i/len(param_combinations)*100:.1f}%)")
            
            total_wins = 0
            total_losses = 0
            total_trades = 0
            file_results = []
            
            # Test on all data files
            for file_path, data in data_files:
                result = self.backtest_strategy(data, params)
                total_wins += result['wins']
                total_losses += result['losses']
                total_trades += result['total_trades']
                
                file_results.append({
                    'file': os.path.basename(file_path),
                    'win_rate': result['win_rate'],
                    'trades': result['total_trades']
                })
            
            overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
            
            # Only save results with minimum trades
            if total_trades >= 10:
                self.results.append({
                    'threshold': params[0],
                    'fib_705': params[1],
                    'fib_90': params[2],
                    'risk_reward': params[3],
                    'window_size': params[4],
                    'min_swing_size': params[5],
                    'total_wins': total_wins,
                    'total_losses': total_losses,
                    'total_trades': total_trades,
                    'win_rate': overall_win_rate,
                    'file_results': file_results
                })
        
        # Sort results by win rate
        self.results.sort(key=lambda x: x['win_rate'], reverse=True)
        
        return self.results
    
    def print_top_results(self, top_n=10):
        """Print top performing parameter combinations"""
        print(f"\n{'='*80}")
        print(f"TOP {top_n} PARAMETER COMBINATIONS")
        print(f"{'='*80}")
        
        for i, result in enumerate(self.results[:top_n]):
            print(f"\n#{i+1} - Win Rate: {result['win_rate']:.2f}%")
            print(f"Threshold: {result['threshold']}")
            print(f"Fib 0.705 Level: {result['fib_705']}")
            print(f"Fib 0.9 Level: {result['fib_90']}")
            print(f"Risk/Reward: {result['risk_reward']}")
            print(f"Window Size: {result['window_size']}")
            print(f"Min Swing Size: {result['min_swing_size']}")
            print(f"Total Trades: {result['total_trades']} (Wins: {result['total_wins']}, Losses: {result['total_losses']})")
            print("-" * 50)
    
    def save_results_to_csv(self, filename="optimization_results.csv"):
        """Save all results to CSV file"""
        if not self.results:
            print("No results to save!")
            return
            
        df = pd.DataFrame(self.results)
        df = df.drop('file_results', axis=1)  # Remove nested data for CSV
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

def main():
    optimizer = ParameterOptimizer()
    
    print("Starting parameter optimization...")
    start_time = datetime.now()
    
    results = optimizer.optimize_parameters()
    
    end_time = datetime.now()
    print(f"\nOptimization completed in {end_time - start_time}")
    
    if results:
        optimizer.print_top_results(15)
        optimizer.save_results_to_csv()
        
        print(f"\nTotal parameter combinations tested: {len(results)}")
        print(f"Best win rate achieved: {results[0]['win_rate']:.2f}%")
    else:
        print("No valid results found!")

if __name__ == '__main__':
    main()