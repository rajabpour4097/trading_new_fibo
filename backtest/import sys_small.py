import sys
import os
import pandas as pd
import numpy as np
from itertools import product
import glob
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swing import get_swing_points

class QuickParameterOptimizer:
    def __init__(self):
        self.results = []
        
    def load_data_files(self):
        """Load multiple CSV files for better testing"""
        data_files = []
        backtest_dir = "backtest/"
        
        csv_files = glob.glob(os.path.join(backtest_dir, "*.csv"))[:5]  # Use 5 files instead of 2
        
        for file_path in csv_files:
            try:
                data = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")
                if len(data) > 200:  # Minimum 200 rows
                    data['status'] = np.where(data['open'] > data['close'], 'bearish', 'bullish')
                    # Use first 2000 rows instead of 1000
                    data_files.append((file_path, data.head(2000)))
                    print(f"Loaded: {file_path} with {len(data.head(2000))} rows")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        return data_files
    
    def modified_get_legs(self, data, threshold):
        legs = []
        start_index = data.index[0]
        j = 0
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
        
    def fibonacci_retracement(self, start_price, end_price, fib_705, fib_90):
        fib_levels = {
            '0.0': start_price,
            '0.705': start_price + fib_705 * (end_price - start_price),
            '0.9': start_price + fib_90 * (end_price - start_price),
            '1.0': end_price
        }
        return fib_levels

    def backtest_strategy(self, data, threshold, fib_705, fib_90, risk_reward, window_size, min_swing_size, lookback_period):
        wins = 0
        losses = 0
        total_trades = 0
        
        start_index = 0
        step_size = max(1, window_size // 10)  # More granular stepping
        
        while start_index < len(data) - window_size - lookback_period:
            window_data = data[start_index:start_index+window_size]
            
            try:
                legs = self.modified_get_legs(window_data, threshold)
                
                if len(legs) >= min_swing_size:
                    swing_result = get_swing_points(window_data, legs[-3:])
                    
                    if swing_result and len(swing_result) == 2:
                        swing_type, is_swing = swing_result
                        
                        if is_swing:
                            total_trades += 1
                            entry = window_data.iloc[-1]['close']
                            
                            if swing_type == 'bullish':
                                stop = legs[-2]['end_value']
                                stop_distance = abs(entry - stop)
                                target = entry + risk_reward * stop_distance
                                
                                # Look ahead for trade result
                                future_start = start_index + window_size
                                future_end = min(future_start + lookback_period, len(data))
                                future_data = data[future_start:future_end]
                                
                                if len(future_data) > 0:
                                    hit_stop = any(future_data['low'] <= stop)
                                    hit_target = any(future_data['high'] >= target)
                                    
                                    if hit_stop and not hit_target:
                                        losses += 1
                                    elif hit_target and not hit_stop:
                                        wins += 1
                                    elif hit_stop and hit_target:
                                        # Check which was hit first
                                        stop_hit_idx = future_data[future_data['low'] <= stop].index[0] if any(future_data['low'] <= stop) else None
                                        target_hit_idx = future_data[future_data['high'] >= target].index[0] if any(future_data['high'] >= target) else None
                                        
                                        if stop_hit_idx and target_hit_idx:
                                            if stop_hit_idx < target_hit_idx:
                                                losses += 1
                                            else:
                                                wins += 1
                                        
                            elif swing_type == 'bearish':
                                stop = legs[-2]['end_value']
                                stop_distance = abs(stop - entry)
                                target = entry - risk_reward * stop_distance
                                
                                future_start = start_index + window_size
                                future_end = min(future_start + lookback_period, len(data))
                                future_data = data[future_start:future_end]
                                
                                if len(future_data) > 0:
                                    hit_stop = any(future_data['high'] >= stop)
                                    hit_target = any(future_data['low'] <= target)
                                    
                                    if hit_stop and not hit_target:
                                        losses += 1
                                    elif hit_target and not hit_stop:
                                        wins += 1
                                    elif hit_stop and hit_target:
                                        # Check which was hit first
                                        stop_hit_idx = future_data[future_data['high'] >= stop].index[0] if any(future_data['high'] >= stop) else None
                                        target_hit_idx = future_data[future_data['low'] <= target].index[0] if any(future_data['low'] <= target) else None
                                        
                                        if stop_hit_idx and target_hit_idx:
                                            if stop_hit_idx < target_hit_idx:
                                                losses += 1
                                            else:
                                                wins += 1
                                                
            except Exception as e:
                pass
                
            start_index += step_size
            
        return wins, losses, total_trades

    def optimize_parameters(self):
        """Test more comprehensive parameter combinations"""
        print("Loading data files...")
        data_files = self.load_data_files()
        
        if not data_files:
            print("No data files found!")
            return []
        
        # پارامترهای بهتر و کامل‌تر
        thresholds = [6]  # 6 values
        fib_705_levels = [0.705]  # 4 values - classical fib levels
        fib_90_levels = [0.9]  # 3 values
        risk_rewards = [1.2]  # 5 values
        window_sizes = [100]  # 4 values
        min_swing_sizes = [6]  # 4 values
        lookback_periods = [20]  # 3 values - how far to look ahead
        
        # Total combinations: 6×4×3×5×4×4×3 = 8,640 combinations
        
        param_combinations = list(product(
            thresholds, fib_705_levels, fib_90_levels, 
            risk_rewards, window_sizes, min_swing_sizes, lookback_periods
        ))
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        print("This will take approximately 30-60 minutes...")
        
        for i, (threshold, fib_705, fib_90, risk_reward, window_size, min_swing_size, lookback_period) in enumerate(param_combinations):
            if i % 100 == 0:  # Progress every 100 combinations
                progress = (i / len(param_combinations)) * 100
                print(f"Progress: {progress:.1f}% ({i}/{len(param_combinations)}) - "
                      f"Testing T:{threshold}, F:{fib_705}, RR:{risk_reward}")
            
            total_wins = 0
            total_losses = 0
            total_trades = 0
            
            for file_path, data in data_files:
                wins, losses, trades = self.backtest_strategy(
                    data, threshold, fib_705, fib_90, risk_reward, 
                    window_size, min_swing_size, lookback_period
                )
                total_wins += wins
                total_losses += losses
                total_trades += trades
            
            if total_trades >= 10:  # Minimum 10 trades for statistical significance
                win_rate = (total_wins / total_trades) * 100
                profit_factor = total_wins / max(total_losses, 1)
                
                # Calculate expected return (simplified)
                expected_return = (total_wins * risk_reward - total_losses) / total_trades if total_trades > 0 else 0
                
                self.results.append({
                    'threshold': threshold,
                    'fib_705': fib_705,
                    'fib_90': fib_90,
                    'risk_reward': risk_reward,
                    'window_size': window_size,
                    'min_swing_size': min_swing_size,
                    'lookback_period': lookback_period,
                    'total_trades': total_trades,
                    'wins': total_wins,
                    'losses': total_losses,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'expected_return': expected_return
                })
        
        # Sort by multiple criteria: win_rate and expected_return
        self.results.sort(key=lambda x: (x['win_rate'] * 0.7 + x['expected_return'] * 0.3), reverse=True)
        return self.results
    
    def save_results(self):
        """Save results to CSV file with enhanced analysis"""
        if self.results:
            df = pd.DataFrame(self.results)
            
            # Save all results
            output_file = "enhanced_optimization_results.csv"
            df.to_csv(output_file, index=False)
            print(f"\n✅ Full results saved to: {output_file}")
            
            # Save top 50 results
            top_file = "top_50_parameters.csv"
            df.head(50).to_csv(top_file, index=False)
            print(f"✅ Top 50 results saved to: {top_file}")
            
            # Print detailed top 15 results
            print(f"\n🏆 Top 15 Best Parameter Combinations:")
            print("="*120)
            print(f"{'#':<2} {'WinRate':<8} {'Trades':<7} {'PF':<6} {'ExpRet':<7} "
                  f"{'Thresh':<6} {'Fib705':<7} {'RR':<4} {'Window':<7} {'MinSwing':<8} {'Lookback'}")
            print("-"*120)
            
            for i, result in enumerate(self.results[:15]):
                print(f"{i+1:<2} {result['win_rate']:6.1f}%  {result['total_trades']:5d}   "
                      f"{result['profit_factor']:5.2f}  {result['expected_return']:6.2f}  "
                      f"{result['threshold']:5d}   {result['fib_705']:6.3f}  "
                      f"{result['risk_reward']:3.1f}  {result['window_size']:5d}    "
                      f"{result['min_swing_size']:6d}     {result['lookback_period']:6d}")
            
            # Statistical summary
            print(f"\n📊 Statistical Summary:")
            print(f"Total tested combinations: {len(self.results)}")
            print(f"Average win rate: {df['win_rate'].mean():.1f}%")
            print(f"Best win rate: {df['win_rate'].max():.1f}%")
            print(f"Average trades per combination: {df['total_trades'].mean():.0f}")
            print(f"Most active combination trades: {df['total_trades'].max()}")
            
        else:
            print("No results to save!")

def main():
    optimizer = QuickParameterOptimizer()
    
    print("Starting Enhanced Parameter Optimization...")
    print("This will test over 8,000 parameter combinations")
    print("Estimated time: 30-60 minutes depending on data size")
    print("-" * 60)
    
    start_time = datetime.now()
    
    results = optimizer.optimize_parameters()
    optimizer.save_results()
    
    end_time = datetime.now()
    print(f"\n⏱️ Optimization completed in {end_time - start_time}")
    print(f"📈 Found {len(results)} valid parameter combinations")

if __name__ == '__main__':
    main()