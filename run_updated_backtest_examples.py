#!/usr/bin/env python3
"""
Example script to run backtest with the updated features from main_metatrader.py

This script demonstrates how to run backtests with:
- Updated logic from main_metatrader.py
- Dynamic risk management stages
- Commission-based stages
- Two-touch validation
- Analytics logging
- Sell-only filter (for weekly testing)
"""

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

def run_backtest_example():
    """Run several example backtests with different configurations"""
    
    print("🚀 Running Updated Backtest Examples")
    print("=" * 60)
    
    # Base command
    base_cmd = [
        sys.executable,
        "backtest/run_backtest.py",
        "--symbol", "EURUSD",
        "--full-logic",  # Use the most advanced engine
        "--initial-balance", "10000",
        "--risk-pct", "0.01"
    ]
    
    examples = [
        {
            "name": "Weekly Test - Sell Only with Commission Stages",
            "cmd": base_cmd + [
                "--sell-only",  # Only SELL positions
                "--adv-commission-per-lot", "4.5",  # Updated commission
                "--adv-threshold", "7",  # Updated threshold
                "--mt5-month",  # Fetch last month data
                "--debug"  # Enable detailed logging
            ],
            "description": "Replicates the weekly test setup from main_metatrader.py"
        },
        {
            "name": "Full Logic with Two-Touch Validation",
            "cmd": base_cmd + [
                "--adv-commission-per-lot", "4.5",
                "--adv-threshold", "7", 
                "--mt5-month"
                # require_second_touch=True by default
            ],
            "description": "Full strategy with two-touch 0.705 validation"
        },
        {
            "name": "Single Touch Strategy (Less Strict)",
            "cmd": base_cmd + [
                "--single-touch",  # Allow single touch
                "--adv-commission-per-lot", "4.5",
                "--adv-threshold", "7",
                "--mt5-month"
            ],
            "description": "Strategy with single 0.705 touch requirement"
        },
        {
            "name": "Historical Range Test",
            "cmd": base_cmd + [
                "--from", "2024-08-01",
                "--to", "2024-08-31",
                "--adv-commission-per-lot", "4.5",
                "--adv-threshold", "7",
                "--sell-only"
            ],
            "description": "Test on specific date range (August 2024)"
        }
    ]
    
    print(f"📅 Available Examples ({len(examples)} total):")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print(f"   {example['description']}")
    
    print("\n" + "=" * 60)
    
    # Ask user which example to run
    try:
        choice = input("\nEnter example number to run (1-{}) or 'all' for all examples: ".format(len(examples)))
        
        if choice.lower() == 'all':
            selected_examples = examples
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(examples):
                selected_examples = [examples[idx]]
            else:
                print("❌ Invalid choice")
                return False
    except (ValueError, KeyboardInterrupt):
        print("\n❌ Invalid input or cancelled")
        return False
    
    # Run selected examples
    for example in selected_examples:
        print(f"\n🔄 Running: {example['name']}")
        print("-" * 50)
        print(f"📝 Description: {example['description']}")
        print(f"🔧 Command: {' '.join(example['cmd'])}")
        print("")
        
        try:
            result = subprocess.run(
                example['cmd'],
                cwd=Path(__file__).parent,
                capture_output=False,  # Show output in real-time
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ {example['name']} completed successfully")
            else:
                print(f"❌ {example['name']} failed with return code {result.returncode}")
                
        except Exception as e:
            print(f"❌ Error running {example['name']}: {e}")
        
        print("-" * 50)
    
    return True

def show_results_info():
    """Show information about where to find results"""
    print("\n📊 Results Information:")
    print("=" * 40)
    print("📁 Backtest results are saved in:")
    print("   - backtest/results/ (CSV trades and JSON summary)")
    print("📁 Analytics logs are saved in:")
    print("   - trading-analytics-logger/data/raw/signals/")
    print("   - trading-analytics-logger/data/raw/events/")
    print("\n📈 Key Metrics to Check:")
    print("   - Total trades and win rate")
    print("   - Net R and average R per trade")
    print("   - Profit factor")
    print("   - Stage utilization (commission, 0.5R, 1.0R, 1.5R)")
    print("   - Skip reasons (why trades were not taken)")

def main():
    """Main function"""
    print("Updated Backtest Runner")
    print("Based on main_metatrader.py logic")
    print("With dynamic risk management and analytics")
    print("")
    
    # Show what's new
    print("🆕 New Features:")
    print("   ✅ Dynamic risk management stages")
    print("   ✅ Commission-based stage management")
    print("   ✅ Two-touch 0.705 validation")
    print("   ✅ Enhanced stop loss placement logic")
    print("   ✅ Analytics integration")
    print("   ✅ Sell-only filter for weekly testing")
    print("   ✅ Updated default parameters (threshold=7, commission=4.5)")
    print("")
    
    success = run_backtest_example()
    
    if success:
        show_results_info()
        print("\n🎉 Backtest examples completed!")
        print("💡 Tip: Compare results with live trading performance")
    else:
        print("\n❌ Backtest execution failed")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Cancelled by user")
        sys.exit(1)
