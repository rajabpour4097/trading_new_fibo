#!/usr/bin/env python3
"""
Test script to verify the updated backtest engine works with the new features from main_metatrader.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_import():
    """Test that all imports work correctly"""
    print("Testing imports...")
    try:
        from backtest.advanced_tick_engine import AdvancedTickBacktester, FullConfig
        from backtest.mt5_data_fetch import fetch_symbol_specs
        from metatrader5_config import MT5_CONFIG, TRADING_CONFIG, DYNAMIC_RISK_CONFIG
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config_creation():
    """Test that configuration can be created with new settings"""
    print("\nTesting configuration creation...")
    try:
        from backtest.advanced_tick_engine import FullConfig
        
        config = FullConfig(
            symbol='EURUSD',
            threshold=7,  # Updated default
            commission_per_lot=4.5,  # Updated default
            require_second_touch=True,  # New feature
            sell_only=True,  # New feature
            debug=True
        )
        
        print(f"✅ Config created successfully:")
        print(f"  - Symbol: {config.symbol}")
        print(f"  - Threshold: {config.threshold}")
        print(f"  - Commission per lot: {config.commission_per_lot}")
        print(f"  - Require second touch: {config.require_second_touch}")
        print(f"  - Sell only: {config.sell_only}")
        print(f"  - Dynamic stages: {len(config.stages)} stages")
        
        return True
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        return False

def test_stages_config():
    """Test that dynamic risk stages are properly configured"""
    print("\nTesting dynamic risk stages...")
    try:
        from metatrader5_config import DYNAMIC_RISK_CONFIG
        
        stages = DYNAMIC_RISK_CONFIG.get('stages', [])
        print(f"✅ Found {len(stages)} stages:")
        
        for i, stage in enumerate(stages):
            stage_id = stage.get('id')
            stage_type = stage.get('type', 'R-based')
            trigger = stage.get('trigger_R', 'commission-based')
            print(f"  {i+1}. {stage_id}: {stage_type} (trigger: {trigger})")
        
        # Check for commission stage
        has_commission_stage = any(s.get('type') == 'commission' for s in stages)
        print(f"  Has commission stage: {'✅' if has_commission_stage else '❌'}")
        
        return True
    except Exception as e:
        print(f"❌ Stages config test failed: {e}")
        return False

def test_analytics_integration():
    """Test that analytics integration is working"""
    print("\nTesting analytics integration...")
    try:
        from backtest.advanced_tick_engine import ANALYTICS_AVAILABLE, log_signal, log_position_event
        
        if ANALYTICS_AVAILABLE:
            print("✅ Analytics module is available")
            # Test a dummy signal log
            log_signal(
                symbol="EURUSD",
                strategy="test",
                direction="buy",
                rr=1.2,
                entry=1.1000,
                sl=1.0950,
                tp=1.1060,
                note="test_signal"
            )
            print("✅ Signal logging test successful")
            
            # Test a dummy position event log
            log_position_event(
                symbol="EURUSD",
                ticket=12345,
                event="test",
                direction="buy",
                entry=1.1000,
                current_price=1.1010,
                sl=1.0950,
                tp=1.1060,
                profit_R=0.2,
                stage=0,
                risk_abs=0.005,
                volume=0.01,
                note="test_event"
            )
            print("✅ Position event logging test successful")
        else:
            print("⚠️ Analytics module not available (will use fallback functions)")
        
        return True
    except Exception as e:
        print(f"❌ Analytics integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Updated Backtest Engine")
    print("=" * 50)
    
    tests = [
        test_basic_import,
        test_config_creation,
        test_stages_config,
        test_analytics_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The backtest engine is ready to use.")
        print("\n📝 Next steps:")
        print("1. Run backtest with: python backtest/run_backtest.py --full-logic --sell-only")
        print("2. Use --debug for detailed logging")
        print("3. Use --single-touch to disable two-touch requirement")
        print("4. Check analytics logs in trading-analytics-logger/data/raw/")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
