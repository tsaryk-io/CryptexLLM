#!/usr/bin/env python3
"""
Simple test script for Validation & Evaluation System
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.validation_evaluation import (
        TradingPerformanceAnalyzer,
        BenchmarkComparator,
        quick_evaluation,
        compare_models
    )
    VALIDATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Validation components not available: {e}")
    VALIDATION_AVAILABLE = False


def create_simple_test_data(n_points: int = 100):
    """Create simple test data"""
    print(f"Creating simple test data with {n_points} points...")
    
    np.random.seed(42)
    
    # Create realistic price data
    base_price = 50000
    prices = [base_price]
    
    for i in range(1, n_points):
        change = np.random.normal(0.0001, 0.02)  # Small drift, moderate volatility
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create DataFrame
    timestamps = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(n_points)]
    
    df = pd.DataFrame({
        'timestamp': [int(ts.timestamp()) for ts in timestamps],
        'close': prices
    })
    
    return df


def test_trading_performance():
    """Test trading performance analysis"""
    print("\n" + "=" * 50)
    print("TESTING TRADING PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    if not VALIDATION_AVAILABLE:
        print("Validation components not available - skipping")
        return False
    
    try:
        # Create simple test data
        data = create_simple_test_data(200)
        actual_prices = data['close'].values
        
        # Create simple predictions (slightly ahead of actual)
        predictions = actual_prices.copy()
        for i in range(1, len(predictions)):
            # Add predictive signal
            trend = actual_prices[i] - actual_prices[i-1]
            predictions[i-1] = actual_prices[i-1] + trend * 0.6 + np.random.normal(0, actual_prices[i-1] * 0.002)
        
        # Test trading performance analysis
        analyzer = TradingPerformanceAnalyzer()
        timestamps = pd.to_datetime(data['timestamp'], unit='s')
        
        performance = analyzer.analyze_trading_performance(
            predictions=predictions,
            actual=actual_prices,
            timestamps=timestamps
        )
        
        print("Trading Performance Metrics:")
        for metric, value in performance.items():
            if isinstance(value, float):
                if 'ratio' in metric or 'return' in metric:
                    print(f"  {metric}: {value:.4f}")
                elif 'rate' in metric:
                    print(f"  {metric}: {value:.2%}")
                else:
                    print(f"  {metric}: {value:.6f}")
            else:
                print(f"  {metric}: {value}")
        
        print("\n✓ Trading performance analysis test passed")
        return True, performance
        
    except Exception as e:
        print(f"✗ Trading performance analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def test_benchmark_comparison():
    """Test benchmark comparison system"""
    print("\n" + "=" * 50)
    print("TESTING BENCHMARK COMPARISON")
    print("=" * 50)
    
    if not VALIDATION_AVAILABLE:
        print("Validation components not available - skipping")
        return False
    
    try:
        # Create test data
        data = create_simple_test_data(200)
        actual_prices = data['close'].values
        
        # Create multiple model predictions
        models = {}
        
        # Model 1: Baseline (simple moving average)
        baseline_pred = np.zeros_like(actual_prices)
        for i in range(1, len(actual_prices)):
            if i < 10:
                baseline_pred[i] = actual_prices[i-1]
            else:
                baseline_pred[i] = np.mean(actual_prices[i-10:i])
        models['TimeLLM_Baseline'] = (baseline_pred, actual_prices)
        
        # Model 2: Enhanced (better trend following)
        enhanced_pred = np.zeros_like(actual_prices)
        for i in range(1, len(actual_prices)):
            if i < 5:
                enhanced_pred[i] = actual_prices[i-1]
            else:
                trend = (actual_prices[i-1] - actual_prices[i-5]) / 5
                enhanced_pred[i] = actual_prices[i-1] + trend * 0.8 + np.random.normal(0, actual_prices[i-1] * 0.003)
        models['TimeLLM_Enhanced'] = (enhanced_pred, actual_prices)
        
        # Model 3: Advanced (even better)
        advanced_pred = np.zeros_like(actual_prices)
        for i in range(1, len(actual_prices)):
            if i < 10:
                advanced_pred[i] = actual_prices[i-1]
            else:
                short_trend = (actual_prices[i-1] - actual_prices[i-3]) / 3
                long_trend = (actual_prices[i-1] - actual_prices[i-10]) / 10
                combined_trend = short_trend * 0.7 + long_trend * 0.3
                advanced_pred[i] = actual_prices[i-1] + combined_trend * 0.9 + np.random.normal(0, actual_prices[i-1] * 0.002)
        models['TimeLLM_Advanced'] = (advanced_pred, actual_prices)
        
        print(f"Testing {len(models)} models:")
        for model_name in models.keys():
            print(f"  • {model_name}")
        
        # Test benchmark comparison
        comparator = BenchmarkComparator()
        timestamps = pd.to_datetime(data['timestamp'], unit='s')
        
        benchmark_results = []
        for model_name, (predictions, actual) in models.items():
            result = comparator.add_benchmark_result(
                model_name=model_name,
                dataset_type="CRYPTEX_TEST",
                predictions=predictions,
                actual=actual,
                timestamps=timestamps
            )
            benchmark_results.append(result)
        
        print(f"\nBenchmark results for {len(benchmark_results)} models:")
        for result in benchmark_results:
            print(f"  {result.model_name}: Score {result.overall_score:.3f}, "
                  f"Sharpe {result.trading_metrics['sharpe_ratio']:.3f}")
        
        # Generate comparison report
        comparison_report = comparator.generate_comparison_report()
        print(f"\nComparison report generated: {len(comparison_report)} characters")
        
        # Test saving results
        os.makedirs("./test_results/", exist_ok=True)
        results_file, report_file = comparator.save_benchmark_results("./test_results/")
        print(f"Results saved to: {results_file}")
        print(f"Report saved to: {report_file}")
        
        print("\n✓ Benchmark comparison test passed")
        return True, benchmark_results
        
    except Exception as e:
        print(f"✗ Benchmark comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, []


def test_convenience_functions():
    """Test convenience functions"""
    print("\n" + "=" * 50)
    print("TESTING CONVENIENCE FUNCTIONS")
    print("=" * 50)
    
    if not VALIDATION_AVAILABLE:
        print("Validation components not available - skipping")
        return False
    
    try:
        # Create test data
        data = create_simple_test_data(100)
        actual_prices = data['close'].values
        
        # Create predictions
        predictions = actual_prices * (1 + np.random.normal(0, 0.005, len(actual_prices)))
        
        # Test quick evaluation
        print("Testing quick_evaluation function...")
        quick_results = quick_evaluation(predictions, actual_prices, "Test_Model")
        
        print(f"Quick evaluation results: {len(quick_results)} metrics")
        print(f"  Sharpe Ratio: {quick_results['sharpe_ratio']:.3f}")
        print(f"  Win Rate: {quick_results['win_rate']:.2%}")
        
        # Test compare_models function
        print("\nTesting compare_models function...")
        model_results = {
            'Model_A': (predictions, actual_prices),
            'Model_B': (actual_prices * 1.002, actual_prices),  # Slightly better
            'Model_C': (actual_prices * 0.998, actual_prices)   # Slightly worse
        }
        
        comparison_report = compare_models(model_results)
        print(f"Model comparison report generated: {len(comparison_report)} characters")
        
        print("\n✓ Convenience functions test passed")
        return True
        
    except Exception as e:
        print(f"✗ Convenience functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("Simple Validation & Evaluation System Test")
    print("=" * 60)
    
    test_results = []
    
    if not VALIDATION_AVAILABLE:
        print("Validation system not available. Please install required dependencies.")
        return False
    
    # Test 1: Trading performance analysis
    success1, trading_perf = test_trading_performance()
    test_results.append(("Trading Performance Analysis", success1))
    
    # Test 2: Benchmark comparison
    success2, benchmark_res = test_benchmark_comparison()
    test_results.append(("Benchmark Comparison", success2))
    
    # Test 3: Convenience functions
    success3 = test_convenience_functions()
    test_results.append(("Convenience Functions", success3))
    
    # Summary
    print("\n" + "=" * 60)
    print("SIMPLE VALIDATION TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:35} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nValidation & Evaluation system core functionality is working!")
        print("\nKey capabilities validated:")
        print("• Trading performance metrics calculation")
        print("• Risk-adjusted performance analysis")
        print("• Multi-model benchmarking and comparison")
        print("• Convenience functions for quick evaluation")
        return True
    else:
        print(f"\n{total-passed} tests failed. Check errors above.")
        return False


if __name__ == "__main__":
    main()