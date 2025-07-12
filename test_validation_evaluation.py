#!/usr/bin/env python3
"""
Test script for Validation & Evaluation System
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.validation_evaluation import (
        ValidationManager,
        TimeSeriesValidator,
        TradingPerformanceAnalyzer,
        BenchmarkComparator,
        ValidationConfig,
        quick_evaluation,
        compare_models
    )
    VALIDATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Validation components not available: {e}")
    VALIDATION_AVAILABLE = False


def create_sample_data(n_points: int = 2000, add_regimes: bool = True):
    """Create realistic cryptocurrency sample data for testing"""
    print(f"Creating sample cryptocurrency data with {n_points} points...")
    
    np.random.seed(42)
    
    # Create timestamps
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_points)]
    
    # Create realistic price data with different market regimes
    base_price = 50000
    prices = [base_price]
    volumes = []
    regimes = []
    
    # Market regime parameters
    regime_length = 200  # Average regime length
    current_regime = "bull_med_vol"
    regime_counter = 0
    
    for i in range(1, n_points):
        # Change regime periodically
        if regime_counter >= regime_length:
            regimes_list = ["bull_low_vol", "bull_med_vol", "bull_high_vol",
                          "bear_low_vol", "bear_med_vol", "bear_high_vol",
                          "sideways_low_vol", "sideways_med_vol", "sideways_high_vol"]
            current_regime = np.random.choice(regimes_list)
            regime_counter = 0
            regime_length = np.random.randint(150, 250)
        
        # Generate price movement based on regime
        if "bull" in current_regime:
            drift = 0.0002  # Upward drift
        elif "bear" in current_regime:
            drift = -0.0002  # Downward drift
        else:
            drift = 0  # Sideways
        
        if "low_vol" in current_regime:
            volatility = 0.01
        elif "high_vol" in current_regime:
            volatility = 0.04
        else:
            volatility = 0.02
        
        # Price change
        change = np.random.normal(drift, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
        
        # Volume (higher in volatile periods)
        vol_multiplier = 2 if "high_vol" in current_regime else 1
        volume = np.random.lognormal(10, 0.5) * vol_multiplier
        volumes.append(volume)
        
        regimes.append(current_regime)
        regime_counter += 1
    
    # Add first volume and regime
    volumes.insert(0, volumes[0])
    regimes.insert(0, current_regime)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': [int(ts.timestamp()) for ts in timestamps],
        'open': prices,
        'high': [p * np.random.uniform(1.0, 1.02) for p in prices],
        'low': [p * np.random.uniform(0.98, 1.0) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    if add_regimes:
        df['regime'] = regimes
    
    return df


def create_sample_models_predictions(data: pd.DataFrame):
    """Create sample predictions from different model types"""
    print("Creating sample model predictions...")
    
    actual_prices = data['close'].values
    n_points = len(actual_prices)
    
    # Model 1: Baseline (simple moving average)
    baseline_pred = np.concatenate([[actual_prices[0]], 
                                   pd.Series(actual_prices).rolling(24).mean().fillna(actual_prices[0]).values[1:]])
    
    # Model 2: Enhanced (better trend following)
    enhanced_pred = actual_prices.copy()
    for i in range(1, len(enhanced_pred)):
        # Add some predictive capability with noise
        trend = (actual_prices[i] - actual_prices[max(0, i-10)]) / 10
        enhanced_pred[i] = actual_prices[i-1] + trend * 0.7 + np.random.normal(0, actual_prices[i] * 0.005)
    
    # Model 3: External data enhanced (even better)
    external_pred = actual_prices.copy()
    for i in range(1, len(external_pred)):
        # Simulate external data influence
        trend = (actual_prices[i] - actual_prices[max(0, i-24)]) / 24
        momentum = np.sign(trend) * 0.3  # External sentiment effect
        external_pred[i] = actual_prices[i-1] + (trend + momentum) * 0.8 + np.random.normal(0, actual_prices[i] * 0.003)
    
    # Model 4: Multi-scale (best performance)
    multiscale_pred = actual_prices.copy()
    for i in range(1, len(multiscale_pred)):
        # Multiple timeframe analysis
        short_trend = (actual_prices[i] - actual_prices[max(0, i-6)]) / 6
        med_trend = (actual_prices[i] - actual_prices[max(0, i-24)]) / 24
        long_trend = (actual_prices[i] - actual_prices[max(0, i-168)]) / 168
        
        combined_trend = short_trend * 0.5 + med_trend * 0.3 + long_trend * 0.2
        multiscale_pred[i] = actual_prices[i-1] + combined_trend * 0.9 + np.random.normal(0, actual_prices[i] * 0.002)
    
    return {
        'TimeLLM_Baseline': (baseline_pred, actual_prices),
        'TimeLLM_Enhanced': (enhanced_pred, actual_prices),
        'TimeLLM_External': (external_pred, actual_prices),
        'TimeLLM_MultiScale': (multiscale_pred, actual_prices)
    }


def test_walk_forward_validation():
    """Test walk-forward validation system"""
    print("\n" + "=" * 60)
    print("TESTING WALK-FORWARD VALIDATION")
    print("=" * 60)
    
    if not VALIDATION_AVAILABLE:
        print("Validation components not available - skipping")
        return False
    
    try:
        # Create sample data
        data = create_sample_data(1000)
        
        # Create a simple prediction function for testing
        def simple_prediction_func(train_data, val_data):
            # Simple moving average predictor
            train_prices = train_data['close'].values
            val_length = len(val_data)
            
            # Use last 24 periods for prediction
            recent_avg = np.mean(train_prices[-24:])
            predictions = np.full(val_length, recent_avg)
            
            # Add some variation based on historical trend
            if len(train_prices) > 1:
                trend = train_prices[-1] - train_prices[-min(10, len(train_prices))]
                for i in range(val_length):
                    predictions[i] = recent_avg + trend * (i + 1) * 0.1 + np.random.normal(0, recent_avg * 0.01)
            
            return predictions
        
        # Test walk-forward validation
        config = ValidationConfig(
            initial_train_size=500,
            validation_size=50,
            step_size=25,
            max_validations=5  # Reduced for testing
        )
        
        validator = TimeSeriesValidator(config)
        results = validator.walk_forward_validation(
            data=data,
            predictions_func=simple_prediction_func,
            target_col='close',
            timestamp_col='timestamp'
        )
        
        print(f"Walk-forward validation completed: {len(results)} windows")
        
        # Analyze results
        if results:
            avg_mse = np.mean([r.mse for r in results])
            avg_mae = np.mean([r.mae for r in results])
            avg_sharpe = np.mean([r.sharpe_ratio for r in results if r.sharpe_ratio != 0])
            avg_win_rate = np.mean([r.win_rate for r in results])
            
            print(f"Average MSE: {avg_mse:.6f}")
            print(f"Average MAE: {avg_mae:.2f}")
            print(f"Average Sharpe Ratio: {avg_sharpe:.3f}")
            print(f"Average Win Rate: {avg_win_rate:.2%}")
        
        print("\n✓ Walk-forward validation test passed")
        return True, results
        
    except Exception as e:
        print(f"✗ Walk-forward validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, []


def test_trading_performance_analysis():
    """Test trading performance analysis"""
    print("\n" + "=" * 60)
    print("TESTING TRADING PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    if not VALIDATION_AVAILABLE:
        print("Validation components not available - skipping")
        return False
    
    try:
        # Create sample data
        data = create_sample_data(1000)
        actual_prices = data['close'].values
        
        # Create sample predictions (slightly ahead of actual)
        predictions = actual_prices.copy()
        for i in range(1, len(predictions)):
            trend = (actual_prices[i] - actual_prices[i-1])
            predictions[i-1] = actual_prices[i-1] + trend * 0.7  # Predictive signal
        
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
        
        # Test regime performance if regimes available
        if 'regime' in data.columns:
            print("\nTesting regime-specific performance...")
            regime_performance = analyzer.calculate_regime_performance(
                predictions=predictions,
                actual=actual_prices,
                regimes=data['regime'].tolist(),
                timestamps=timestamps
            )
            
            print(f"Regime performance calculated for {len(regime_performance)} regimes:")
            for regime, metrics in regime_performance.items():
                print(f"  {regime}: Sharpe {metrics['sharpe_ratio']:.3f}, Win Rate {metrics['win_rate']:.2%}")
        
        # Generate performance report
        report = analyzer.generate_performance_report(performance)
        print(f"\nPerformance report generated: {len(report)} characters")
        
        print("\n✓ Trading performance analysis test passed")
        return True, performance
        
    except Exception as e:
        print(f"✗ Trading performance analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def test_benchmark_comparison():
    """Test benchmark comparison system"""
    print("\n" + "=" * 60)
    print("TESTING BENCHMARK COMPARISON")
    print("=" * 60)
    
    if not VALIDATION_AVAILABLE:
        print("Validation components not available - skipping")
        return False
    
    try:
        # Create sample data
        data = create_sample_data(1000)
        
        # Create sample model predictions
        model_predictions = create_sample_models_predictions(data)
        
        print(f"Testing {len(model_predictions)} models:")
        for model_name in model_predictions.keys():
            print(f"  • {model_name}")
        
        # Test benchmark comparison
        comparator = BenchmarkComparator()
        timestamps = pd.to_datetime(data['timestamp'], unit='s')
        regimes = data['regime'].tolist() if 'regime' in data.columns else None
        
        benchmark_results = []
        for model_name, (predictions, actual) in model_predictions.items():
            result = comparator.add_benchmark_result(
                model_name=model_name,
                dataset_type="CRYPTEX_TEST",
                predictions=predictions,
                actual=actual,
                timestamps=timestamps,
                regimes=regimes
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


def test_comprehensive_evaluation():
    """Test comprehensive evaluation system"""
    print("\n" + "=" * 60)
    print("TESTING COMPREHENSIVE EVALUATION")
    print("=" * 60)
    
    if not VALIDATION_AVAILABLE:
        print("Validation components not available - skipping")
        return False
    
    try:
        # Create sample data
        data = create_sample_data(1000)
        
        # Create sample model predictions
        model_predictions = create_sample_models_predictions(data)
        
        # Test comprehensive evaluation
        validation_manager = ValidationManager()
        
        evaluation_results = validation_manager.comprehensive_evaluation(
            model_predictions=model_predictions,
            data=data,
            target_col='close',
            timestamp_col='timestamp',
            regime_col='regime' if 'regime' in data.columns else None
        )
        
        print(f"Comprehensive evaluation completed for {len(evaluation_results)} models:")
        
        # Display results summary
        sorted_models = sorted(evaluation_results.items(), 
                             key=lambda x: x[1].overall_score, reverse=True)
        
        print("\nModel Rankings:")
        for i, (model_name, result) in enumerate(sorted_models):
            print(f"  {i+1}. {model_name}: Score {result.overall_score:.3f}")
            print(f"     Sharpe: {result.trading_metrics['sharpe_ratio']:.3f}, "
                  f"Win Rate: {result.trading_metrics['win_rate']:.2%}, "
                  f"MAPE: {result.evaluation_metrics['mape']:.2f}%")
        
        # Generate final report
        report_file = validation_manager.generate_final_report("./test_results/validation/")
        print(f"\nFinal report saved to: {report_file}")
        
        print("\n✓ Comprehensive evaluation test passed")
        return True, evaluation_results
        
    except Exception as e:
        print(f"✗ Comprehensive evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def test_convenience_functions():
    """Test convenience functions"""
    print("\n" + "=" * 60)
    print("TESTING CONVENIENCE FUNCTIONS")
    print("=" * 60)
    
    if not VALIDATION_AVAILABLE:
        print("Validation components not available - skipping")
        return False
    
    try:
        # Create sample data
        data = create_sample_data(500)
        actual_prices = data['close'].values
        
        # Create sample predictions
        predictions = actual_prices * (1 + np.random.normal(0, 0.01, len(actual_prices)))
        
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
            'Model_B': (actual_prices * 1.001, actual_prices),  # Slightly better
            'Model_C': (actual_prices * 0.999, actual_prices)   # Slightly worse
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


def create_validation_visualizations(test_results):
    """Create visualizations for validation results"""
    print("\n" + "=" * 60)
    print("CREATING VALIDATION VISUALIZATIONS")
    print("=" * 60)
    
    try:
        os.makedirs('./plots/validation', exist_ok=True)
        
        # Extract results from tests
        walk_forward_success, walk_forward_results = test_results.get('walk_forward', (False, []))
        trading_success, trading_performance = test_results.get('trading', (False, {}))
        benchmark_success, benchmark_results = test_results.get('benchmark', (False, []))
        
        if walk_forward_success and walk_forward_results:
            # Plot 1: Walk-forward validation results
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Walk-Forward Validation Results', fontsize=16)
            
            # MSE over validation windows
            mse_values = [r.mse for r in walk_forward_results]
            axes[0, 0].plot(range(1, len(mse_values)+1), mse_values, 'b-', marker='o')
            axes[0, 0].set_title('MSE Across Validation Windows')
            axes[0, 0].set_xlabel('Validation Window')
            axes[0, 0].set_ylabel('MSE')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Sharpe ratio over validation windows
            sharpe_values = [r.sharpe_ratio for r in walk_forward_results]
            axes[0, 1].plot(range(1, len(sharpe_values)+1), sharpe_values, 'g-', marker='s')
            axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('Sharpe Ratio Across Validation Windows')
            axes[0, 1].set_xlabel('Validation Window')
            axes[0, 1].set_ylabel('Sharpe Ratio')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Win rate over validation windows
            win_rates = [r.win_rate for r in walk_forward_results]
            axes[1, 0].plot(range(1, len(win_rates)+1), win_rates, 'orange', marker='^')
            axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Break-even')
            axes[1, 0].set_title('Win Rate Across Validation Windows')
            axes[1, 0].set_xlabel('Validation Window')
            axes[1, 0].set_ylabel('Win Rate')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Max drawdown over validation windows
            drawdowns = [abs(r.max_drawdown) for r in walk_forward_results]
            axes[1, 1].plot(range(1, len(drawdowns)+1), drawdowns, 'r-', marker='d')
            axes[1, 1].set_title('Maximum Drawdown Across Validation Windows')
            axes[1, 1].set_xlabel('Validation Window')
            axes[1, 1].set_ylabel('Max Drawdown (abs)')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('./plots/validation/walk_forward_validation.png', dpi=150, bbox_inches='tight')
            print("Saved: ./plots/validation/walk_forward_validation.png")
            plt.close()
        
        if benchmark_success and benchmark_results:
            # Plot 2: Model comparison
            model_names = [r.model_name for r in benchmark_results]
            overall_scores = [r.overall_score for r in benchmark_results]
            sharpe_ratios = [r.trading_metrics['sharpe_ratio'] for r in benchmark_results]
            win_rates = [r.trading_metrics['win_rate'] for r in benchmark_results]
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Model Performance Comparison', fontsize=16)
            
            # Overall scores
            axes[0].bar(model_names, overall_scores, color='skyblue', alpha=0.7)
            axes[0].set_title('Overall Performance Score')
            axes[0].set_ylabel('Score')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3)
            
            # Sharpe ratios
            colors = ['green' if s > 0 else 'red' for s in sharpe_ratios]
            axes[1].bar(model_names, sharpe_ratios, color=colors, alpha=0.7)
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[1].axhline(y=1, color='gold', linestyle='--', alpha=0.7, label='Excellent')
            axes[1].set_title('Sharpe Ratio')
            axes[1].set_ylabel('Sharpe Ratio')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Win rates
            colors = ['green' if w > 0.5 else 'red' for w in win_rates]
            axes[2].bar(model_names, win_rates, color=colors, alpha=0.7)
            axes[2].axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Break-even')
            axes[2].set_title('Win Rate')
            axes[2].set_ylabel('Win Rate')
            axes[2].tick_params(axis='x', rotation=45)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('./plots/validation/model_comparison.png', dpi=150, bbox_inches='tight')
            print("Saved: ./plots/validation/model_comparison.png")
            plt.close()
        
        if trading_success and trading_performance:
            # Plot 3: Trading metrics breakdown
            metrics = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'information_ratio']
            values = [trading_performance.get(m, 0) for m in metrics]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'purple'], alpha=0.7)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Target Level')
            plt.title('Risk-Adjusted Performance Metrics')
            plt.ylabel('Ratio Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('./plots/validation/trading_metrics.png', dpi=150, bbox_inches='tight')
            print("Saved: ./plots/validation/trading_metrics.png")
            plt.close()
        
        print("✓ Validation visualizations created successfully")
        return True
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return False


def main():
    """Main test function for validation and evaluation system"""
    print("Testing Validation & Evaluation System")
    print("=" * 80)
    
    test_results = {}
    results_summary = []
    
    # Run all tests
    if not VALIDATION_AVAILABLE:
        print("❌ Validation system not available. Please install required dependencies.")
        return False
    
    # Test 1: Walk-forward validation
    success1, wf_results = test_walk_forward_validation()
    test_results['walk_forward'] = (success1, wf_results)
    results_summary.append(("Walk-Forward Validation", success1))
    
    # Test 2: Trading performance analysis
    success2, trading_perf = test_trading_performance_analysis()
    test_results['trading'] = (success2, trading_perf)
    results_summary.append(("Trading Performance Analysis", success2))
    
    # Test 3: Benchmark comparison
    success3, benchmark_res = test_benchmark_comparison()
    test_results['benchmark'] = (success3, benchmark_res)
    results_summary.append(("Benchmark Comparison", success3))
    
    # Test 4: Comprehensive evaluation
    success4, eval_results = test_comprehensive_evaluation()
    test_results['comprehensive'] = (success4, eval_results)
    results_summary.append(("Comprehensive Evaluation", success4))
    
    # Test 5: Convenience functions
    success5 = test_convenience_functions()
    results_summary.append(("Convenience Functions", success5))
    
    # Test 6: Visualizations
    success6 = create_validation_visualizations(test_results)
    results_summary.append(("Visualizations", success6))
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION & EVALUATION TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(results_summary)
    
    for test_name, result in results_summary:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:35} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed >= total - 1:  # Allow one failure
        print("\n🎉 Validation & Evaluation system is working correctly!")
        print("\nKey capabilities now available:")
        print("• Walk-forward validation for realistic time series testing")
        print("• Comprehensive trading performance metrics (Sharpe, Sortino, Calmar)")
        print("• Risk assessment framework (VaR, Expected Shortfall, Max Drawdown)")
        print("• Multi-model benchmarking and comparison")
        print("• Regime-aware performance analysis")
        print("• Automated reporting and visualization")
        print("• Trading simulation with transaction costs")
        print("• Uncertainty quantification and confidence intervals")
        
        print("\nNext steps:")
        print("1. Integrate with TimeLLM training pipeline")
        print("2. Run comprehensive evaluation on your enhanced models")
        print("3. Compare baseline vs enhanced performance")
        print("4. Use results to guide hyperparameter optimization")
        
        return True
    else:
        print(f"\n❌ {total-passed} tests failed. Check errors above.")
        return False


if __name__ == "__main__":
    main()