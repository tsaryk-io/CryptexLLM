#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import os
import sys
from itertools import product
import json
from datetime import datetime

# Import the backtesting functions from trader_enhanced
sys.path.append('.')
from trader_enhanced import (
    backtest_enhanced, calculate_buy_hold_benchmark, 
    strat_directional_accuracy, strat_predicted_enhanced,
    strat_hybrid_enhanced
)

def optimize_deepseek_directional(csv_file, output_dir='./optimization_results'):
    """
    Comprehensive parameter optimization for DEEPSEEK Directional strategy
    """
    
    print("="*80)
    print("DEEPSEEK DIRECTIONAL STRATEGY OPTIMIZATION")
    print("="*80)
    
    # Load the DEEPSEEK prediction data
    try:
        df = pd.read_csv(csv_file, header=0)
        print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f'Error reading {csv_file}: {e}')
        sys.exit(1)
    
    # Clean column names and prepare data
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Handle timestamp/date column
    date_col = next((c for c in df.columns if 'date' in c or 'time' in c), None)
    if not date_col:
        print('No date/time column found.')
        sys.exit(1)
    
    # Parse dates
    try:
        col = df[date_col]
        if pd.api.types.is_numeric_dtype(col):
            df['dt'] = pd.to_datetime(col, unit='s', errors='raise')
        else:
            df['dt'] = pd.to_datetime(col, errors='raise')
    except Exception as e:
        print(f'Could not parse {date_col}: {e}')
        sys.exit(1)
    
    df.set_index('dt', inplace=True)
    
    # Verify required columns
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            print(f'Missing required column: {col}')
            sys.exit(1)
    
    # Clean prediction data
    pred_cols = [col for col in df.columns if 'predicted' in col]
    print(f"Available prediction columns: {pred_cols}")
    
    for col in pred_cols:
        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    # Remove rows with missing close prices
    df = df.dropna(subset=['close'])
    print(f"Data ready: {len(df)} rows")
    
    # Parameter ranges to test
    param_grid = {
        'confidence_threshold': [0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
        'fee': [0.0005, 0.001, 0.002],
        'change_threshold': [0.01, 0.015, 0.02, 0.025, 0.03],
        'prediction_col': ['close_predicted_1', 'close_predicted_2', 'close_predicted_3'],
        'initial_capital': [1000.0]  # Keep constant for comparison
    }
    
    # Filter available prediction columns
    available_pred_cols = [col for col in param_grid['prediction_col'] if col in df.columns]
    param_grid['prediction_col'] = available_pred_cols
    
    print(f"\\nTesting parameter combinations:")
    print(f"  Confidence thresholds: {param_grid['confidence_threshold']}")
    print(f"  Trading fees: {param_grid['fee']}")
    print(f"  Change thresholds: {param_grid['change_threshold']}")
    print(f"  Prediction columns: {param_grid['prediction_col']}")
    
    # Calculate total combinations
    total_combinations = 1
    for key, values in param_grid.items():
        if key != 'initial_capital':
            total_combinations *= len(values)
    
    print(f"\\nTotal combinations to test: {total_combinations}")
    print("\\nStarting optimization...")
    
    # Storage for results
    results = []
    
    # Calculate Buy & Hold benchmark once
    benchmark_metrics, _ = calculate_buy_hold_benchmark(df, param_grid['initial_capital'][0])
    
    # Test all combinations
    combination_num = 0
    for params in product(*[param_grid[key] for key in sorted(param_grid.keys())]):
        combination_num += 1
        
        # Map parameters
        param_dict = dict(zip(sorted(param_grid.keys()), params))
        
        print(f"\\rTesting combination {combination_num}/{total_combinations}: "
              f"conf={param_dict['confidence_threshold']:.3f}, "
              f"fee={param_dict['fee']:.3f}, "
              f"change={param_dict['change_threshold']:.3f}, "
              f"pred={param_dict['prediction_col']}", end="", flush=True)
        
        try:
            # Generate signals with current parameters
            signals = strat_directional_accuracy(
                df, 
                prediction_col=param_dict['prediction_col'],
                confidence_threshold=param_dict['confidence_threshold']
            )
            
            # Run backtest
            metrics, equity, trades = backtest_enhanced(
                df, signals, 
                fee=param_dict['fee'], 
                initial_capital=param_dict['initial_capital']
            )
            
            # Store results
            result = {
                **param_dict,
                **metrics,
                'num_trades': len(trades),
                'combination_num': combination_num
            }
            results.append(result)
            
        except Exception as e:
            print(f"\\nError with combination {combination_num}: {e}")
            continue
    
    print("\\n\\nOptimization complete!")
    
    # Convert to DataFrame and analyze
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("No successful combinations found!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f'deepseek_optimization_{timestamp}.csv')
    results_df.to_csv(results_file, index=False)
    
    print(f"\\nFull results saved to: {results_file}")
    
    # Analysis and ranking
    print("\\n" + "="*80)
    print("OPTIMIZATION RESULTS ANALYSIS")
    print("="*80)
    
    # Top 10 by Total Return
    top_return = results_df.nlargest(10, 'Total Return')
    print("\\nTOP 10 BY TOTAL RETURN:")
    print("-"*60)
    for _, row in top_return.iterrows():
        print(f"Return: {row['Total Return']:.2%} | "
              f"Sharpe: {row['Sharpe Ratio']:.3f} | "
              f"Drawdown: {row['Max Drawdown']:.2%} | "
              f"Trades: {row['num_trades']} | "
              f"Conf: {row['confidence_threshold']:.3f} | "
              f"Fee: {row['fee']:.3f} | "
              f"Change: {row['change_threshold']:.3f} | "
              f"Pred: {row['prediction_col']}")
    
    # Top 10 by Sharpe Ratio
    top_sharpe = results_df.nlargest(10, 'Sharpe Ratio')
    print("\\nTOP 10 BY SHARPE RATIO:")
    print("-"*60)
    for _, row in top_sharpe.iterrows():
        print(f"Sharpe: {row['Sharpe Ratio']:.3f} | "
              f"Return: {row['Total Return']:.2%} | "
              f"Drawdown: {row['Max Drawdown']:.2%} | "
              f"Trades: {row['num_trades']} | "
              f"Conf: {row['confidence_threshold']:.3f} | "
              f"Fee: {row['fee']:.3f} | "
              f"Change: {row['change_threshold']:.3f} | "
              f"Pred: {row['prediction_col']}")
    
    # Best overall (balanced metric: Return / abs(Drawdown) * Sharpe)
    results_df['Score'] = (results_df['Total Return'] / abs(results_df['Max Drawdown'])) * results_df['Sharpe Ratio']
    best_overall = results_df.nlargest(10, 'Score')
    
    print("\\nTOP 10 BY BALANCED SCORE (Return/Drawdown*Sharpe):")
    print("-"*60)
    for _, row in best_overall.iterrows():
        print(f"Score: {row['Score']:.3f} | "
              f"Return: {row['Total Return']:.2%} | "
              f"Sharpe: {row['Sharpe Ratio']:.3f} | "
              f"Drawdown: {row['Max Drawdown']:.2%} | "
              f"Conf: {row['confidence_threshold']:.3f} | "
              f"Fee: {row['fee']:.3f} | "
              f"Change: {row['change_threshold']:.3f} | "
              f"Pred: {row['prediction_col']}")
    
    # Parameter sensitivity analysis
    print("\\n" + "="*80)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Average performance by parameter
    for param in ['confidence_threshold', 'fee', 'change_threshold', 'prediction_col']:
        avg_by_param = results_df.groupby(param).agg({
            'Total Return': ['mean', 'std', 'max'],
            'Sharpe Ratio': ['mean', 'std', 'max']
        }).round(4)
        
        print(f"\\nAverage performance by {param.upper()}:")
        print(avg_by_param)
    
    # Comparison to baseline
    baseline_return = 7.7096  # Current 770.96% as decimal
    baseline_sharpe = 1.041
    
    improvements = results_df[
        (results_df['Total Return'] > baseline_return) & 
        (results_df['Sharpe Ratio'] > baseline_sharpe)
    ]
    
    print(f"\\n" + "="*80)
    print(f"IMPROVEMENTS OVER BASELINE (Return > {baseline_return:.1%}, Sharpe > {baseline_sharpe:.3f})")
    print("="*80)
    
    if len(improvements) > 0:
        print(f"Found {len(improvements)} combinations that beat baseline on both metrics!")
        best_improvement = improvements.nlargest(1, 'Total Return').iloc[0]
        print(f"\\nBest improvement:")
        print(f"  Return: {best_improvement['Total Return']:.2%} (vs {baseline_return:.1%})")
        print(f"  Sharpe: {best_improvement['Sharpe Ratio']:.3f} (vs {baseline_sharpe:.3f})")
        print(f"  Parameters: conf={best_improvement['confidence_threshold']:.3f}, "
              f"fee={best_improvement['fee']:.3f}, "
              f"change={best_improvement['change_threshold']:.3f}, "
              f"pred={best_improvement['prediction_col']}")
    else:
        print("No combinations beat the baseline on both metrics.")
        print("Best single-metric improvements:")
        best_return_only = results_df.nlargest(1, 'Total Return').iloc[0]
        best_sharpe_only = results_df.nlargest(1, 'Sharpe Ratio').iloc[0]
        print(f"  Best Return: {best_return_only['Total Return']:.2%}")
        print(f"  Best Sharpe: {best_sharpe_only['Sharpe Ratio']:.3f}")
    
    # Save summary
    summary = {
        'optimization_timestamp': timestamp,
        'total_combinations_tested': len(results_df),
        'baseline_return': baseline_return,
        'baseline_sharpe': baseline_sharpe,
        'best_return': results_df['Total Return'].max(),
        'best_sharpe': results_df['Sharpe Ratio'].max(),
        'best_overall_params': best_overall.iloc[0].to_dict()
    }
    
    summary_file = os.path.join(output_dir, f'optimization_summary_{timestamp}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\\nSummary saved to: {summary_file}")
    
    return results_df, summary

def main():
    parser = argparse.ArgumentParser(description='Optimize DEEPSEEK Directional Strategy')
    parser.add_argument('csv_file', help='Path to DEEPSEEK prediction CSV file')
    parser.add_argument('--output_dir', default='./optimization_results', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run optimization
    results_df, summary = optimize_deepseek_directional(args.csv_file, args.output_dir)
    
    print(f"\\nOptimization complete! Check {args.output_dir} for detailed results.")

if __name__ == '__main__':
    main()