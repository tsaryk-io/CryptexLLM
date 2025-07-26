#!/usr/bin/env python3

import pandas as pd
import glob
import os
import sys

def analyze_backtest_results(results_dir='./backtest_results'):
    """Analyze and compare all backtesting results"""
    
    # Find all CSV result files
    pattern = os.path.join(results_dir, 'backtest_results_*.csv')
    files = glob.glob(pattern)
    
    if not files:
        print(f"No backtest result files found in {results_dir}")
        sys.exit(1)
    
    print(f"Found {len(files)} result files:")
    for f in files:
        print(f"  {os.path.basename(f)}")
    print()
    
    all_results = []
    
    for file in files:
        try:
            df = pd.read_csv(file, index_col=0)  # Strategy is index
            
            # Extract model name from filename
            basename = os.path.basename(file)
            # Remove 'backtest_results_enhanced_' and '.csv.csv'
            model_part = basename.replace('backtest_results_enhanced_', '').replace('.csv.csv', '')
            
            # Extract just the model name (before _L6)
            if '_L6' in model_part:
                model_name = model_part.split('_L6')[0]
            else:
                model_name = model_part
            
            # Add model column
            df_reset = df.reset_index()
            df_reset['Model'] = model_name
            all_results.append(df_reset)
            
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not all_results:
        print("No valid results found")
        sys.exit(1)
    
    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)
    
    # Create pivot tables for easy comparison
    metrics = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Volatility']
    
    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    
    for metric in metrics:
        if metric in combined.columns:
            print(f"\n{metric.upper()}:")
            print("-" * 60)
            pivot = combined.pivot_table(index='Model', columns='Strategy', values=metric, aggfunc='first')
            if metric == 'Total Return':
                print(pivot.to_string(float_format='{:.2%}'.format))
            elif metric in ['Sharpe Ratio']:
                print(pivot.to_string(float_format='{:.3f}'.format))
            elif metric in ['Max Drawdown']:
                print(pivot.to_string(float_format='{:.2%}'.format))
            elif metric in ['Win Rate']:
                print(pivot.to_string(float_format='{:.1%}'.format))
            else:
                print(pivot.to_string(float_format='{:.4f}'.format))
    
    # Find best performers
    print("\n" + "="*80)
    print("TOP PERFORMERS BY STRATEGY")
    print("="*80)
    
    strategies = combined['Strategy'].unique()
    strategies = [s for s in strategies if s != 'Buy & Hold']  # Exclude benchmark
    
    for strategy in strategies:
        strategy_data = combined[combined['Strategy'] == strategy]
        if len(strategy_data) > 0:
            best_return = strategy_data.loc[strategy_data['Total Return'].idxmax()]
            best_sharpe = strategy_data.loc[strategy_data['Sharpe Ratio'].idxmax()]
            
            print(f"\n{strategy.upper()} Strategy:")
            print(f"  Best Return: {best_return['Model']} ({best_return['Total Return']:.2%})")
            print(f"  Best Sharpe: {best_sharpe['Model']} ({best_sharpe['Sharpe Ratio']:.3f})")
    
    # Overall best model
    print("\n" + "="*80)
    print("OVERALL RANKINGS")
    print("="*80)
    
    # Exclude Buy & Hold for rankings
    model_data = combined[combined['Strategy'] != 'Buy & Hold']
    
    # Average performance by model
    model_avg = model_data.groupby('Model').agg({
        'Total Return': 'mean',
        'Sharpe Ratio': 'mean',
        'Max Drawdown': 'mean',
        'Win Rate': 'mean'
    }).round(4)
    
    print("\nAverage Performance Across All Strategies:")
    print(model_avg.sort_values('Total Return', ascending=False))
    
    # Save combined results
    output_file = os.path.join(results_dir, 'combined_analysis.csv')
    combined.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze backtesting results')
    parser.add_argument('--results_dir', default='./backtest_results', 
                       help='Directory containing backtest result CSV files')
    
    args = parser.parse_args()
    analyze_backtest_results(args.results_dir)