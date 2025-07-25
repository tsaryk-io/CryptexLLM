#!/usr/bin/env python3

import argparse
import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------------
# Enhanced Backtester for 26-feature enhanced dataset models
# ------------------------------------------------------------------

def backtest_enhanced(df, signals, fee=0.0, initial_capital=1000.0):
    """
    Enhanced backtesting with detailed trade tracking and metrics
    """
    cash = initial_capital
    pos = 0.0
    equity = []
    trades = []
    returns = []
    
    for i, (date, signal) in enumerate(signals.items()):
        price = df.at[date, 'close']
        
        if signal == 1 and cash > 0.0:
            # BUY: invest cash after fee
            invest_amt = cash * (1.0 - fee)
            pos = invest_amt / price
            cash = 0.0
            trades.append((date, price, 'buy'))
            
        elif signal == -1 and pos > 0.0:
            # SELL: liquidate position minus fee
            proceeds = pos * price
            cash = proceeds * (1.0 - fee)
            pos = 0.0
            trades.append((date, price, 'sell'))
        
        current_equity = cash + pos * price
        equity.append(current_equity)
        
        # Calculate daily returns
        if i > 0:
            daily_return = (current_equity - equity[i-1]) / equity[i-1]
            returns.append(daily_return)
        else:
            returns.append(0.0)
    
    # Convert to pandas Series
    equity_series = pd.Series(equity, index=signals.index)
    returns_series = pd.Series(returns, index=signals.index)
    
    # Calculate comprehensive metrics
    total_return = (equity_series.iloc[-1] / initial_capital) - 1
    
    # Annualized return (assuming daily data)
    trading_days = len(equity_series)
    if trading_days > 0:
        ann_return = (1 + total_return) ** (252 / trading_days) - 1
    else:
        ann_return = 0.0
    
    # Sharpe ratio
    if returns_series.std() > 0:
        sharpe = (returns_series.mean() / returns_series.std()) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Maximum drawdown
    rolling_max = equity_series.expanding().max()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Win rate
    if len(trades) >= 2:
        profitable_trades = 0
        total_trades = 0
        for i in range(1, len(trades)):
            if trades[i-1][2] == 'buy' and trades[i][2] == 'sell':
                if trades[i][1] > trades[i-1][1]:  # Sell price > Buy price
                    profitable_trades += 1
                total_trades += 1
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0.0
    else:
        win_rate = 0.0
    
    # Volatility
    volatility = returns_series.std() * np.sqrt(252)
    
    return {
        'Total Return': total_return,
        'Ann. Return': ann_return,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown,
        'Volatility': volatility,
        'Win Rate': win_rate,
        'Num Trades': len(trades),
        'Final Equity': equity_series.iloc[-1]
    }, equity_series, trades

def calculate_buy_hold_benchmark(df, initial_capital=1000.0):
    """Calculate buy-and-hold benchmark performance"""
    initial_price = df['close'].iloc[0]
    final_price = df['close'].iloc[-1]
    
    shares = initial_capital / initial_price
    final_value = shares * final_price
    
    total_return = (final_value / initial_capital) - 1
    
    # Calculate buy-hold equity curve
    equity_curve = (df['close'] / initial_price) * initial_capital
    returns = equity_curve.pct_change().fillna(0)
    
    # Annualized return
    trading_days = len(df)
    ann_return = (1 + total_return) ** (252 / trading_days) - 1
    
    # Sharpe ratio
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Maximum drawdown
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Volatility
    volatility = returns.std() * np.sqrt(252)
    
    return {
        'Total Return': total_return,
        'Ann. Return': ann_return,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown,
        'Volatility': volatility,
        'Win Rate': 1.0 if total_return > 0 else 0.0,
        'Num Trades': 1,
        'Final Equity': final_value
    }, equity_curve

# ------------------------------------------------------------------
# Enhanced Strategy Definitions
# ------------------------------------------------------------------

def strat_predicted_enhanced(df, horizon=0, change=0.0, prediction_col='close_predicted_1'):
    """
    Enhanced prediction strategy with multiple signal generation methods
    """
    if prediction_col not in df.columns:
        raise KeyError(f'DataFrame must contain {prediction_col} column')
    
    preds = df[prediction_col].fillna(method='ffill')  # Forward fill NaN values
    sig = pd.Series(0, index=df.index)
    
    if horizon < 1:
        # Method 1: Compare prediction vs current price
        pct_diff = (preds - df['close']) / df['close']
        sig[pct_diff > change] = 1   # Buy if prediction > current price + threshold
        sig[pct_diff < -change] = -1  # Sell if prediction < current price - threshold
    else:
        # Method 2: Compare prediction trend over horizon
        for i in range(horizon, len(preds) - 1):
            if i - horizon + 1 >= 0:
                prev_pred = preds.iloc[i - horizon + 1]
                curr_pred = preds.iloc[i + 1]
                if prev_pred != 0:
                    rel_change = (curr_pred - prev_pred) / prev_pred
                    if rel_change > change:
                        sig.iloc[i] = 1
                    elif rel_change < -change:
                        sig.iloc[i] = -1
    
    return sig

def strat_directional_accuracy(df, prediction_col='close_predicted_1', confidence_threshold=0.01):
    """
    Strategy based on directional accuracy with confidence threshold
    """
    if prediction_col not in df.columns:
        raise KeyError(f'DataFrame must contain {prediction_col} column')
    
    sig = pd.Series(0, index=df.index)
    preds = df[prediction_col].fillna(method='ffill')
    
    # Calculate predicted vs actual direction
    actual_direction = np.sign(df['close'].diff())
    predicted_direction = np.sign(preds.diff())
    
    # Calculate prediction confidence (magnitude of predicted change)
    pred_change_magnitude = np.abs(preds.pct_change())
    
    # Generate signals when:
    # 1. Prediction confidence is high (change > threshold)
    # 2. Direction is clear (not flat)
    for i in range(1, len(df)):
        if pred_change_magnitude.iloc[i] > confidence_threshold:
            if predicted_direction.iloc[i] > 0:  # Predicted upward movement
                sig.iloc[i] = 1
            elif predicted_direction.iloc[i] < 0:  # Predicted downward movement
                sig.iloc[i] = -1
    
    return sig

def strat_hybrid_enhanced(df, prediction_col='close_predicted_1', short_ma=5, long_ma=20, pred_weight=0.7):
    """
    Hybrid strategy combining predictions with technical indicators
    """
    if prediction_col not in df.columns:
        raise KeyError(f'DataFrame must contain {prediction_col} column')
    
    sig = pd.Series(0, index=df.index)
    preds = df[prediction_col].fillna(method='ffill')
    
    # Calculate moving averages
    short_sma = df['close'].rolling(window=short_ma).mean()
    long_sma = df['close'].rolling(window=long_ma).mean()
    
    # Technical signal: 1 if short_sma > long_sma, -1 otherwise
    tech_signal = np.where(short_sma > long_sma, 1, -1)
    
    # Prediction signal: based on predicted vs current price
    pred_signal = np.where(preds > df['close'], 1, -1)
    
    # Combine signals with weighting
    combined_signal = pred_weight * pred_signal + (1 - pred_weight) * tech_signal
    
    # Convert to discrete signals
    sig[combined_signal > 0.3] = 1   # Strong buy
    sig[combined_signal < -0.3] = -1  # Strong sell
    
    return sig

# ------------------------------------------------------------------
# Enhanced Visualization
# ------------------------------------------------------------------

def create_enhanced_plot(df, equity_curves, trades_map, benchmark_equity, strategies):
    """
    Create comprehensive visualization with multiple subplots
    """
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Price & Predictions', 'Portfolio Performance', 'Drawdown Analysis'),
        vertical_spacing=0.1,
        row_heights=[0.4, 0.4, 0.2]
    )
    
    # Subplot 1: Price and predictions
    fig.add_trace(
        go.Scatter(x=df.index, y=df['close'], name='Actual Close', 
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # Add prediction if available
    pred_cols = [col for col in df.columns if 'predicted' in col]
    if pred_cols:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[pred_cols[0]], name='Predicted Close',
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
    
    # Add buy/sell signals for best strategy
    best_strategy = max(strategies.keys(), key=lambda k: strategies[k]['Total Return'])
    trades = trades_map.get(best_strategy, [])
    
    buys = [(d, p) for d, p, s in trades if s == 'buy']
    sells = [(d, p) for d, p, s in trades if s == 'sell']
    
    if buys:
        fig.add_trace(
            go.Scatter(x=[d for d, _ in buys], y=[p for _, p in buys],
                      mode='markers', name='Buy Signals',
                      marker=dict(symbol='triangle-up', size=12, color='green')),
            row=1, col=1
        )
    
    if sells:
        fig.add_trace(
            go.Scatter(x=[d for d, _ in sells], y=[p for _, p in sells],
                      mode='markers', name='Sell Signals',
                      marker=dict(symbol='triangle-down', size=12, color='red')),
            row=1, col=1
        )
    
    # Subplot 2: Portfolio performance
    fig.add_trace(
        go.Scatter(x=benchmark_equity.index, y=benchmark_equity.values,
                  name='Buy & Hold', line=dict(color='gray', dash='dot')),
        row=2, col=1
    )
    
    colors = ['green', 'purple', 'orange', 'brown']
    for i, (strategy_name, equity) in enumerate(equity_curves.items()):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(x=equity.index, y=equity.values, name=f'{strategy_name} Portfolio',
                      line=dict(color=color, width=2)),
            row=2, col=1
        )
    
    # Subplot 3: Drawdown analysis
    for i, (strategy_name, equity) in enumerate(equity_curves.items()):
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max * 100
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values, 
                      name=f'{strategy_name} Drawdown',
                      line=dict(color=color), fill='tonexty' if i == 0 else None),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        title='Enhanced Trading Strategy Analysis',
        height=900,
        showlegend=True,
        legend=dict(x=0.01, y=0.99)
    )
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
    
    return fig

# ------------------------------------------------------------------
# Main Program
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced backtesting for 26-feature TimeLLM models'
    )
    parser.add_argument('filename', help='Enhanced CSV file with predictions')
    parser.add_argument('--horizon', '-n', type=int, default=0,
                       help='Prediction horizon (default: 0)')
    parser.add_argument('--fee', '-f', type=float, default=0.001,
                       help='Per-trade fee as fraction (default: 0.1%)')
    parser.add_argument('--change', '-c', type=float, default=0.02,
                       help='Signal threshold (default: 2%)')
    parser.add_argument('--initial_capital', type=float, default=1000.0,
                       help='Initial capital (default: $1000)')
    parser.add_argument('--prediction_col', type=str, default='close_predicted_1',
                       help='Prediction column name')
    parser.add_argument('--output_dir', type=str, default='./backtest_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load enhanced dataset
    try:
        df = pd.read_csv(args.filename, header=0)
        print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f'❌ Error reading {args.filename}: {e}', file=sys.stderr)
        sys.exit(1)
    
    # Clean column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Handle timestamp/date column
    date_col = next((c for c in df.columns if 'date' in c or 'time' in c), None)
    if not date_col:
        print('❌ No date/time column found.', file=sys.stderr)
        sys.exit(1)
    
    # Parse dates
    try:
        col = df[date_col]
        if pd.api.types.is_numeric_dtype(col):
            df['dt'] = pd.to_datetime(col, unit='s', errors='raise')
        else:
            df['dt'] = pd.to_datetime(col, errors='raise')
    except Exception as e:
        print(f'❌ Could not parse {date_col}: {e}', file=sys.stderr)
        sys.exit(1)
    
    df.set_index('dt', inplace=True)
    
    # Verify required columns
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            print(f'❌ Missing required column: {col}', file=sys.stderr)
            sys.exit(1)
    
    # Check for prediction column
    pred_col = args.prediction_col.lower()
    if pred_col not in df.columns:
        print(f'❌ Prediction column {pred_col} not found.', file=sys.stderr)
        print(f'Available columns: {list(df.columns)}')
        sys.exit(1)
    
    print(f"✅ Using prediction column: {pred_col}")
    
    # Fill NaN values in prediction column
    df[pred_col] = df[pred_col].fillna(method='ffill').fillna(method='bfill')
    
    # Remove rows where close price is NaN
    initial_rows = len(df)
    df = df.dropna(subset=['close'])
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} rows with missing close prices")
    
    # Define strategies
    strategies_funcs = {
        'Predicted_Basic': lambda df: strat_predicted_enhanced(df, args.horizon, args.change, pred_col),
        'Directional': lambda df: strat_directional_accuracy(df, pred_col, confidence_threshold=0.01),
        'Hybrid': lambda df: strat_hybrid_enhanced(df, pred_col),
    }
    
    print(f"\n🚀 Running backtest on {len(df)} data points...")
    print(f"Initial capital: ${args.initial_capital}")
    print(f"Trading fee: {args.fee*100:.2f}%")
    print(f"Signal threshold: {args.change*100:.2f}%")
    
    # Calculate buy-and-hold benchmark
    benchmark_metrics, benchmark_equity = calculate_buy_hold_benchmark(df, args.initial_capital)
    
    # Run strategies
    results = []
    equity_curves = {}
    trades_map = {}
    
    for name, func in strategies_funcs.items():
        try:
            sig = func(df)
            metrics, equity, trades = backtest_enhanced(df, sig, args.fee, args.initial_capital)
            metrics['Strategy'] = name
            results.append(metrics)
            equity_curves[name] = equity
            trades_map[name] = trades
            print(f"✅ {name}: {len(trades)} trades, {metrics['Total Return']:.2%} return")
        except Exception as e:
            print(f"❌ Error running {name}: {e}")
    
    # Add benchmark to results
    benchmark_metrics['Strategy'] = 'Buy & Hold'
    results.append(benchmark_metrics)
    
    # Create results summary
    summary = pd.DataFrame(results).set_index('Strategy')
    
    print('\n📊 Strategy Performance Summary:')
    print('=' * 80)
    print(summary.to_string(float_format='{:.4f}'.format))
    
    # Find best strategy
    strategy_names = [r['Strategy'] for r in results[:-1]]  # Exclude benchmark
    best_strategy = summary.loc[strategy_names, 'Total Return'].idxmax()
    
    print(f'\n🏆 Best Strategy: {best_strategy}')
    print(f'   Total Return: {summary.loc[best_strategy, "Total Return"]:.2%}')
    print(f'   Sharpe Ratio: {summary.loc[best_strategy, "Sharpe Ratio"]:.3f}')
    print(f'   Max Drawdown: {summary.loc[best_strategy, "Max Drawdown"]:.2%}')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save detailed results
    output_file = os.path.join(args.output_dir, f'backtest_results_{os.path.basename(args.filename)}.csv')
    summary.to_csv(output_file)
    print(f'\n💾 Results saved to: {output_file}')
    
    # Create enhanced visualization
    try:
        fig = create_enhanced_plot(df, equity_curves, trades_map, benchmark_equity, 
                                 {name: summary.loc[name].to_dict() for name in strategy_names})
        
        plot_file = os.path.join(args.output_dir, f'backtest_plot_{os.path.basename(args.filename)}.html')
        fig.write_html(plot_file)
        print(f'📈 Interactive plot saved to: {plot_file}')
        
        # Show plot if running interactively
        fig.show()
        
    except Exception as e:
        print(f"⚠️  Could not create plot: {e}")

if __name__ == '__main__':
    main()