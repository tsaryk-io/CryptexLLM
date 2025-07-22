#!/usr/bin/env python3

import os
import sys
import time
import pandas as pd
import numpy as np
from utils.feature_selection import CorrelationBasedFeatureSelector

def create_sample_enhanced_data(data):
    """
    Create sample enhanced data with technical indicators
    (simplified version for testing without full dependencies)
    """
    print("Creating sample enhanced data with technical indicators...")
    
    enhanced_data = data.copy()
    
    # Simple moving averages
    for window in [5, 10, 20, 50]:
        enhanced_data[f'sma_{window}'] = enhanced_data['close'].rolling(window=window).mean()
        enhanced_data[f'volume_sma_{window}'] = enhanced_data['volume'].rolling(window=window).mean()
    
    # Price-based features
    enhanced_data['high_low_ratio'] = enhanced_data['high'] / enhanced_data['low']
    enhanced_data['open_close_ratio'] = enhanced_data['open'] / enhanced_data['close']
    enhanced_data['price_range'] = enhanced_data['high'] - enhanced_data['low']
    enhanced_data['body_size'] = abs(enhanced_data['close'] - enhanced_data['open'])
    enhanced_data['upper_shadow'] = enhanced_data['high'] - enhanced_data[['open', 'close']].max(axis=1)
    enhanced_data['lower_shadow'] = enhanced_data[['open', 'close']].min(axis=1) - enhanced_data['low']
    
    # Volatility indicators
    enhanced_data['price_volatility'] = enhanced_data['close'].rolling(window=10).std()
    enhanced_data['volume_volatility'] = enhanced_data['volume'].rolling(window=10).std()
    
    # Momentum indicators (simplified)
    for window in [5, 10, 14]:
        enhanced_data[f'momentum_{window}'] = enhanced_data['close'] / enhanced_data['close'].shift(window)
        enhanced_data[f'roc_{window}'] = (enhanced_data['close'] - enhanced_data['close'].shift(window)) / enhanced_data['close'].shift(window) * 100
    
    # Simple RSI approximation
    delta = enhanced_data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    enhanced_data['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands approximation
    enhanced_data['bb_middle'] = enhanced_data['close'].rolling(window=20).mean()
    bb_std = enhanced_data['close'].rolling(window=20).std()
    enhanced_data['bb_upper'] = enhanced_data['bb_middle'] + (bb_std * 2)
    enhanced_data['bb_lower'] = enhanced_data['bb_middle'] - (bb_std * 2)
    enhanced_data['bb_width'] = enhanced_data['bb_upper'] - enhanced_data['bb_lower']
    enhanced_data['bb_position'] = (enhanced_data['close'] - enhanced_data['bb_lower']) / enhanced_data['bb_width']
    
    # MACD approximation
    ema_12 = enhanced_data['close'].ewm(span=12).mean()
    ema_26 = enhanced_data['close'].ewm(span=26).mean()
    enhanced_data['macd'] = ema_12 - ema_26
    enhanced_data['macd_signal'] = enhanced_data['macd'].ewm(span=9).mean()
    enhanced_data['macd_histogram'] = enhanced_data['macd'] - enhanced_data['macd_signal']
    
    # Volume indicators
    enhanced_data['volume_price_trend'] = enhanced_data['volume'] * ((enhanced_data['close'] - enhanced_data['close'].shift(1)) / enhanced_data['close'].shift(1))
    enhanced_data['on_balance_volume'] = (enhanced_data['volume'] * np.where(enhanced_data['close'] > enhanced_data['close'].shift(1), 1, 
                                                                          np.where(enhanced_data['close'] < enhanced_data['close'].shift(1), -1, 0))).cumsum()
    
    # Price action patterns
    enhanced_data['doji'] = (abs(enhanced_data['close'] - enhanced_data['open']) <= (enhanced_data['high'] - enhanced_data['low']) * 0.1).astype(int)
    enhanced_data['hammer'] = ((enhanced_data['lower_shadow'] > 2 * enhanced_data['body_size']) & 
                              (enhanced_data['upper_shadow'] < enhanced_data['body_size'])).astype(int)
    
    # Trend indicators
    enhanced_data['price_above_sma_20'] = (enhanced_data['close'] > enhanced_data['sma_20']).astype(int)
    enhanced_data['price_above_sma_50'] = (enhanced_data['close'] > enhanced_data['sma_50']).astype(int)
    enhanced_data['sma_20_above_50'] = (enhanced_data['sma_20'] > enhanced_data['sma_50']).astype(int)
    
    # Additional features to reach 68+ total
    for i in range(1, 11):
        enhanced_data[f'lag_close_{i}'] = enhanced_data['close'].shift(i)
        enhanced_data[f'lag_volume_{i}'] = enhanced_data['volume'].shift(i)
    
    # Ratio features
    enhanced_data['volume_to_sma_volume'] = enhanced_data['volume'] / enhanced_data['volume_sma_20']
    enhanced_data['close_to_sma_ratio'] = enhanced_data['close'] / enhanced_data['sma_20']
    
    print(f"Enhanced data created with {enhanced_data.shape[1]} total features")
    
    # Remove rows with NaN values (due to rolling calculations)
    enhanced_data = enhanced_data.dropna()
    print(f"After removing NaN: {enhanced_data.shape}")
    
    return enhanced_data


def test_feature_selection():
    """
    Test the correlation-based feature selection
    """
    print("=" * 80)
    print("TESTING CORRELATION-BASED FEATURE SELECTION")
    print("=" * 80)
    
    # Configuration
    data_path = "./dataset/cryptex/candlesticks-D.csv"
    target_features = 20  # Reduce from 68+ to 20 features (~70% reduction)
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return False
    
    print(f"Loading data from: {data_path}")
    
    # Load and analyze data
    start_time = time.time()
    data = pd.read_csv(data_path)
    load_time = time.time() - start_time
    
    print(f"Data loaded in {load_time:.2f} seconds")
    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Create enhanced data with technical indicators
    print("\n" + "-" * 60)
    print("CREATING ENHANCED DATA")
    print("-" * 60)
    
    enhanced_data = create_sample_enhanced_data(data)
    
    print(f"Enhanced data shape: {enhanced_data.shape}")
    print(f"Features added: {enhanced_data.shape[1] - data.shape[1]}")
    
    # Show sample of enhanced features
    technical_features = [col for col in enhanced_data.columns 
                        if col not in data.columns]
    print(f"Technical features ({len(technical_features)}): {technical_features[:10]}...")
    
    # Test correlation-based feature selection
    print("\n" + "-" * 60)
    print("RUNNING FEATURE SELECTION")
    print("-" * 60)
    
    selector = CorrelationBasedFeatureSelector(
        target_features=target_features,
        correlation_threshold=0.9
    )
    
    selection_start = time.time()
    selected_features = selector.fit_select(enhanced_data, target_col='close')
    selection_time = time.time() - selection_start
    
    print(f"\nFeature selection completed in {selection_time:.2f} seconds")
    
    # Show results
    print("\n" + "-" * 60)
    print("SELECTION RESULTS")
    print("-" * 60)
    
    original_feature_count = len([col for col in enhanced_data.columns 
                                if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])
    selected_feature_count = len(selected_features)
    reduction_pct = (1 - selected_feature_count / original_feature_count) * 100
    speedup_estimate = original_feature_count / selected_feature_count
    
    print(f"Original features: {original_feature_count}")
    print(f"Selected features: {selected_feature_count}")
    print(f"Reduction: {reduction_pct:.1f}%")
    print(f"Expected speedup: {speedup_estimate:.1f}x")
    
    print(f"\nSelected features:")
    for i, feature in enumerate(selected_features, 1):
        importance = selector.feature_importance_scores.get(feature, 0)
        print(f"  {i:2d}. {feature:<30} | Importance: {importance:.4f}")
    
    # Test data transformation
    print("\n" + "-" * 60)
    print("TESTING DATA TRANSFORMATION")
    print("-" * 60)
    
    transform_start = time.time()
    optimized_data = selector.transform(enhanced_data)
    transform_time = time.time() - transform_start
    
    print(f"Data transformation completed in {transform_time:.3f} seconds")
    print(f"Optimized data shape: {optimized_data.shape}")
    
    # Memory usage comparison
    original_memory = enhanced_data.memory_usage(deep=True).sum() / 1024**2
    optimized_memory = optimized_data.memory_usage(deep=True).sum() / 1024**2
    memory_reduction = (1 - optimized_memory / original_memory) * 100
    
    print(f"\nMemory usage:")
    print(f"  Original: {original_memory:.1f} MB")
    print(f"  Optimized: {optimized_memory:.1f} MB")
    print(f"  Reduction: {memory_reduction:.1f}%")
    
    # Save results
    output_dir = "./feature_selection_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save selection configuration
    selector.save_selection(f"{output_dir}/feature_selection_config.json")
    
    # Save optimized dataset sample
    sample_size = min(1000, len(optimized_data))
    optimized_data.head(sample_size).to_csv(f"{output_dir}/optimized_data_sample.csv", index=False)
    
    # Create performance summary
    summary = {
        'original_features': original_feature_count,
        'selected_features': selected_feature_count,
        'reduction_percentage': reduction_pct,
        'expected_speedup': speedup_estimate,
        'selection_time_seconds': selection_time,
        'memory_reduction_percentage': memory_reduction,
        'selected_feature_list': selected_features
    }
    
    import json
    with open(f"{output_dir}/performance_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}/")
    
    print("\n" + "=" * 80)
    print("FEATURE SELECTION TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Reduced features from {original_feature_count} to {selected_feature_count}")
    print(f"Expected training speedup: {speedup_estimate:.1f}x")
    print(f"Memory reduction: {memory_reduction:.1f}%")
    print(f"Selection completed in {selection_time:.2f} seconds")
    
    return True


if __name__ == "__main__":
    print("Time-LLM-Cryptex Feature Selection Test (Simplified)")
    print("Author: Claude (Anthropic)")
    print("Purpose: Optimize training by reducing feature dimensionality\n")
    
    # Run test
    success = test_feature_selection()
    
    if success:
        print("\nReady to use optimized features for faster training!")
        print("\nNext steps:")
        print("1. Use the selected features in your training configuration")
        print("2. Update enhanced_data_loader.py to use selected features")
        print("3. Run training experiments with reduced computational cost")
        print("4. Compare training time and performance metrics")
    else:
        print("\nFeature selection test failed")
        print("Please check the error messages above")