#!/usr/bin/env python3

import os
import sys
import time
import pandas as pd
import numpy as np
# import torch  # Skip torch for testing core feature selection
from utils.feature_selection import CorrelationBasedFeatureSelector, quick_feature_selection

def test_feature_selection():
    """
    Test the correlation-based feature selection on Time-LLM-Cryptex data
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
        print("Available data files:")
        for file in os.listdir("./dataset/"):
            if file.endswith('.csv'):
                print(f"  - {file}")
        return
    
    print(f"Loading data from: {data_path}")
    
    # Load and analyze data
    start_time = time.time()
    data = pd.read_csv(data_path)
    load_time = time.time() - start_time
    
    print(f"Data loaded in {load_time:.2f} seconds")
    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Test enhanced data loading with feature engineering
    print("\n" + "-" * 60)
    print("TESTING ENHANCED DATA LOADING")
    print("-" * 60)
    
    try:
        from data_provider.enhanced_data_loader import Dataset_CRYPTEX_Enhanced
        from utils.feature_engineering import apply_all_technical_indicators
        
        # Apply feature engineering to see full feature set
        print("Applying technical indicators...")
        enhanced_data = apply_all_technical_indicators(data)
        
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
                                    if col not in ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']])
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
        
        # Quick training test simulation
        print("\n" + "-" * 60)
        print("TRAINING PERFORMANCE SIMULATION")
        print("-" * 60)
        
        # Simulate training data loading times
        batch_size = 32
        seq_len = 21
        
        def simulate_batch_processing(data, batch_size, seq_len):
            """Simulate time to process a batch of sequences"""
            n_samples = len(data) - seq_len
            n_batches = n_samples // batch_size
            
            start_time = time.time()
            
            for i in range(min(10, n_batches)):  # Test first 10 batches
                # Simulate feature extraction
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                batch_data = []
                for j in range(start_idx, min(end_idx, n_samples)):
                    sequence = data.iloc[j:j+seq_len].select_dtypes(include=[np.number]).values
                    batch_data.append(sequence)
                
                # Simulate tensor operations
                if batch_data:
                    # batch_tensor = torch.tensor(np.array(batch_data), dtype=torch.float32)
                    batch_array = np.array(batch_data)  # Use numpy instead of torch
            
            return time.time() - start_time
        
        original_time = simulate_batch_processing(enhanced_data, batch_size, seq_len)
        optimized_time = simulate_batch_processing(optimized_data, batch_size, seq_len)
        
        actual_speedup = original_time / optimized_time if optimized_time > 0 else 0
        
        print(f"Batch processing simulation:")
        print(f"  Original: {original_time:.3f} seconds")
        print(f"  Optimized: {optimized_time:.3f} seconds")
        print(f"  Actual speedup: {actual_speedup:.1f}x")
        
        print("\n" + "=" * 80)
        print("FEATURE SELECTION TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Reduced features from {original_feature_count} to {selected_feature_count}")
        print(f"Expected training speedup: {speedup_estimate:.1f}x")
        print(f"Memory reduction: {memory_reduction:.1f}%")
        print(f"Selection completed in {selection_time:.2f} seconds")
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure enhanced_data_loader.py and feature_engineering.py are available")
        return False
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quick_selection():
    """
    Test the quick feature selection utility
    """
    print("\n" + "=" * 80)
    print("TESTING QUICK FEATURE SELECTION UTILITY")
    print("=" * 80)
    
    data_path = "./dataset/cryptex/candlesticks-D.csv"
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return False
    
    try:
        selected_features = quick_feature_selection(data_path, target_features=15)
        
        print(f"\nQuick selection completed!")
        print(f"Selected {len(selected_features)} features:")
        for i, feature in enumerate(selected_features[:10], 1):
            print(f"  {i:2d}. {feature}")
        
        if len(selected_features) > 10:
            print(f"  ... and {len(selected_features) - 10} more")
        
        return True
        
    except Exception as e:
        print(f"Quick selection failed: {e}")
        return False


if __name__ == "__main__":
    print("Time-LLM-Cryptex Feature Selection Test")
    print("Author: Claude (Anthropic)")
    print("Purpose: Optimize training by reducing feature dimensionality\n")
    
    # Run comprehensive test
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
        print("Please check the error messages above and ensure all dependencies are available")