#!/usr/bin/env python3
"""
Create enhanced dataset with all implemented features:
- Real sentiment data (Reddit, News, Fear & Greed)
- Macro economic indicators
- On-chain metrics
- Feature selection (top 20 indicators)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.external_data_integration import create_external_data_manager
from utils.feature_selection import CorrelationBasedFeatureSelector

def create_complete_enhanced_dataset():
    """Create the complete enhanced dataset with all features"""
    
    print("=" * 80)
    print("CREATING COMPLETE ENHANCED DATASET")
    print("=" * 80)
    
    # 1. Load basic price data
    print("1. Loading basic price data...")
    price_data = pd.read_csv('dataset/cryptex/candlesticks-D.csv')
    print(f"   Basic price data: {price_data.shape}")
    
    # Convert timestamp for external data fetching
    price_data['datetime'] = pd.to_datetime(price_data['timestamp'], unit='s')
    start_date = price_data['datetime'].min().strftime('%Y-%m-%d')
    end_date = price_data['datetime'].max().strftime('%Y-%m-%d')
    
    print(f"   Date range: {start_date} to {end_date}")
    
    # 2. Add external data integration
    print("\n2. Fetching external data (sentiment + macro + on-chain)...")
    
    config_file = 'external_data_config.json'
    if os.path.exists(config_file):
        manager = create_external_data_manager(config_file)
        print("   Using external data config with NewsAPI key")
    else:
        manager = create_external_data_manager()
        print("   Using default config (Reddit + Fear & Greed only)")
    
    # Fetch all external data
    external_data = manager.fetch_all_data(start_date, end_date)
    
    # Show what we got
    for source, data in external_data.items():
        if not data.empty:
            print(f"   {source}: {data.shape[0]} records, {data.shape[1]} features")
        else:
            print(f"   {source}: No data")
    
    # 3. Merge all data sources
    print("\n3. Merging all data sources...")
    enhanced_data = manager.align_and_merge_data(price_data, external_data)
    print(f"   Enhanced dataset: {enhanced_data.shape}")
    
    # 4. Apply feature selection (top 20 features)
    print("\n4. Applying correlation-based feature selection...")
    selector = CorrelationBasedFeatureSelector(target_features=20, correlation_threshold=0.95)
    
    # Select features
    selected_features = selector.fit_select(enhanced_data, target_col='close')
    
    # Transform dataset to use only selected features
    final_dataset = selector.transform(enhanced_data)
    print(f"   Final dataset: {final_dataset.shape}")
    
    # 5. Save enhanced dataset
    print("\n5. Saving enhanced dataset...")
    output_path = 'dataset/cryptex/candlesticks-D-enhanced.csv'
    final_dataset.to_csv(output_path, index=False)
    print(f"   Saved to: {output_path}")
    
    # 6. Save feature selection results
    print("\n6. Saving feature selection results...")
    selector.save_selection('dataset/cryptex/selected_features.json')
    
    # 7. Create summary report
    print("\n7. Dataset Enhancement Summary:")
    print(f"   Original features: 6 (OHLCV + timestamp)")
    print(f"   With external data: {enhanced_data.shape[1]} features")
    print(f"   After feature selection: {final_dataset.shape[1]} features")
    print(f"   Selected features: {selected_features}")
    
    return final_dataset, selected_features

if __name__ == "__main__":
    enhanced_data, features = create_complete_enhanced_dataset()
    
    print("\n" + "=" * 80)
    print("ENHANCED DATASET READY FOR EXPERIMENTS")
    print("=" * 80)
    print("Now you can run experiments with:")
    print("- Real sentiment data integration")
    print("- Macro economic indicators") 
    print("- On-chain blockchain metrics")
    print("- Top 20 most predictive features")
    print("- MLFlow experiment tracking")
    print("- Optuna hyperparameter optimization")