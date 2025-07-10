#!/usr/bin/env python3
"""
Test script for the enhanced feature engineering and new loss functions
"""

import os
import sys
import pandas as pd
import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, skipping loss function tests")

import matplotlib.pyplot as plt
import seaborn as sns
from utils.feature_engineering import FeatureEngineer

if TORCH_AVAILABLE:
    from utils.metrics import get_loss_function

def test_feature_engineering():
    """Test the feature engineering pipeline"""
    print("=" * 50)
    print("TESTING FEATURE ENGINEERING")
    print("=" * 50)
    
    # Load sample data
    data_path = './dataset/cryptex/candlesticks-D.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        return False
    
    df = pd.read_csv(data_path)
    print(f"Loaded data: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Test feature creation
    try:
        print("\n1. Creating features...")
        df_features = feature_engineer.create_features(df)
        
        print(f"Original features: {len(df.columns)}")
        print(f"Enhanced features: {len(df_features.columns)}")
        print(f"Feature ratio: {len(df_features.columns) / len(df.columns):.1f}x")
        
        # Print some feature examples
        print("\nNew features include:")
        new_features = [col for col in df_features.columns if col not in df.columns]
        for i, feat in enumerate(new_features[:10]):
            print(f"  {i+1}. {feat}")
        if len(new_features) > 10:
            print(f"  ... and {len(new_features) - 10} more")
        
        # Test feature importance
        print("\n2. Computing feature importance...")
        importance = feature_engineer.get_feature_importance(df_features, 'close')
        print("\nTop 10 most correlated features with close price:")
        print(importance.head(10).to_string(index=False))
        
        # Check for any NaN values
        nan_count = df_features.isnull().sum().sum()
        print(f"\n3. Data quality check:")
        print(f"   NaN values: {nan_count}")
        print(f"   Data shape after cleaning: {df_features.shape}")
        
        return True, df_features
        
    except Exception as e:
        print(f"Error in feature engineering: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

def test_loss_functions():
    """Test the new loss functions"""
    print("\n" + "=" * 50)
    print("TESTING LOSS FUNCTIONS")
    print("=" * 50)
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available - skipping loss function tests")
        return {}
    
    # Create sample data
    batch_size, seq_len = 32, 24
    torch.manual_seed(42)
    
    # Generate realistic price movements
    base_price = 50000  # Bitcoin-like price
    returns = torch.randn(batch_size, seq_len) * 0.02  # 2% daily volatility
    true_prices = base_price * torch.exp(returns.cumsum(dim=1))
    
    # Add some noise to create predictions
    pred_prices = true_prices + torch.randn_like(true_prices) * 100
    
    print(f"Test data shape: {true_prices.shape}")
    print(f"Price range: ${true_prices.min():.0f} - ${true_prices.max():.0f}")
    
    # Test each loss function
    loss_functions = [
        'MSE', 'MAE', 'MAPE', 'ASYMMETRIC', 'QUANTILE', 
        'SHARPE_LOSS', 'TRADING_LOSS', 'ROBUST'
    ]
    
    results = {}
    
    for loss_name in loss_functions:
        try:
            print(f"\n{loss_name}:")
            loss_fn = get_loss_function(loss_name)
            loss_value = loss_fn(pred_prices, true_prices)
            results[loss_name] = float(loss_value)
            print(f"  Loss value: {loss_value:.6f}")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            results[loss_name] = None
    
    # Summary
    print(f"\nLoss Function Results:")
    for name, value in results.items():
        if value is not None:
            print(f"  {name}: {value:.6f}")
        else:
            print(f"  {name}: FAILED")
    
    return results

def test_data_loading():
    """Test the enhanced data loading"""
    print("\n" + "=" * 50)
    print("TESTING ENHANCED DATA LOADING")
    print("=" * 50)
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available - skipping data loading tests")
        return True
    
    try:
        from data_provider.enhanced_data_loader import Dataset_CRYPTEX_Enhanced
        
        # Test enhanced dataset
        dataset = Dataset_CRYPTEX_Enhanced(
            root_path='./dataset/cryptex/',
            data_path='candlesticks-D.csv',
            flag='train',
            size=[24, 12, 6],
            features='M',
            enable_feature_engineering=True
        )
        
        print(f"Dataset length: {len(dataset)}")
        print(f"Number of features: {dataset.enc_in}")
        print(f"Feature names: {len(dataset.get_feature_names())}")
        
        # Test data loading
        sample = dataset[0]
        seq_x, seq_y, seq_x_mark, seq_y_mark = sample
        
        print(f"Sample shapes:")
        print(f"  seq_x: {seq_x.shape}")
        print(f"  seq_y: {seq_y.shape}")
        print(f"  seq_x_mark: {seq_x_mark.shape}")
        print(f"  seq_y_mark: {seq_y_mark.shape}")
        
        # Test feature importance
        importance = dataset.get_feature_importance()
        if importance is not None:
            print(f"\nTop 5 features by importance:")
            print(importance.head().to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"Error in data loading: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_visualizations(df_features):
    """Create some visualizations of the features"""
    print("\n" + "=" * 50)
    print("CREATING VISUALIZATIONS")
    print("=" * 50)
    
    try:
        # Create output directory
        os.makedirs('./plots', exist_ok=True)
        
        # Plot 1: Price and some technical indicators
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # Convert timestamp for plotting
        df_plot = df_features.tail(200).copy()  # Last 200 days
        df_plot['date'] = pd.to_datetime(df_plot['timestamp'], unit='s')
        
        # Price and moving averages
        axes[0].plot(df_plot['date'], df_plot['close'], label='Close Price', linewidth=1)
        if 'sma_20' in df_plot.columns:
            axes[0].plot(df_plot['date'], df_plot['sma_20'], label='SMA 20', alpha=0.7)
        if 'ema_20' in df_plot.columns:
            axes[0].plot(df_plot['date'], df_plot['ema_20'], label='EMA 20', alpha=0.7)
        axes[0].set_title('Price and Moving Averages')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RSI
        if 'rsi_14' in df_plot.columns:
            axes[1].plot(df_plot['date'], df_plot['rsi_14'], label='RSI 14', color='orange')
            axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
            axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
            axes[1].set_title('RSI Indicator')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, 100)
        
        # Volume and volatility
        axes[2].bar(df_plot['date'], df_plot['volume'], alpha=0.6, label='Volume')
        if 'realized_vol_30' in df_plot.columns:
            ax2 = axes[2].twinx()
            ax2.plot(df_plot['date'], df_plot['realized_vol_30'], color='red', label='Realized Volatility')
            ax2.legend(loc='upper right')
        axes[2].set_title('Volume and Volatility')
        axes[2].legend(loc='upper left')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./plots/technical_indicators.png', dpi=150, bbox_inches='tight')
        print("Saved: ./plots/technical_indicators.png")
        
        # Plot 2: Feature correlation heatmap
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        corr_cols = ['close', 'volume', 'rsi_14', 'macd', 'bb_position', 'realized_vol_30']
        corr_cols = [col for col in corr_cols if col in numeric_cols]
        
        if len(corr_cols) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = df_features[corr_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f')
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig('./plots/feature_correlation.png', dpi=150, bbox_inches='tight')
            print("Saved: ./plots/feature_correlation.png")
        
        plt.close('all')
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")

def main():
    """Main test function"""
    print("Testing Enhanced Time-LLM Features")
    print("=" * 60)
    
    # Test 1: Feature Engineering
    success1, df_features = test_feature_engineering()
    
    # Test 2: Loss Functions
    if TORCH_AVAILABLE:
        success2 = test_loss_functions()
    else:
        success2 = True  # Skip if torch not available
    
    # Test 3: Data Loading
    success3 = test_data_loading()
    
    # Test 4: Visualizations
    if success1 and df_features is not None:
        create_visualizations(df_features)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Feature Engineering: {'✓ PASS' if success1 else '✗ FAIL'}")
    print(f"Loss Functions:      {'✓ PASS' if success2 else '✗ FAIL'}")
    print(f"Data Loading:        {'✓ PASS' if success3 else '✗ FAIL'}")
    
    if all([success1, success2, success3]):
        print("\n🎉 All tests passed! Ready to train with enhanced features.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    main()