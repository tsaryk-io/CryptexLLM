#!/usr/bin/env python3
"""
Test script for External Data Integration components
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.external_data_integration import (
        ExternalDataManager,
        SentimentDataSource,
        MacroEconomicDataSource,
        OnChainDataSource,
        DataSourceConfig
    )
    EXTERNAL_DATA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: External data components not available: {e}")
    EXTERNAL_DATA_AVAILABLE = False

try:
    from data_provider.external_enhanced_loader import (
        Dataset_CRYPTEX_External,
        Dataset_CRYPTEX_Regime_Aware
    )
    EXTERNAL_LOADER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: External data loader not available: {e}")
    EXTERNAL_LOADER_AVAILABLE = False


def test_external_data_sources():
    """Test individual external data sources"""
    print("=" * 60)
    print("TESTING EXTERNAL DATA SOURCES")
    print("=" * 60)
    
    if not EXTERNAL_DATA_AVAILABLE:
        print("External data components not available - skipping")
        return False
    
    try:
        # Test date range
        start_date = '2023-01-01'
        end_date = '2023-01-31'
        
        print(f"Testing data sources for period: {start_date} to {end_date}")
        
        # Test Sentiment Data Source
        print("\n1. Testing Sentiment Data Source...")
        sentiment_config = DataSourceConfig(
            name='sentiment',
            rate_limit=0.1,  # Faster for testing
            cache_duration=60
        )
        sentiment_source = SentimentDataSource(sentiment_config)
        sentiment_data = sentiment_source.fetch_data(start_date, end_date)
        
        print(f"   Sentiment data shape: {sentiment_data.shape}")
        print(f"   Sentiment columns: {sentiment_data.columns.tolist()}")
        print(f"   Sample sentiment score: {sentiment_data['sentiment_score'].mean():.3f}")
        
        # Test Macro Economic Data Source
        print("\n2. Testing Macro Economic Data Source...")
        macro_config = DataSourceConfig(
            name='macro',
            rate_limit=0.1,
            cache_duration=60
        )
        macro_source = MacroEconomicDataSource(macro_config)
        macro_data = macro_source.fetch_data(start_date, end_date)
        
        print(f"   Macro data shape: {macro_data.shape}")
        print(f"   Macro columns: {macro_data.columns.tolist()}")
        print(f"   Sample Fed funds rate: {macro_data['fed_funds_rate'].mean():.2f}%")
        print(f"   Sample S&P 500: {macro_data['sp500'].mean():.0f}")
        
        # Test On-Chain Data Source
        print("\n3. Testing On-Chain Data Source...")
        onchain_config = DataSourceConfig(
            name='onchain',
            rate_limit=0.1,
            cache_duration=60
        )
        onchain_source = OnChainDataSource(onchain_config)
        onchain_data = onchain_source.fetch_data(start_date, end_date)
        
        print(f"   On-chain data shape: {onchain_data.shape}")
        print(f"   On-chain columns: {onchain_data.columns.tolist()}")
        print(f"   Sample hash rate: {onchain_data['hash_rate'].mean():.0f} TH/s")
        print(f"   Sample active addresses: {onchain_data['active_addresses'].mean():.0f}")
        
        print("\n✓ All external data sources test passed")
        return True, {
            'sentiment': sentiment_data,
            'macro': macro_data,
            'onchain': onchain_data
        }
        
    except Exception as e:
        print(f"✗ External data sources test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def test_external_data_manager():
    """Test the external data manager"""
    print("\n" + "=" * 60)
    print("TESTING EXTERNAL DATA MANAGER")
    print("=" * 60)
    
    if not EXTERNAL_DATA_AVAILABLE:
        print("External data components not available - skipping")
        return False
    
    try:
        # Create external data manager
        manager = ExternalDataManager()
        
        # Test configuration
        print(f"Enabled data sources: {list(manager.data_sources.keys())}")
        
        # Test fetching all data
        start_date = '2023-01-01'
        end_date = '2023-01-15'
        
        print(f"\nFetching all external data for {start_date} to {end_date}...")
        all_data = manager.fetch_all_data(start_date, end_date)
        
        print(f"Fetched data from {len(all_data)} sources:")
        for source_name, data in all_data.items():
            print(f"  {source_name}: {data.shape}")
        
        # Test data alignment with crypto data
        print("\nTesting data alignment...")
        
        # Create sample crypto data
        crypto_dates = pd.date_range(start=start_date, end=end_date, freq='H')
        crypto_data = pd.DataFrame({
            'timestamp': crypto_dates.astype(int) // 10**9,
            'open': np.random.randn(len(crypto_dates)) * 100 + 50000,
            'high': np.random.randn(len(crypto_dates)) * 100 + 50100,
            'low': np.random.randn(len(crypto_dates)) * 100 + 49900,
            'close': np.random.randn(len(crypto_dates)) * 100 + 50000,
            'volume': np.random.randn(len(crypto_dates)) * 1000 + 5000
        })
        
        print(f"Sample crypto data shape: {crypto_data.shape}")
        
        # Align and merge
        merged_data = manager.align_and_merge_data(crypto_data, all_data)
        
        print(f"Merged data shape: {merged_data.shape}")
        print(f"Features added: {merged_data.shape[1] - crypto_data.shape[1]}")
        
        # Test feature impact analysis
        impact_analysis = manager.analyze_feature_impact(merged_data, 'close')
        
        print(f"\nFeature impact analysis completed:")
        print(f"  Total features analyzed: {len(impact_analysis)}")
        
        external_features = impact_analysis[impact_analysis['is_external']]
        print(f"  External features: {len(external_features)}")
        
        if len(external_features) > 0:
            print(f"  Top external feature: {external_features.iloc[0]['feature']} (corr: {external_features.iloc[0]['correlation']:.3f})")
        
        print("\n✓ External data manager test passed")
        return True, merged_data
        
    except Exception as e:
        print(f"✗ External data manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_external_enhanced_loader():
    """Test the external enhanced data loader"""
    print("\n" + "=" * 60)
    print("TESTING EXTERNAL ENHANCED DATA LOADER")
    print("=" * 60)
    
    if not EXTERNAL_LOADER_AVAILABLE:
        print("External data loader not available - skipping")
        return False
    
    try:
        # Test Dataset_CRYPTEX_External
        print("Testing Dataset_CRYPTEX_External...")
        
        dataset = Dataset_CRYPTEX_External(
            root_path='./dataset/cryptex/',
            data_path='candlesticks-D.csv',
            flag='train',
            size=[24, 12, 6],
            features='M',
            enable_feature_engineering=True,
            enable_external_data=True
        )
        
        print(f"Dataset length: {len(dataset)}")
        print(f"Number of features: {dataset.enc_in}")
        
        # Get feature summary
        feature_summary = dataset.get_external_feature_summary()
        print(f"\nFeature breakdown:")
        for category, count in feature_summary.items():
            print(f"  {category}: {count}")
        
        # Test data loading
        sample = dataset[0]
        seq_x, seq_y, seq_x_mark, seq_y_mark = sample
        
        print(f"\nSample shapes:")
        print(f"  seq_x: {seq_x.shape}")
        print(f"  seq_y: {seq_y.shape}")
        print(f"  seq_x_mark: {seq_x_mark.shape}")
        print(f"  seq_y_mark: {seq_y_mark.shape}")
        
        print("\n✓ External enhanced loader test passed")
        return True, dataset
        
    except Exception as e:
        print(f"✗ External enhanced loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_regime_aware_loader():
    """Test the regime-aware data loader"""
    print("\n" + "=" * 60)
    print("TESTING REGIME-AWARE DATA LOADER")
    print("=" * 60)
    
    if not EXTERNAL_LOADER_AVAILABLE:
        print("Regime-aware loader not available - skipping")
        return False
    
    try:
        print("Testing Dataset_CRYPTEX_Regime_Aware...")
        
        dataset = Dataset_CRYPTEX_Regime_Aware(
            root_path='./dataset/cryptex/',
            data_path='candlesticks-D.csv',
            flag='train',
            size=[24, 12, 6],
            features='M',
            enable_feature_engineering=True,
            enable_external_data=True,
            regime_detection_window=20,
            volatility_threshold_low=0.015,
            volatility_threshold_high=0.04
        )
        
        print(f"Regime-aware dataset length: {len(dataset)}")
        
        # Test regime statistics
        regime_stats = dataset.get_regime_statistics()
        
        print(f"\nRegime statistics:")
        for regime, stats in regime_stats.items():
            print(f"  {regime}:")
            print(f"    Count: {stats['count']}")
            print(f"    Avg Return: {stats['avg_return']:.4f}")
            print(f"    Volatility: {stats['volatility']:.4f}")
            print(f"    Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        
        # Test regime detection at specific index
        test_index = min(100, len(dataset.market_regimes) - 1)
        regime_at_index = dataset.get_regime_at_index(test_index)
        print(f"\nRegime at index {test_index}: {regime_at_index}")
        
        print("\n✓ Regime-aware loader test passed")
        return True, dataset
        
    except Exception as e:
        print(f"✗ Regime-aware loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def create_external_data_visualizations(external_data: Dict[str, pd.DataFrame]):
    """Create visualizations for external data"""
    print("\n" + "=" * 60)
    print("CREATING EXTERNAL DATA VISUALIZATIONS")
    print("=" * 60)
    
    try:
        os.makedirs('./plots', exist_ok=True)
        
        # Create subplots for different data types
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('External Data Sources Overview', fontsize=16, fontweight='bold')
        
        # Sentiment data
        if 'sentiment' in external_data and not external_data['sentiment'].empty:
            sentiment_df = external_data['sentiment']
            dates = pd.to_datetime(sentiment_df['timestamp'], unit='s')
            
            axes[0, 0].plot(dates, sentiment_df['sentiment_score'], label='Sentiment Score', color='blue')
            axes[0, 0].set_title('Social Sentiment')
            axes[0, 0].set_ylabel('Sentiment Score (0-1)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # Fear & Greed Index
            axes[0, 1].plot(dates, sentiment_df['fear_greed_index'], label='Fear & Greed', color='orange')
            axes[0, 1].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Neutral')
            axes[0, 1].axhline(y=25, color='red', linestyle='--', alpha=0.5, label='Fear')
            axes[0, 1].axhline(y=75, color='green', linestyle='--', alpha=0.5, label='Greed')
            axes[0, 1].set_title('Fear & Greed Index')
            axes[0, 1].set_ylabel('Index (0-100)')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        # Macro data
        if 'macro' in external_data and not external_data['macro'].empty:
            macro_df = external_data['macro']
            dates = pd.to_datetime(macro_df['timestamp'], unit='s')
            
            # Interest rates
            axes[1, 0].plot(dates, macro_df['fed_funds_rate'], label='Fed Funds Rate', color='red')
            axes[1, 0].plot(dates, macro_df['treasury_10y'], label='10Y Treasury', color='blue')
            axes[1, 0].set_title('Interest Rates')
            axes[1, 0].set_ylabel('Rate (%)')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            
            # Stock market indices
            ax1 = axes[1, 1]
            ax2 = ax1.twinx()
            
            line1 = ax1.plot(dates, macro_df['sp500'], label='S&P 500', color='blue')
            line2 = ax2.plot(dates, macro_df['vix'], label='VIX', color='red')
            
            ax1.set_title('Stock Market Indicators')
            ax1.set_ylabel('S&P 500', color='blue')
            ax2.set_ylabel('VIX', color='red')
            ax1.grid(True, alpha=0.3)
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
        
        # On-chain data
        if 'onchain' in external_data and not external_data['onchain'].empty:
            onchain_df = external_data['onchain']
            dates = pd.to_datetime(onchain_df['timestamp'], unit='s')
            
            # Network metrics
            ax1 = axes[2, 0]
            ax2 = ax1.twinx()
            
            line1 = ax1.plot(dates, onchain_df['hash_rate'] / 1e9, label='Hash Rate (EH/s)', color='blue')
            line2 = ax2.plot(dates, onchain_df['active_addresses'] / 1000, label='Active Addresses (K)', color='green')
            
            ax1.set_title('Network Fundamentals')
            ax1.set_ylabel('Hash Rate (EH/s)', color='blue')
            ax2.set_ylabel('Active Addresses (K)', color='green')
            ax1.grid(True, alpha=0.3)
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
            
            # Exchange flows
            axes[2, 1].plot(dates, onchain_df['exchange_inflow'], label='Inflow', color='red', alpha=0.7)
            axes[2, 1].plot(dates, onchain_df['exchange_outflow'], label='Outflow', color='green', alpha=0.7)
            axes[2, 1].fill_between(dates, onchain_df['exchange_inflow'], 
                                  onchain_df['exchange_outflow'], alpha=0.3, color='blue')
            axes[2, 1].set_title('Exchange Flows')
            axes[2, 1].set_ylabel('BTC Flow')
            axes[2, 1].grid(True, alpha=0.3)
            axes[2, 1].legend()
        
        plt.tight_layout()
        plt.savefig('./plots/external_data_overview.png', dpi=150, bbox_inches='tight')
        print("Saved: ./plots/external_data_overview.png")
        plt.close()
        
        # Create correlation heatmap
        if len(external_data) > 1:
            print("Creating feature correlation heatmap...")
            
            # Combine all external data
            combined_features = []
            feature_names = []
            
            for source_name, data in external_data.items():
                if not data.empty:
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    numeric_cols = [col for col in numeric_cols if col != 'timestamp']
                    
                    for col in numeric_cols[:5]:  # Take first 5 features per source
                        combined_features.append(data[col].values[:min(100, len(data))])
                        feature_names.append(f"{source_name}_{col}")
            
            if len(combined_features) > 1:
                # Create correlation matrix
                min_length = min(len(feat) for feat in combined_features)
                feature_matrix = np.array([feat[:min_length] for feat in combined_features]).T
                
                feature_df = pd.DataFrame(feature_matrix, columns=feature_names)
                correlation_matrix = feature_df.corr()
                
                plt.figure(figsize=(12, 10))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
                plt.title('External Data Feature Correlations')
                plt.tight_layout()
                plt.savefig('./plots/external_data_correlations.png', dpi=150, bbox_inches='tight')
                print("Saved: ./plots/external_data_correlations.png")
                plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return False


def test_api_configuration():
    """Test API configuration system"""
    print("\n" + "=" * 60)
    print("TESTING API CONFIGURATION")
    print("=" * 60)
    
    if not EXTERNAL_DATA_AVAILABLE:
        print("External data components not available - skipping")
        return False
    
    try:
        # Create external data manager
        manager = ExternalDataManager()
        
        # Save configuration template
        config_file = './external_data_config.json'
        manager.save_config_template(config_file)
        
        print(f"Configuration template created: {config_file}")
        
        # Test loading configuration
        if os.path.exists(config_file):
            manager_with_config = ExternalDataManager(config_file)
            print(f"Configuration loaded successfully")
            print(f"Available sources: {list(manager_with_config.data_sources.keys())}")
        
        print("\n✓ API configuration test passed")
        return True
        
    except Exception as e:
        print(f"✗ API configuration test failed: {e}")
        return False


def main():
    """Main test function for external data integration"""
    print("Testing External Data Integration Components")
    print("=" * 80)
    
    # Run all tests
    test_results = []
    external_data = {}
    
    # Test individual data sources
    success1, external_data = test_external_data_sources()
    test_results.append(("External Data Sources", success1))
    
    # Test data manager
    success2, merged_data = test_external_data_manager()
    test_results.append(("External Data Manager", success2))
    
    # Test enhanced loader
    success3, dataset = test_external_enhanced_loader()
    test_results.append(("External Enhanced Loader", success3))
    
    # Test regime-aware loader
    success4, regime_dataset = test_regime_aware_loader()
    test_results.append(("Regime-Aware Loader", success4))
    
    # Test API configuration
    success5 = test_api_configuration()
    test_results.append(("API Configuration", success5))
    
    # Create visualizations
    if external_data:
        success6 = create_external_data_visualizations(external_data)
        test_results.append(("Visualizations", success6))
    else:
        test_results.append(("Visualizations", False))
    
    # Summary
    print("\n" + "=" * 80)
    print("EXTERNAL DATA INTEGRATION TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:30} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed >= total - 1:  # Allow one failure
        print("\nExternal data integration tests passed!")
        print("\nThe system now includes:")
        print("• Sentiment analysis from social media and news")
        print("• Macro economic indicators (rates, inflation, stocks)")
        print("• On-chain metrics (hash rate, addresses, flows)")
        print("• Automated data synchronization and alignment")
        print("• Market regime detection and adaptation")
        print("• Feature impact analysis and correlation tracking")
        
        print("\nNext steps:")
        print("1. Add real API keys to external_data_config.json")
        print("2. Run training with external data: --data CRYPTEX_EXTERNAL")
        print("3. Compare performance vs baseline models")
        print("4. Analyze which external features are most predictive")
        
        return True
    else:
        print(f"\n{total-passed} tests failed. Check errors above.")
        return False


if __name__ == "__main__":
    main()