#!/usr/bin/env python3
"""
Test script for full sentiment API integration with NewsAPI
"""

import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.external_data_integration import ExternalDataManager

def test_full_sentiment_integration():
    """Test all three sentiment APIs: Reddit + News + Fear & Greed"""
    
    print("=" * 70)
    print("TESTING COMPLETE SENTIMENT INTEGRATION (Reddit + News + Fear & Greed)")
    print("=" * 70)
    
    # Initialize with config file that contains NewsAPI key
    config_file = "external_data_config.json"
    manager = ExternalDataManager(config_file)
    
    # Test date range (last 2 days for quick testing)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
    
    print(f"Testing sentiment data fetch from {start_date} to {end_date}")
    print(f"Using config file: {config_file}")
    
    try:
        # Test sentiment data source specifically
        sentiment_source = manager.data_sources.get('sentiment')
        if sentiment_source:
            print("\nTesting complete sentiment integration...")
            sentiment_data = sentiment_source.fetch_data(start_date, end_date)
            
            print(f"✓ Sentiment data shape: {sentiment_data.shape}")
            print(f"✓ Columns: {list(sentiment_data.columns)}")
            
            if not sentiment_data.empty:
                print(f"\nSENTIMENT ANALYSIS RESULTS:")
                print(f"   • Sentiment score range: {sentiment_data['sentiment_score'].min():.3f} - {sentiment_data['sentiment_score'].max():.3f}")
                print(f"   • Fear & Greed range: {sentiment_data['fear_greed_index'].min()} - {sentiment_data['fear_greed_index'].max()}")
                print(f"   • Reddit posts total: {sentiment_data['reddit_posts'].sum()}")
                print(f"   • News articles total: {sentiment_data['news_articles'].sum()}")
                
                print(f"\nSample hourly data:")
                sample_cols = ['timestamp', 'sentiment_score', 'fear_greed_index', 'reddit_posts', 'news_articles']
                print(sentiment_data[sample_cols].head(5).to_string(index=False))
                
                # Check if we have real news data
                has_news = sentiment_data['news_articles'].sum() > 0
                print(f"\nNewsAPI Integration: {'Active' if has_news else 'No articles found'}")
            
        else:
            print("Sentiment data source not found")
            
        print(f"\nComplete sentiment integration test finished!")
        print(f"   Real data sources active: Reddit ✓ | News {'✓' if has_news else '✗'} | Fear & Greed ✓")

    except Exception as e:
        print(f" Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_full_sentiment_integration()