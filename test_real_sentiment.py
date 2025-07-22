#!/usr/bin/env python3
"""
Test script for real sentiment API integration
"""

import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.external_data_integration import ExternalDataManager

def test_real_sentiment_integration():
    """Test the real sentiment API integration"""
    
    print("=" * 60)
    print("TESTING REAL SENTIMENT API INTEGRATION")
    print("=" * 60)
    
    # Initialize the external data manager (no config file, will use defaults)
    manager = ExternalDataManager()
    
    # Test date range (last 3 days for quick testing)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    
    print(f"Testing sentiment data fetch from {start_date} to {end_date}")
    
    try:
        # Test fetching sentiment data specifically
        sentiment_source = manager.data_sources.get('sentiment')
        if sentiment_source:
            print("\n1. Testing sentiment data source...")
            sentiment_data = sentiment_source.fetch_data(start_date, end_date)
            
            print(f"✓ Sentiment data shape: {sentiment_data.shape}")
            print(f"✓ Columns: {list(sentiment_data.columns)}")
            
            if not sentiment_data.empty:
                print(f"✓ Sample sentiment score range: {sentiment_data['sentiment_score'].min():.3f} - {sentiment_data['sentiment_score'].max():.3f}")
                print(f"✓ Sample data:")
                print(sentiment_data.head(3)[['timestamp', 'sentiment_score', 'fear_greed_index', 'reddit_posts', 'news_articles']])
            
        else:
            print("Sentiment data source not found in manager")
            
        print("\n2. Testing full data integration...")
        
        # Test full data fetch (this will include sentiment + macro + onchain)
        all_data = manager.fetch_all_data(start_date, end_date)
        
        for source_name, data in all_data.items():
            print(f"{source_name}: {data.shape if not data.empty else 'No data'}")
            
        print(f"\nReal sentiment API integration test completed!")
        print(f"   - Sentiment API integration: {'✓ Working' if not sentiment_data.empty else 'Using fallback'}")
        print(f"   - Total data sources: {len([d for d in all_data.values() if not d.empty])}/3")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_sentiment_integration()