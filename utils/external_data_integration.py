import os
import time
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore')


@dataclass
class DataSourceConfig:
    """Configuration for external data sources"""
    name: str
    api_key: Optional[str] = None
    base_url: str = ""
    rate_limit: float = 1.0  # Seconds between requests
    cache_duration: int = 3600  # Cache duration in seconds
    enabled: bool = True


class ExternalDataSource(ABC):
    """Abstract base class for external data sources"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.last_request_time = 0
        self.cache = {}
        
    def _respect_rate_limit(self):
        """Ensure rate limiting between API calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.config.rate_limit:
            time.sleep(self.config.rate_limit - time_since_last)
        self.last_request_time = time.time()
    
    def _get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get cached data if still valid"""
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.config.cache_duration:
                return data
        return None
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame):
        """Cache data with timestamp"""
        self.cache[cache_key] = (data, time.time())
    
    @abstractmethod
    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from the external source"""
        pass


class SentimentDataSource(ExternalDataSource):
    """
    Sentiment analysis data from social media and news
    """
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.sentiment_keywords = [
            'bitcoin', 'BTC', 'cryptocurrency', 'crypto', 'blockchain',
            'hodl', 'moon', 'diamond hands', 'paper hands', 'bull market',
            'bear market', 'altcoin', 'DeFi', 'NFT'
        ]
    
    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch sentiment data (simulated for now - replace with real APIs)
        
        Real implementations could use:
        - Twitter API v2 for tweets
        - Reddit API for r/Bitcoin, r/cryptocurrency posts
        - News APIs like NewsAPI, Alpha Vantage News
        - Crypto-specific sentiment providers like Santiment, LunarCrush
        """
        cache_key = f"sentiment_{start_date}_{end_date}"
        cached = self._get_cached_data(cache_key)
        if cached is not None:
            return cached
        
        print(f"Fetching sentiment data from {start_date} to {end_date}")
        
        # Simulate sentiment data (replace with real API calls)
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Simulate realistic sentiment patterns
        np.random.seed(42)
        n_points = len(date_range)
        
        # Base sentiment with trending patterns
        trend = np.sin(np.linspace(0, 4*np.pi, n_points)) * 0.3
        noise = np.random.normal(0, 0.2, n_points)
        sentiment_score = 0.5 + trend + noise  # Range roughly 0-1
        sentiment_score = np.clip(sentiment_score, 0, 1)
        
        # Volume metrics
        base_volume = 1000
        volume_trend = np.exp(np.linspace(0, 1, n_points)) * base_volume
        volume_noise = np.random.lognormal(0, 0.5, n_points)
        tweet_volume = volume_trend * volume_noise
        
        # Additional sentiment metrics
        fear_greed_index = (sentiment_score * 100).astype(int)  # 0-100 scale
        bullish_ratio = sentiment_score * 0.8 + np.random.normal(0, 0.1, n_points)
        bullish_ratio = np.clip(bullish_ratio, 0, 1)
        
        sentiment_df = pd.DataFrame({
            'timestamp': date_range,
            'sentiment_score': sentiment_score,  # 0-1 scale
            'tweet_volume': tweet_volume.astype(int),
            'reddit_posts': (tweet_volume * 0.1).astype(int),
            'news_articles': (tweet_volume * 0.05).astype(int),
            'fear_greed_index': fear_greed_index,
            'bullish_ratio': bullish_ratio,
            'bearish_ratio': 1 - bullish_ratio,
            'social_dominance': np.random.beta(2, 5, n_points),  # Crypto dominance in social
            'engagement_rate': np.random.beta(3, 7, n_points)  # Social engagement quality
        })
        
        # Convert timestamp to unix for consistency
        sentiment_df['timestamp'] = sentiment_df['timestamp'].astype(int) // 10**9
        
        self._cache_data(cache_key, sentiment_df)
        return sentiment_df
    
    def fetch_real_twitter_sentiment(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Template for real Twitter API integration
        """
        # Real implementation would use Twitter API v2
        headers = {
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Example Twitter API v2 query
        query_params = {
            'query': 'bitcoin OR BTC OR cryptocurrency',
            'tweet.fields': 'created_at,public_metrics,context_annotations',
            'start_time': start_date,
            'end_time': end_date,
            'max_results': 100
        }
        
        # This would be the actual API call
        # response = requests.get(f"{self.config.base_url}/tweets/search/recent", 
        #                        headers=headers, params=query_params)
        
        # For now, return simulated data
        return self.fetch_data(start_date, end_date)


class MacroEconomicDataSource(ExternalDataSource):
    """
    Macro economic indicators that influence crypto markets
    """
    
    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch macro economic data
        
        Real implementations could use:
        - FRED API for US economic data
        - Alpha Vantage for financial indicators
        - Yahoo Finance for stock indices
        - Trading Economics API
        """
        cache_key = f"macro_{start_date}_{end_date}"
        cached = self._get_cached_data(cache_key)
        if cached is not None:
            return cached
        
        print(f"Fetching macro economic data from {start_date} to {end_date}")
        
        # Simulate macro economic data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        n_points = len(date_range)
        
        np.random.seed(123)
        
        # Simulate realistic macro indicators
        # Fed funds rate (relatively stable with occasional changes)
        fed_rate_base = 5.25  # Current approximate rate
        fed_rate_changes = np.random.normal(0, 0.01, n_points)
        fed_rate = np.cumsum(fed_rate_changes) + fed_rate_base
        fed_rate = np.clip(fed_rate, 0, 10)
        
        # 10-year Treasury yield
        treasury_10y = fed_rate + np.random.normal(0.5, 0.2, n_points)
        treasury_10y = np.clip(treasury_10y, 0, 8)
        
        # Stock market indices (trending upward with volatility)
        sp500_base = 4500
        sp500_returns = np.random.normal(0.0005, 0.02, n_points)  # ~0.05% daily return, 2% volatility
        sp500 = sp500_base * np.exp(np.cumsum(sp500_returns))
        
        nasdaq_base = 14000
        nasdaq_returns = np.random.normal(0.0007, 0.025, n_points)  # Higher vol than S&P
        nasdaq = nasdaq_base * np.exp(np.cumsum(nasdaq_returns))
        
        # VIX (volatility index - mean reverting)
        vix_mean = 20
        vix = np.random.gamma(2, vix_mean/2, n_points)  # Gamma distribution for VIX-like behavior
        vix = np.clip(vix, 10, 80)
        
        # Dollar index (DXY)
        dxy_base = 103
        dxy_changes = np.random.normal(0, 0.3, n_points)
        dxy = dxy_base + np.cumsum(dxy_changes) * 0.01
        dxy = np.clip(dxy, 90, 120)
        
        # Inflation indicators
        cpi_base = 3.2  # Annual CPI
        cpi_changes = np.random.normal(0, 0.05, n_points)
        cpi_annual = cpi_base + np.cumsum(cpi_changes) * 0.01
        cpi_annual = np.clip(cpi_annual, 0, 10)
        
        # Unemployment rate
        unemployment_base = 3.8
        unemployment_changes = np.random.normal(0, 0.02, n_points)
        unemployment = unemployment_base + np.cumsum(unemployment_changes) * 0.01
        unemployment = np.clip(unemployment, 2, 15)
        
        macro_df = pd.DataFrame({
            'timestamp': date_range,
            'fed_funds_rate': fed_rate,
            'treasury_10y': treasury_10y,
            'sp500': sp500,
            'nasdaq': nasdaq,
            'vix': vix,
            'dxy': dxy,  # Dollar strength
            'cpi_annual': cpi_annual,
            'unemployment_rate': unemployment,
            'yield_curve_spread': treasury_10y - fed_rate,  # 10y-2y spread approximation
            'risk_on_sentiment': (sp500 / sp500_base - 1) * 100,  # Stock performance as risk sentiment
        })
        
        # Convert timestamp to unix
        macro_df['timestamp'] = macro_df['timestamp'].astype(int) // 10**9
        
        self._cache_data(cache_key, macro_df)
        return macro_df
    
    def fetch_real_fred_data(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Template for real FRED API integration
        """
        if not self.config.api_key:
            raise ValueError("FRED API key required")
        
        self._respect_rate_limit()
        
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': series_id,
            'api_key': self.config.api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date
        }
        
        # This would be the actual API call
        # response = requests.get(url, params=params)
        # data = response.json()
        
        # For now, return simulated data
        return self.fetch_data(start_date, end_date)


class OnChainDataSource(ExternalDataSource):
    """
    Bitcoin on-chain metrics from blockchain data
    """
    
    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch on-chain metrics
        
        Real implementations could use:
        - Glassnode API for comprehensive on-chain data
        - CoinMetrics API for institutional-grade metrics
        - Blockchain.info API for basic metrics
        - CryptoQuant for exchange flows
        """
        cache_key = f"onchain_{start_date}_{end_date}"
        cached = self._get_cached_data(cache_key)
        if cached is not None:
            return cached
        
        print(f"Fetching on-chain data from {start_date} to {end_date}")
        
        # Simulate on-chain data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        n_points = len(date_range)
        
        np.random.seed(456)
        
        # Network fundamentals
        # Hash rate (trending upward with noise)
        hash_rate_base = 400_000_000  # TH/s
        hash_rate_trend = np.linspace(1, 1.2, n_points)  # 20% growth over period
        hash_rate_noise = np.random.lognormal(0, 0.1, n_points)
        hash_rate = hash_rate_base * hash_rate_trend * hash_rate_noise
        
        # Mining difficulty (follows hash rate)
        difficulty_base = 60_000_000_000_000
        difficulty = difficulty_base * (hash_rate / hash_rate_base) * np.random.uniform(0.95, 1.05, n_points)
        
        # Active addresses (network usage)
        active_addresses_base = 900_000
        active_addr_trend = np.sin(np.linspace(0, 4*np.pi, n_points)) * 0.2 + 1
        active_addresses = active_addresses_base * active_addr_trend * np.random.lognormal(0, 0.15, n_points)
        
        # Transaction metrics
        tx_count_base = 300_000
        tx_count = tx_count_base * np.random.lognormal(0, 0.2, n_points)
        
        tx_volume_base = 2_000_000  # BTC
        tx_volume = tx_volume_base * np.random.lognormal(0, 0.3, n_points)
        
        # Exchange flows (important for price movements)
        exchange_inflow_base = 15_000  # BTC per day
        exchange_inflow = exchange_inflow_base * np.random.lognormal(0, 0.4, n_points)
        
        exchange_outflow_base = 16_000  # Slightly more outflow (bullish)
        exchange_outflow = exchange_outflow_base * np.random.lognormal(0, 0.4, n_points)
        
        # HODL metrics
        coin_days_destroyed_base = 500_000
        coin_days_destroyed = coin_days_destroyed_base * np.random.lognormal(0, 0.5, n_points)
        
        # Percentage of supply not moved for 1+ years
        long_term_holders_base = 65  # 65% of supply
        lth_changes = np.random.normal(0, 0.5, n_points)
        long_term_holders = long_term_holders_base + np.cumsum(lth_changes) * 0.01
        long_term_holders = np.clip(long_term_holders, 50, 80)
        
        # Whale metrics
        whale_addresses_base = 2_200  # Addresses with 1000+ BTC
        whale_addresses = whale_addresses_base + np.random.randint(-10, 10, n_points)
        
        # Network value metrics
        nvt_ratio_base = 55  # Network Value to Transactions
        nvt_ratio = nvt_ratio_base * np.random.lognormal(0, 0.3, n_points)
        
        mvrv_ratio_base = 1.8  # Market Value to Realized Value
        mvrv_ratio = mvrv_ratio_base * np.random.lognormal(0, 0.2, n_points)
        
        onchain_df = pd.DataFrame({
            'timestamp': date_range,
            'hash_rate': hash_rate,
            'mining_difficulty': difficulty,
            'active_addresses': active_addresses.astype(int),
            'transaction_count': tx_count.astype(int),
            'transaction_volume_btc': tx_volume,
            'exchange_inflow': exchange_inflow,
            'exchange_outflow': exchange_outflow,
            'net_exchange_flow': exchange_inflow - exchange_outflow,  # Negative = outflow (bullish)
            'coin_days_destroyed': coin_days_destroyed,
            'long_term_holders_pct': long_term_holders,
            'whale_addresses': whale_addresses.astype(int),
            'nvt_ratio': nvt_ratio,
            'mvrv_ratio': mvrv_ratio,
            'hash_rate_momentum': np.gradient(hash_rate),  # Hash rate change
            'address_activity_ratio': active_addresses / 1_000_000,  # Normalized activity
        })
        
        # Convert timestamp to unix
        onchain_df['timestamp'] = onchain_df['timestamp'].astype(int) // 10**9
        
        self._cache_data(cache_key, onchain_df)
        return onchain_df
    
    def fetch_real_glassnode_data(self, metric: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Template for real Glassnode API integration
        """
        if not self.config.api_key:
            raise ValueError("Glassnode API key required")
        
        self._respect_rate_limit()
        
        url = f"https://api.glassnode.com/v1/metrics/{metric}"
        params = {
            'a': 'BTC',  # Asset
            'api_key': self.config.api_key,
            'since': start_date,
            'until': end_date,
            'format': 'JSON'
        }
        
        # This would be the actual API call
        # response = requests.get(url, params=params)
        # data = response.json()
        
        # For now, return simulated data
        return self.fetch_data(start_date, end_date)


class ExternalDataManager:
    """
    Manages all external data sources and provides unified interface
    """
    
    def __init__(self, config_file: str = None):
        self.data_sources = {}
        self.config = self._load_config(config_file)
        self._initialize_data_sources()
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'sentiment': {
                'enabled': True,
                'api_key': None,
                'rate_limit': 1.0,
                'cache_duration': 3600
            },
            'macro': {
                'enabled': True,
                'api_key': None,  # FRED API key
                'rate_limit': 2.0,
                'cache_duration': 86400  # Daily data, cache longer
            },
            'onchain': {
                'enabled': True,
                'api_key': None,  # Glassnode API key
                'rate_limit': 1.0,
                'cache_duration': 3600
            }
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                # Merge user config with defaults
                for source, settings in user_config.items():
                    if source in default_config:
                        default_config[source].update(settings)
        
        return default_config
    
    def _initialize_data_sources(self):
        """Initialize all enabled data sources"""
        if self.config['sentiment']['enabled']:
            sentiment_config = DataSourceConfig(
                name='sentiment',
                api_key=self.config['sentiment'].get('api_key'),
                rate_limit=self.config['sentiment']['rate_limit'],
                cache_duration=self.config['sentiment']['cache_duration']
            )
            self.data_sources['sentiment'] = SentimentDataSource(sentiment_config)
        
        if self.config['macro']['enabled']:
            macro_config = DataSourceConfig(
                name='macro',
                api_key=self.config['macro'].get('api_key'),
                base_url='https://api.stlouisfed.org/fred',
                rate_limit=self.config['macro']['rate_limit'],
                cache_duration=self.config['macro']['cache_duration']
            )
            self.data_sources['macro'] = MacroEconomicDataSource(macro_config)
        
        if self.config['onchain']['enabled']:
            onchain_config = DataSourceConfig(
                name='onchain',
                api_key=self.config['onchain'].get('api_key'),
                base_url='https://api.glassnode.com/v1',
                rate_limit=self.config['onchain']['rate_limit'],
                cache_duration=self.config['onchain']['cache_duration']
            )
            self.data_sources['onchain'] = OnChainDataSource(onchain_config)
    
    def fetch_all_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch data from all enabled sources"""
        all_data = {}
        
        for source_name, source in self.data_sources.items():
            try:
                print(f"Fetching {source_name} data...")
                data = source.fetch_data(start_date, end_date)
                all_data[source_name] = data
                print(f"✓ {source_name}: {len(data)} records")
            except Exception as e:
                print(f"✗ Error fetching {source_name} data: {e}")
                all_data[source_name] = pd.DataFrame()
        
        return all_data
    
    def align_and_merge_data(self, crypto_data: pd.DataFrame, 
                           external_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Align and merge external data with cryptocurrency price data
        """
        print("Aligning and merging external data...")
        
        # Start with crypto data
        merged_data = crypto_data.copy()
        
        # Ensure timestamp column exists and is properly formatted
        if 'timestamp' not in merged_data.columns:
            raise ValueError("Crypto data must have 'timestamp' column")
        
        # Convert crypto timestamp to datetime for alignment
        crypto_timestamps = pd.to_datetime(merged_data['timestamp'], unit='s')
        merged_data['datetime'] = crypto_timestamps
        
        total_features_added = 0
        
        for source_name, ext_data in external_data.items():
            if ext_data.empty:
                print(f"Skipping empty {source_name} data")
                continue
            
            print(f"Merging {source_name} data...")
            
            # Convert external data timestamp to datetime
            ext_data_copy = ext_data.copy()
            ext_data_copy['datetime'] = pd.to_datetime(ext_data_copy['timestamp'], unit='s')
            
            # Sort both dataframes by datetime
            merged_data = merged_data.sort_values('datetime')
            ext_data_copy = ext_data_copy.sort_values('datetime')
            
            # Merge using pandas merge_asof for time-series alignment
            # This aligns each crypto timestamp with the most recent external data point
            feature_cols = [col for col in ext_data_copy.columns if col not in ['timestamp', 'datetime']]
            
            merged_data = pd.merge_asof(
                merged_data,
                ext_data_copy[['datetime'] + feature_cols],
                on='datetime',
                direction='backward',  # Use most recent external data
                suffixes=('', f'_{source_name}')
            )
            
            features_added = len(feature_cols)
            total_features_added += features_added
            print(f"  Added {features_added} {source_name} features")
        
        # Clean up
        merged_data = merged_data.drop('datetime', axis=1)
        merged_data = merged_data.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✓ Merged external data: {total_features_added} total features added")
        print(f"Final dataset shape: {merged_data.shape}")
        
        return merged_data
    
    def analyze_feature_impact(self, merged_data: pd.DataFrame, 
                             target_col: str = 'close') -> pd.DataFrame:
        """Analyze impact of external features on target variable"""
        if target_col not in merged_data.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Calculate correlations
        numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
        correlations = merged_data[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
        
        # Identify external features (those with source suffixes)
        external_features = []
        for col in correlations.index:
            if any(suffix in col for suffix in ['_sentiment', '_macro', '_onchain']):
                external_features.append(col)
        
        # Create impact analysis
        impact_analysis = pd.DataFrame({
            'feature': correlations.index,
            'correlation': correlations.values,
            'is_external': [col in external_features for col in correlations.index],
            'source': ['external' if col in external_features else 'crypto' for col in correlations.index]
        })
        
        print("\nTop 10 External Features by Correlation:")
        external_features_df = impact_analysis[impact_analysis['is_external']].head(10)
        print(external_features_df.to_string(index=False))
        
        return impact_analysis
    
    def save_config_template(self, output_file: str = 'external_data_config.json'):
        """Save a configuration template for API keys"""
        config_template = {
            "sentiment": {
                "enabled": True,
                "api_key": "YOUR_TWITTER_BEARER_TOKEN",
                "rate_limit": 1.0,
                "cache_duration": 3600
            },
            "macro": {
                "enabled": True,
                "api_key": "YOUR_FRED_API_KEY",
                "rate_limit": 2.0,
                "cache_duration": 86400
            },
            "onchain": {
                "enabled": True,
                "api_key": "YOUR_GLASSNODE_API_KEY",
                "rate_limit": 1.0,
                "cache_duration": 3600
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(config_template, f, indent=2)
        
        print(f"Configuration template saved to {output_file}")
        print("Edit this file with your API keys to enable real data fetching")


# Convenience function for quick setup
def create_external_data_manager(config_file: str = None) -> ExternalDataManager:
    """Create and return an ExternalDataManager instance"""
    return ExternalDataManager(config_file)