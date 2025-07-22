import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional
import json
from textblob import TextBlob
import warnings

warnings.filterwarnings('ignore')


class RedditSentimentAPI:
    """
    Real Reddit API integration for Bitcoin sentiment analysis
    Uses Reddit's public API - completely FREE
    """
    
    def __init__(self):
        self.base_url = "https://www.reddit.com/r"
        self.subreddits = ["Bitcoin", "cryptocurrency", "CryptoMarkets"]
        self.bitcoin_keywords = ["bitcoin", "btc", "crypto", "cryptocurrency"]
        
    def fetch_reddit_sentiment(self, start_date: str, end_date: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch real Reddit sentiment data for Bitcoin
        """
        print(f"Fetching Reddit sentiment from {start_date} to {end_date}")
        
        all_posts = []
        
        for subreddit in self.subreddits:
            try:
                # Get hot posts from subreddit
                hot_url = f"{self.base_url}/{subreddit}/hot.json?limit={limit}"
                headers = {'User-Agent': 'TimeLLM-Cryptex/1.0'}
                
                response = requests.get(hot_url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    posts = data['data']['children']
                    
                    for post in posts:
                        post_data = post['data']
                        
                        # Filter for Bitcoin-related posts
                        title = post_data.get('title', '').lower()
                        text = post_data.get('selftext', '').lower()
                        
                        if any(keyword in title or keyword in text for keyword in self.bitcoin_keywords):
                            
                            # Analyze sentiment using TextBlob
                            full_text = f"{post_data.get('title', '')} {post_data.get('selftext', '')}"
                            sentiment = TextBlob(full_text).sentiment
                            
                            all_posts.append({
                                'timestamp': post_data.get('created_utc', time.time()),
                                'subreddit': subreddit,
                                'title': post_data.get('title', ''),
                                'score': post_data.get('score', 0),
                                'num_comments': post_data.get('num_comments', 0),
                                'upvote_ratio': post_data.get('upvote_ratio', 0.5),
                                'sentiment_polarity': sentiment.polarity,  # -1 to 1
                                'sentiment_subjectivity': sentiment.subjectivity,  # 0 to 1
                            })
                    
                    print(f"  Fetched {len(posts)} posts from r/{subreddit}")
                    
                else:
                    print(f"  Failed to fetch from r/{subreddit}: {response.status_code}")
                    
                # Rate limiting - be respectful to Reddit
                time.sleep(1)
                
            except Exception as e:
                print(f"  Error fetching from r/{subreddit}: {e}")
                continue
        
        if not all_posts:
            print("No Reddit posts found")
            return pd.DataFrame()
        
        # Convert to DataFrame and process
        df = pd.DataFrame(all_posts)
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Create aggregated sentiment metrics
        sentiment_metrics = self._aggregate_reddit_sentiment(df)
        
        print(f"Processed {len(all_posts)} Reddit posts into sentiment metrics")
        
        return sentiment_metrics
    
    def _aggregate_reddit_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate Reddit posts into hourly sentiment metrics"""
        
        # Group by hour
        df['hour'] = df['datetime'].dt.floor('H')
        
        # Aggregate metrics
        hourly_sentiment = df.groupby('hour').agg({
            'sentiment_polarity': ['mean', 'std', 'count'],
            'sentiment_subjectivity': 'mean',
            'score': ['mean', 'sum'],
            'num_comments': ['mean', 'sum'],
            'upvote_ratio': 'mean'
        }).reset_index()
        
        # Flatten column names
        hourly_sentiment.columns = [
            'timestamp', 'sentiment_score', 'sentiment_volatility', 'post_count',
            'subjectivity', 'avg_score', 'total_score', 'avg_comments', 'total_comments',
            'avg_upvote_ratio'
        ]
        
        # Convert sentiment to 0-1 scale (from -1 to 1)
        hourly_sentiment['sentiment_score'] = (hourly_sentiment['sentiment_score'] + 1) / 2
        
        # Fill NaN values
        hourly_sentiment = hourly_sentiment.fillna(0.5)
        
        # Convert timestamp to unix
        hourly_sentiment['timestamp'] = hourly_sentiment['timestamp'].astype(int) // 10**9
        
        return hourly_sentiment


class NewsAPISentiment:
    """
    Real NewsAPI integration for crypto news sentiment
    Free tier: 500 requests/day
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key  # Get from newsapi.org
        self.base_url = "https://newsapi.org/v2/everything"
        self.bitcoin_keywords = "bitcoin OR cryptocurrency OR crypto OR BTC"
        
    def fetch_news_sentiment(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch real news sentiment for Bitcoin
        """
        if not self.api_key:
            print("NewsAPI key not provided - skipping news sentiment")
            return pd.DataFrame()
        
        print(f"Fetching news sentiment from {start_date} to {end_date}")
        
        try:
            # API parameters
            params = {
                'q': self.bitcoin_keywords,
                'from': start_date,
                'to': end_date,
                'language': 'en',
                'sortBy': 'publishedAt',
                'apiKey': self.api_key,
                'pageSize': 100
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                news_data = []
                for article in articles:
                    # Analyze sentiment
                    text = f"{article.get('title', '')} {article.get('description', '')}"
                    sentiment = TextBlob(text).sentiment
                    
                    news_data.append({
                        'timestamp': pd.to_datetime(article.get('publishedAt')).timestamp(),
                        'title': article.get('title', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'sentiment_polarity': sentiment.polarity,
                        'sentiment_subjectivity': sentiment.subjectivity,
                    })
                
                if news_data:
                    df = pd.DataFrame(news_data)
                    sentiment_metrics = self._aggregate_news_sentiment(df)
                    print(f"Processed {len(articles)} news articles")
                    return sentiment_metrics
                else:
                    print("No news articles found")
                    return pd.DataFrame()
                    
            else:
                print(f"NewsAPI error: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching news sentiment: {e}")
            return pd.DataFrame()
    
    def _aggregate_news_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate news articles into hourly sentiment metrics"""
        
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['hour'] = df['datetime'].dt.floor('H')
        
        # Aggregate by hour
        hourly_news = df.groupby('hour').agg({
            'sentiment_polarity': ['mean', 'std', 'count'],
            'sentiment_subjectivity': 'mean'
        }).reset_index()
        
        # Flatten columns
        hourly_news.columns = [
            'timestamp', 'news_sentiment', 'news_sentiment_volatility', 
            'news_article_count', 'news_subjectivity'
        ]
        
        # Convert sentiment to 0-1 scale
        hourly_news['news_sentiment'] = (hourly_news['news_sentiment'] + 1) / 2
        hourly_news = hourly_news.fillna(0.5)
        
        # Convert timestamp to unix
        hourly_news['timestamp'] = hourly_news['timestamp'].astype(int) // 10**9
        
        return hourly_news


class FearGreedIndexAPI:
    """
    Real Fear & Greed Index API - completely FREE
    Official crypto market sentiment indicator
    """
    
    def __init__(self):
        self.base_url = "https://api.alternative.me/fng/"
        
    def fetch_fear_greed_index(self, days: int = 30) -> pd.DataFrame:
        """
        Fetch real Fear & Greed Index data
        """
        print(f"Fetching Fear & Greed Index for last {days} days")
        
        try:
            # API call
            params = {'limit': days, 'format': 'json'}
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('data'):
                    fear_greed_data = []
                    
                    for entry in data['data']:
                        fear_greed_data.append({
                            'timestamp': int(entry['timestamp']),
                            'fear_greed_index': int(entry['value']),
                            'fear_greed_classification': entry['value_classification']
                        })
                    
                    df = pd.DataFrame(fear_greed_data)
                    
                    # Convert to 0-1 scale (from 0-100)
                    df['fear_greed_normalized'] = df['fear_greed_index'] / 100
                    
                    print(f"Fetched {len(fear_greed_data)} Fear & Greed Index points")
                    return df
                else:
                    print("No Fear & Greed data available")
                    return pd.DataFrame()
                    
            else:
                print(f"Fear & Greed API error: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching Fear & Greed Index: {e}")
            return pd.DataFrame()


class RealSentimentManager:
    """
    Manager for all real sentiment APIs
    """
    
    def __init__(self, newsapi_key: str = None):
        self.reddit_api = RedditSentimentAPI()
        self.news_api = NewsAPISentiment(newsapi_key)
        self.fear_greed_api = FearGreedIndexAPI()
        
    def fetch_all_sentiment_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch sentiment data from all FREE sources
        """
        print("=" * 60)
        print("FETCHING REAL SENTIMENT DATA")
        print("=" * 60)
        
        sentiment_data = {}
        
        # 1. Reddit sentiment
        try:
            reddit_data = self.reddit_api.fetch_reddit_sentiment(start_date, end_date)
            if not reddit_data.empty:
                sentiment_data['reddit'] = reddit_data
        except Exception as e:
            print(f"Reddit API failed: {e}")
        
        # 2. News sentiment
        try:
            news_data = self.news_api.fetch_news_sentiment(start_date, end_date)
            if not news_data.empty:
                sentiment_data['news'] = news_data
        except Exception as e:
            print(f"News API failed: {e}")
        
        # 3. Fear & Greed Index
        try:
            fear_greed_data = self.fear_greed_api.fetch_fear_greed_index(30)
            if not fear_greed_data.empty:
                sentiment_data['fear_greed'] = fear_greed_data
        except Exception as e:
            print(f"Fear & Greed API failed: {e}")
        
        print(f"Successfully fetched data from {len(sentiment_data)} sentiment sources")
        
        return sentiment_data


if __name__ == "__main__":
    # Test the real sentiment APIs
    print("Testing Real Sentiment APIs")
    print("=" * 40)
    
    # Initialize manager
    manager = RealSentimentManager()  # No NewsAPI key for testing
    
    # Test date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    # Fetch real sentiment data
    sentiment_data = manager.fetch_all_sentiment_data(start_date, end_date)
    
    # Show results
    for source, data in sentiment_data.items():
        print(f"\n{source.upper()} DATA:")
        print(f"  Shape: {data.shape}")
        print(f"  Columns: {list(data.columns)}")
        if not data.empty:
            print(f"  Sample: {data.head(2).to_dict()}")
    
    print("\nReal sentiment API test completed!")