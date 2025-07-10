import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler


class TechnicalIndicators:
    """
    Technical indicators for cryptocurrency price prediction
    """
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2.0) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return {
            'bb_upper': upper_band,
            'bb_middle': sma,
            'bb_lower': lower_band,
            'bb_width': (upper_band - lower_band) / sma,
            'bb_position': (data - lower_band) / (upper_band - lower_band)
        }
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                             k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            'stoch_k': k_percent,
            'stoch_d': d_percent
        }
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))


class VolatilityMeasures:
    """
    Volatility measures for cryptocurrency price prediction
    """
    
    @staticmethod
    def realized_volatility(returns: pd.Series, window: int = 30) -> pd.Series:
        """Realized Volatility (rolling standard deviation of returns)"""
        return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    @staticmethod
    def garch_volatility(returns: pd.Series, window: int = 100) -> pd.Series:
        """Simple GARCH(1,1) approximation"""
        # Simplified GARCH - for full implementation, use arch library
        variance = returns.rolling(window=window).var()
        return np.sqrt(variance * 252)  # Annualized
    
    @staticmethod
    def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 30) -> pd.Series:
        """Parkinson volatility estimator"""
        log_hl = np.log(high / low)
        parkinson_var = (log_hl ** 2) / (4 * np.log(2))
        return np.sqrt(parkinson_var.rolling(window=window).mean() * 252)
    
    @staticmethod
    def garman_klass_volatility(open_: pd.Series, high: pd.Series, low: pd.Series, 
                               close: pd.Series, window: int = 30) -> pd.Series:
        """Garman-Klass volatility estimator"""
        log_hl = np.log(high / low)
        log_cc = np.log(close / open_)
        
        gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_cc ** 2)
        return np.sqrt(gk_var.rolling(window=window).mean() * 252)


class MarketMicrostructure:
    """
    Market microstructure indicators
    """
    
    @staticmethod
    def price_impact(volume: pd.Series, returns: pd.Series, window: int = 20) -> pd.Series:
        """Price impact measure"""
        return abs(returns) / (volume.rolling(window=window).mean() + 1e-8)
    
    @staticmethod
    def volume_weighted_price(high: pd.Series, low: pd.Series, close: pd.Series, 
                             volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
    
    @staticmethod
    def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, 
                        volume: pd.Series, window: int = 14) -> pd.Series:
        """Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        
        positive_mf = positive_flow.rolling(window=window).sum()
        negative_mf = negative_flow.rolling(window=window).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi


class FeatureEngineer:
    """
    Main feature engineering class for cryptocurrency data
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.technical_indicators = TechnicalIndicators()
        self.volatility_measures = VolatilityMeasures()
        self.market_microstructure = MarketMicrostructure()
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        df_features = df.copy()
        
        # Price-based indicators
        df_features['sma_5'] = self.technical_indicators.sma(df['close'], 5)
        df_features['sma_10'] = self.technical_indicators.sma(df['close'], 10)
        df_features['sma_20'] = self.technical_indicators.sma(df['close'], 20)
        df_features['sma_50'] = self.technical_indicators.sma(df['close'], 50)
        
        df_features['ema_5'] = self.technical_indicators.ema(df['close'], 5)
        df_features['ema_10'] = self.technical_indicators.ema(df['close'], 10)
        df_features['ema_20'] = self.technical_indicators.ema(df['close'], 20)
        
        # RSI
        df_features['rsi_14'] = self.technical_indicators.rsi(df['close'], 14)
        df_features['rsi_30'] = self.technical_indicators.rsi(df['close'], 30)
        
        # MACD
        macd_data = self.technical_indicators.macd(df['close'])
        df_features['macd'] = macd_data['macd']
        df_features['macd_signal'] = macd_data['signal']
        df_features['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = self.technical_indicators.bollinger_bands(df['close'])
        df_features['bb_upper'] = bb_data['bb_upper']
        df_features['bb_middle'] = bb_data['bb_middle']
        df_features['bb_lower'] = bb_data['bb_lower']
        df_features['bb_width'] = bb_data['bb_width']
        df_features['bb_position'] = bb_data['bb_position']
        
        # Stochastic Oscillator
        stoch_data = self.technical_indicators.stochastic_oscillator(
            df['high'], df['low'], df['close'])
        df_features['stoch_k'] = stoch_data['stoch_k']
        df_features['stoch_d'] = stoch_data['stoch_d']
        
        # ATR
        df_features['atr'] = self.technical_indicators.atr(df['high'], df['low'], df['close'])
        
        # Williams %R
        df_features['williams_r'] = self.technical_indicators.williams_r(
            df['high'], df['low'], df['close'])
        
        return df_features
    
    def add_volatility_measures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility measures to the dataframe"""
        df_features = df.copy()
        
        # Calculate returns
        df_features['returns'] = df['close'].pct_change()
        df_features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Realized volatility
        df_features['realized_vol_10'] = self.volatility_measures.realized_volatility(
            df_features['returns'], 10)
        df_features['realized_vol_30'] = self.volatility_measures.realized_volatility(
            df_features['returns'], 30)
        
        # GARCH volatility approximation
        df_features['garch_vol'] = self.volatility_measures.garch_volatility(
            df_features['returns'])
        
        # Parkinson volatility
        df_features['parkinson_vol'] = self.volatility_measures.parkinson_volatility(
            df['high'], df['low'])
        
        # Garman-Klass volatility
        df_features['gk_vol'] = self.volatility_measures.garman_klass_volatility(
            df['open'], df['high'], df['low'], df['close'])
        
        return df_features
    
    def add_market_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        df_features = df.copy()
        
        # Price impact
        if 'returns' not in df_features.columns:
            df_features['returns'] = df['close'].pct_change()
        
        df_features['price_impact'] = self.market_microstructure.price_impact(
            df['volume'], df_features['returns'])
        
        # Volume weighted average price
        df_features['vwap'] = self.market_microstructure.volume_weighted_price(
            df['high'], df['low'], df['close'], df['volume'])
        
        # Money Flow Index
        df_features['mfi'] = self.market_microstructure.money_flow_index(
            df['high'], df['low'], df['close'], df['volume'])
        
        # Volume-based features
        df_features['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df_features['volume_ratio'] = df['volume'] / df_features['volume_sma_10']
        
        return df_features
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features based on timestamp"""
        df_features = df.copy()
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            df_features['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        else:
            df_features['datetime'] = df.index
            
        # Time-based features
        df_features['hour'] = df_features['datetime'].dt.hour
        df_features['day_of_week'] = df_features['datetime'].dt.dayofweek
        df_features['month'] = df_features['datetime'].dt.month
        df_features['quarter'] = df_features['datetime'].dt.quarter
        
        # Cyclical encoding
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        df_features['dow_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['dow_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        
        return df_features
    
    def add_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Add lag features for key variables"""
        df_features = df.copy()
        
        key_columns = ['close', 'volume', 'high', 'low']
        
        for col in key_columns:
            if col in df.columns:
                for lag in lags:
                    df_features[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df_features
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features"""
        print("Creating technical indicators...")
        df_features = self.add_technical_indicators(df)
        
        print("Adding volatility measures...")
        df_features = self.add_volatility_measures(df_features)
        
        print("Adding market microstructure features...")
        df_features = self.add_market_microstructure(df_features)
        
        print("Adding temporal features...")
        df_features = self.add_temporal_features(df_features)
        
        print("Adding lag features...")
        df_features = self.add_lag_features(df_features)
        
        # Remove rows with NaN values (caused by rolling windows and lags)
        df_features = df_features.dropna()
        
        print(f"Created {len(df_features.columns)} features from {len(df.columns)} original columns")
        
        return df_features
    
    def get_feature_importance(self, df: pd.DataFrame, target_col: str = 'close') -> pd.DataFrame:
        """Calculate feature importance using correlation analysis"""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        # Calculate correlations with target
        correlations = df.corr()[target_col].abs().sort_values(ascending=False)
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': correlations.index,
            'correlation': correlations.values
        })
        
        return feature_importance[feature_importance['feature'] != target_col]