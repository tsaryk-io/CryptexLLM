import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from utils.timefeatures import time_features
import warnings

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_engineering import FeatureEngineer
from utils.external_data_integration import ExternalDataManager

warnings.filterwarnings('ignore')


class Dataset_CRYPTEX_External(Dataset):
    """
    Enhanced CRYPTEX dataset with external data integration
    Combines price data with sentiment, macro, and on-chain data
    """
    
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='candlesticks-h.csv',
                 target='close', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None, enable_feature_engineering=True,
                 enable_external_data=True, external_config=None,
                 feature_config=None):
        
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        # Init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.enable_feature_engineering = enable_feature_engineering
        self.enable_external_data = enable_external_data
        
        # Initialize feature engineer
        if self.enable_feature_engineering:
            self.feature_engineer = FeatureEngineer(feature_config)
        
        # Initialize external data manager
        if self.enable_external_data:
            self.external_data_manager = ExternalDataManager(external_config)
        
        self.__read_data__()
        
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        
        print(f"External-Enhanced Dataset initialized: {flag} set with {len(self.data_x)} samples, {self.enc_in} features")
    
    def __read_data__(self):
        # Use RobustScaler for better outlier handling
        self.scaler = RobustScaler()
        
        # Read raw crypto data
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # Ensure expected columns exist and are properly ordered
        expected_cols = ['timestamp', 'open', 'close', 'high', 'low', 'volume']
        if not all(col in df_raw.columns for col in expected_cols):
            raise ValueError(f"CSV must contain columns: {expected_cols}")
        
        # Reorder columns to ensure consistent format
        df_raw = df_raw[expected_cols]
        
        # Data quality checks and cleaning
        df_raw = self._clean_data(df_raw)
        
        # Apply feature engineering to crypto data
        if self.enable_feature_engineering:
            print("Applying crypto feature engineering...")
            df_features = self.feature_engineer.create_features(df_raw)
        else:
            df_features = df_raw
        
        # Integrate external data
        if self.enable_external_data:
            df_features = self._integrate_external_data(df_features)
        
        df_processed = df_features
        
        # Split train/val/test
        num_train = int(len(df_processed) * 0.7)
        num_test = int(len(df_processed) * 0.2)
        num_vali = len(df_processed) - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, len(df_processed) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_processed)]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
        
        # Feature selection
        if self.features == 'M' or self.features == 'MS':
            # Use all features except timestamp
            if 'timestamp' in df_processed.columns:
                cols_data = [col for col in df_processed.columns if col != 'timestamp']
            else:
                cols_data = df_processed.columns[1:] if len(df_processed.columns) > 1 else df_processed.columns
            df_data = df_processed[cols_data]
        elif self.features == 'S':
            # Use only target feature
            df_data = df_processed[[self.target]]
        
        # Handle missing values in external data
        df_data = self._handle_missing_external_data(df_data)
        
        # Scaling
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        # Time encoding
        if 'timestamp' in df_processed.columns:
            df_stamp = df_processed[['timestamp']][border1:border2]
            df_stamp['timestamp'] = pd.to_datetime(df_stamp['timestamp'], unit='s')
        else:
            # Create timestamp column if it doesn't exist
            df_stamp = pd.DataFrame()
            df_stamp['timestamp'] = pd.date_range(start='2019-01-01', periods=len(df_processed), freq='H')[border1:border2]
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.timestamp.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.timestamp.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.timestamp.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.timestamp.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['timestamp'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['timestamp'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        
        # Store feature names for analysis
        self.feature_names = df_data.columns.tolist()
        
        print(f"Data shape: {self.data_x.shape}, Features: {len(self.feature_names)}")
        
        # Analyze feature importance if external data is enabled
        if self.enable_external_data and len(self.feature_names) > 10:
            self._analyze_external_feature_impact(df_processed)
    
    def _integrate_external_data(self, crypto_df: pd.DataFrame) -> pd.DataFrame:
        """Integrate external data sources with crypto data"""
        print("Integrating external data sources...")
        
        # Determine date range from crypto data
        min_timestamp = crypto_df['timestamp'].min()
        max_timestamp = crypto_df['timestamp'].max()
        
        # Convert to date strings
        start_date = pd.to_datetime(min_timestamp, unit='s').strftime('%Y-%m-%d')
        end_date = pd.to_datetime(max_timestamp, unit='s').strftime('%Y-%m-%d')
        
        print(f"Fetching external data for period: {start_date} to {end_date}")
        
        try:
            # Fetch all external data
            external_data = self.external_data_manager.fetch_all_data(start_date, end_date)
            
            # Merge with crypto data
            merged_df = self.external_data_manager.align_and_merge_data(crypto_df, external_data)
            
            return merged_df
            
        except Exception as e:
            print(f"Warning: External data integration failed: {e}")
            print("Continuing with crypto data only...")
            return crypto_df
    
    def _handle_missing_external_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in external data features"""
        # Identify external data columns
        external_cols = []
        for col in df.columns:
            if any(suffix in col for suffix in ['sentiment', 'macro', 'onchain']):
                external_cols.append(col)
        
        if not external_cols:
            return df
        
        print(f"Handling missing values in {len(external_cols)} external features...")
        
        # Forward fill then backward fill for external data
        df[external_cols] = df[external_cols].fillna(method='ffill').fillna(method='bfill')
        
        # If still missing, fill with column median
        for col in external_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  Filled {col} missing values with median: {median_val:.4f}")
        
        return df
    
    def _analyze_external_feature_impact(self, df: pd.DataFrame):
        """Analyze the impact of external features"""
        try:
            impact_analysis = self.external_data_manager.analyze_feature_impact(df, self.target)
            
            # Save analysis results
            output_dir = './results/external_analysis'
            os.makedirs(output_dir, exist_ok=True)
            impact_analysis.to_csv(f'{output_dir}/feature_impact_analysis.csv', index=False)
            
            # Print summary
            external_features = impact_analysis[impact_analysis['is_external']]
            if len(external_features) > 0:
                avg_external_corr = external_features['correlation'].mean()
                crypto_features = impact_analysis[~impact_analysis['is_external']]
                avg_crypto_corr = crypto_features['correlation'].mean()
                
                print(f"\nExternal Feature Impact Summary:")
                print(f"  Average external feature correlation: {avg_external_corr:.4f}")
                print(f"  Average crypto feature correlation: {avg_crypto_corr:.4f}")
                print(f"  External data improvement ratio: {avg_external_corr/avg_crypto_corr:.2f}x")
                
        except Exception as e:
            print(f"Warning: Feature impact analysis failed: {e}")
    
    def _clean_data(self, df):
        """Clean and validate the dataset"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        if df.isnull().any().any():
            print("Warning: Missing values detected, forward-filling...")
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove obvious outliers (price spikes/drops > 50%)
        for price_col in ['open', 'high', 'low', 'close']:
            if price_col in df.columns:
                price_change = df[price_col].pct_change().abs()
                outliers = price_change > 0.5
                if outliers.any():
                    print(f"Warning: {outliers.sum()} outliers detected in {price_col}, capping...")
                    # Cap outliers at 50% change
                    df.loc[outliers, price_col] = df[price_col].shift(1)[outliers] * 1.5
        
        # Ensure volume is non-negative
        if 'volume' in df.columns:
            df['volume'] = df['volume'].clip(lower=0)
        
        # Ensure OHLC relationships are maintained
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # High should be >= max(open, close)
            df['high'] = df[['high', 'open', 'close']].max(axis=1)
            # Low should be <= min(open, close)
            df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        return df
    
    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in
    
    def get_feature_names(self):
        """Return list of feature names"""
        return getattr(self, 'feature_names', [])
    
    def get_external_feature_summary(self):
        """Get summary of external features"""
        if not hasattr(self, 'feature_names'):
            return {}
        
        summary = {
            'total_features': len(self.feature_names),
            'crypto_features': 0,
            'sentiment_features': 0,
            'macro_features': 0,
            'onchain_features': 0
        }
        
        for feature in self.feature_names:
            if 'sentiment' in feature:
                summary['sentiment_features'] += 1
            elif 'macro' in feature:
                summary['macro_features'] += 1
            elif 'onchain' in feature:
                summary['onchain_features'] += 1
            else:
                summary['crypto_features'] += 1
        
        return summary
    
    def get_feature_importance(self):
        """Get feature importance if feature engineering is enabled"""
        if self.enable_feature_engineering and hasattr(self, 'feature_engineer'):
            # Create temporary dataframe with features
            temp_df = pd.DataFrame(self.data_x, columns=self.feature_names)
            temp_df[self.target] = self.data_y[:, 0]  # Assuming target is first column
            return self.feature_engineer.get_feature_importance(temp_df, self.target)
        return None


class Dataset_CRYPTEX_Regime_Aware(Dataset_CRYPTEX_External):
    """
    Regime-aware dataset that adapts feature selection based on market conditions
    """
    
    def __init__(self, *args, **kwargs):
        # Add regime detection parameters
        self.regime_detection_window = kwargs.pop('regime_detection_window', 30)
        self.volatility_threshold_low = kwargs.pop('volatility_threshold_low', 0.02)
        self.volatility_threshold_high = kwargs.pop('volatility_threshold_high', 0.05)
        
        super().__init__(*args, **kwargs)
        
        # Detect market regimes
        self._detect_market_regimes()
    
    def _detect_market_regimes(self):
        """Detect bull/bear/sideways market regimes"""
        if not hasattr(self, 'data_x') or len(self.feature_names) == 0:
            return
        
        print("Detecting market regimes...")
        
        # Find close price column
        close_col_idx = None
        for i, name in enumerate(self.feature_names):
            if name == 'close' or name.endswith('close'):
                close_col_idx = i
                break
        
        if close_col_idx is None:
            print("Warning: Could not find close price for regime detection")
            return
        
        # Extract close prices
        close_prices = self.data_x[:, close_col_idx]
        
        # Calculate rolling returns and volatility
        window = self.regime_detection_window
        returns = np.diff(close_prices) / close_prices[:-1]
        
        # Rolling statistics
        rolling_returns = []
        rolling_volatility = []
        
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            rolling_returns.append(np.mean(window_returns))
            rolling_volatility.append(np.std(window_returns))
        
        rolling_returns = np.array(rolling_returns)
        rolling_volatility = np.array(rolling_volatility)
        
        # Classify regimes
        regimes = []
        for i in range(len(rolling_returns)):
            ret = rolling_returns[i]
            vol = rolling_volatility[i]
            
            if vol < self.volatility_threshold_low:
                if ret > 0.001:  # 0.1% daily return threshold
                    regime = 'bull_low_vol'
                elif ret < -0.001:
                    regime = 'bear_low_vol'
                else:
                    regime = 'sideways_low_vol'
            elif vol > self.volatility_threshold_high:
                if ret > 0:
                    regime = 'bull_high_vol'
                elif ret < 0:
                    regime = 'bear_high_vol'
                else:
                    regime = 'sideways_high_vol'
            else:  # Medium volatility
                if ret > 0.0005:
                    regime = 'bull_med_vol'
                elif ret < -0.0005:
                    regime = 'bear_med_vol'
                else:
                    regime = 'sideways_med_vol'
            
            regimes.append(regime)
        
        # Pad with first regime for missing data
        regimes = [regimes[0]] * (len(close_prices) - len(regimes)) + regimes
        
        self.market_regimes = regimes
        
        # Print regime distribution
        regime_counts = {}
        for regime in regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        print("Market regime distribution:")
        for regime, count in sorted(regime_counts.items()):
            pct = count / len(regimes) * 100
            print(f"  {regime}: {count} periods ({pct:.1f}%)")
    
    def get_regime_at_index(self, index):
        """Get market regime at specific index"""
        if hasattr(self, 'market_regimes') and index < len(self.market_regimes):
            return self.market_regimes[index]
        return 'unknown'
    
    def get_regime_statistics(self):
        """Get statistics for each market regime"""
        if not hasattr(self, 'market_regimes'):
            return {}
        
        # Find close price column
        close_col_idx = None
        for i, name in enumerate(self.feature_names):
            if name == 'close' or name.endswith('close'):
                close_col_idx = i
                break
        
        if close_col_idx is None:
            return {}
        
        close_prices = self.data_x[:, close_col_idx]
        returns = np.diff(close_prices) / close_prices[:-1]
        
        regime_stats = {}
        unique_regimes = set(self.market_regimes)
        
        for regime in unique_regimes:
            # Find indices for this regime
            regime_indices = [i for i, r in enumerate(self.market_regimes) if r == regime]
            
            if len(regime_indices) > 1:
                regime_returns = [returns[i] for i in regime_indices[:-1]]  # Exclude last to match returns length
                
                regime_stats[regime] = {
                    'count': len(regime_indices),
                    'avg_return': np.mean(regime_returns),
                    'volatility': np.std(regime_returns),
                    'sharpe_ratio': np.mean(regime_returns) / (np.std(regime_returns) + 1e-8)
                }
        
        return regime_stats