import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from utils.timefeatures import time_features
import warnings

# Add parent directory to path to import feature engineering
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_engineering import FeatureEngineer

warnings.filterwarnings('ignore')


class Dataset_CRYPTEX_Enhanced(Dataset):
    """
    Enhanced CRYPTEX dataset with comprehensive feature engineering
    """
    
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='candlesticks-D.csv',
                 target='close', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None, enable_feature_engineering=True,
                 feature_config=None, use_enhanced_data=True):
        
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
        self.use_enhanced_data = use_enhanced_data
        self.enable_feature_engineering = enable_feature_engineering
        
        # Use enhanced dataset if available, otherwise fall back to basic
        enhanced_path = 'cryptex/candlesticks-D-enhanced.csv'
        if self.use_enhanced_data and os.path.exists(os.path.join(root_path, enhanced_path)):
            self.data_path = enhanced_path
            self.enable_feature_engineering = False  # Disable feature engineering for enhanced datasets
            print("Using enhanced dataset with sentiment + macro + on-chain data")
        else:
            self.data_path = data_path
            if self.use_enhanced_data:
                print(f"Enhanced dataset not found at {os.path.join(root_path, enhanced_path)}, using basic dataset")
        
        # Initialize feature engineer
        if self.enable_feature_engineering:
            self.feature_engineer = FeatureEngineer(feature_config)
        
        self.__read_data__()
        
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        
        print(f"Dataset initialized: {flag} set with {len(self.data_x)} samples, {self.enc_in} features")
    
    def __read_data__(self):
        # Use RobustScaler for better outlier handling
        self.scaler = RobustScaler()
        
        # Read raw data
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # Ensure expected columns exist for basic validation
        expected_cols = ['timestamp', 'open', 'close', 'high', 'low', 'volume']
        if not all(col in df_raw.columns for col in expected_cols):
            raise ValueError(f"CSV must contain columns: {expected_cols}")
        
        # Only reorder/filter columns if not using enhanced data (to preserve all enhanced features)
        if not self.use_enhanced_data:
            df_raw = df_raw[expected_cols]
        
        # Data quality checks and cleaning
        df_raw = self._clean_data(df_raw)
        
        # Feature engineering
        if self.enable_feature_engineering:
            print("Applying feature engineering...")
            df_features = self.feature_engineer.create_features(df_raw)
            
            # Update target column name if needed
            if self.target == 'close' and 'close' in df_features.columns:
                pass  # Target already exists
            else:
                # Ensure target column exists
                if self.target not in df_features.columns:
                    raise ValueError(f"Target column '{self.target}' not found in features")
            
            df_processed = df_features
        else:
            df_processed = df_raw
            
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
            # Use all features except timestamp and datetime columns
            exclude_cols = ['timestamp', 'datetime']
            cols_data = [col for col in df_processed.columns if col not in exclude_cols]
            df_data = df_processed[cols_data]
            print(f"Selected {len(cols_data)} columns for training (excluded: {exclude_cols})")
        elif self.features == 'S':
            # Use only target feature
            df_data = df_processed[[self.target]]
        
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
    
    def get_feature_importance(self):
        """Get feature importance if feature engineering is enabled"""
        if self.enable_feature_engineering and hasattr(self, 'feature_engineer'):
            # Create temporary dataframe with features
            temp_df = pd.DataFrame(self.data_x, columns=self.feature_names)
            temp_df[self.target] = self.data_y[:, 0]  # Assuming target is first column
            return self.feature_engineer.get_feature_importance(temp_df, self.target)
        return None


class Dataset_CRYPTEX_MultiScale(Dataset):
    """
    Multi-scale CRYPTEX dataset that combines multiple timeframes
    """
    
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_paths=['candlesticks-h.csv', 'candlesticks-D.csv'],
                 target='close', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None, enable_feature_engineering=True):
        
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
        self.data_paths = data_paths
        self.enable_feature_engineering = enable_feature_engineering
        
        # Initialize feature engineer
        if self.enable_feature_engineering:
            self.feature_engineer = FeatureEngineer()
        
        self.__read_multi_scale_data__()
        
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        
        print(f"Multi-scale dataset initialized: {flag} set with {len(self.data_x)} samples, {self.enc_in} features")
    
    def __read_multi_scale_data__(self):
        """Read and combine multiple timeframe data"""
        all_features = []
        
        for data_path in self.data_paths:
            print(f"Processing {data_path}...")
            df_raw = pd.read_csv(os.path.join(self.root_path, data_path))
            
            # Basic cleaning
            expected_cols = ['timestamp', 'open', 'close', 'high', 'low', 'volume']
            if not all(col in df_raw.columns for col in expected_cols):
                print(f"Warning: {data_path} missing expected columns, skipping...")
                continue
            
            df_raw = df_raw[expected_cols]
            
            # Feature engineering for this timeframe
            if self.enable_feature_engineering:
                df_features = self.feature_engineer.create_features(df_raw)
                
                # Add timeframe suffix to distinguish features
                timeframe_suffix = data_path.split('.')[0].split('-')[-1]  # e.g., 'h' or 'D'
                df_features = df_features.add_suffix(f'_{timeframe_suffix}')
                
                # Keep timestamp without suffix
                if f'timestamp_{timeframe_suffix}' in df_features.columns:
                    df_features['timestamp'] = df_features[f'timestamp_{timeframe_suffix}']
                    df_features = df_features.drop(f'timestamp_{timeframe_suffix}', axis=1)
            else:
                df_features = df_raw
            
            all_features.append(df_features)
        
        # Combine all timeframes (align by timestamp)
        if len(all_features) > 1:
            combined_df = all_features[0]
            for df in all_features[1:]:
                combined_df = pd.merge(combined_df, df, on='timestamp', how='inner')
        else:
            combined_df = all_features[0]
        
        # Continue with standard processing
        num_train = int(len(combined_df) * 0.7)
        num_test = int(len(combined_df) * 0.2)
        num_vali = len(combined_df) - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, len(combined_df) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(combined_df)]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
        
        # Feature selection
        if self.features == 'M' or self.features == 'MS':
            # Use all features except timestamp and datetime columns
            exclude_cols = ['timestamp', 'datetime']
            cols_data = [col for col in combined_df.columns if col not in exclude_cols]
            df_data = combined_df[cols_data]
        elif self.features == 'S':
            # Find target column (may have suffix)
            target_cols = [col for col in combined_df.columns if col.startswith(self.target)]
            if target_cols:
                df_data = combined_df[target_cols[:1]]  # Use first match
            else:
                raise ValueError(f"Target column '{self.target}' not found")
        
        # Scaling
        self.scaler = RobustScaler()
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        # Time encoding
        df_stamp = combined_df[['timestamp']][border1:border2]
        df_stamp['timestamp'] = pd.to_datetime(df_stamp['timestamp'], unit='s')
        
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
        
        print(f"Multi-scale data shape: {self.data_x.shape}, Features: {len(self.feature_names)}")
    
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