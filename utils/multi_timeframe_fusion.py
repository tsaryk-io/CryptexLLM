import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TimeframeType(Enum):
    """Timeframe types for multi-scale analysis"""
    MINUTE = "1min"
    FIVE_MINUTE = "5min"
    FIFTEEN_MINUTE = "15min"
    HOURLY = "1h"
    FOUR_HOURLY = "4h"
    DAILY = "1D"
    WEEKLY = "1W"


@dataclass
class TimeframeConfig:
    """Configuration for each timeframe"""
    timeframe: TimeframeType
    weight: float
    seq_len: int
    pred_len: int
    features: List[str]
    importance: float = 1.0


class TemporalHierarchy:
    """
    Manages temporal hierarchy relationships between timeframes
    """
    
    def __init__(self):
        # Define hierarchy (lower index = higher frequency)
        self.hierarchy = [
            TimeframeType.MINUTE,
            TimeframeType.FIVE_MINUTE, 
            TimeframeType.FIFTEEN_MINUTE,
            TimeframeType.HOURLY,
            TimeframeType.FOUR_HOURLY,
            TimeframeType.DAILY,
            TimeframeType.WEEKLY
        ]
        
        # Conversion ratios
        self.conversion_ratios = {
            (TimeframeType.MINUTE, TimeframeType.FIVE_MINUTE): 5,
            (TimeframeType.MINUTE, TimeframeType.FIFTEEN_MINUTE): 15,
            (TimeframeType.MINUTE, TimeframeType.HOURLY): 60,
            (TimeframeType.FIVE_MINUTE, TimeframeType.FIFTEEN_MINUTE): 3,
            (TimeframeType.FIVE_MINUTE, TimeframeType.HOURLY): 12,
            (TimeframeType.FIFTEEN_MINUTE, TimeframeType.HOURLY): 4,
            (TimeframeType.HOURLY, TimeframeType.FOUR_HOURLY): 4,
            (TimeframeType.HOURLY, TimeframeType.DAILY): 24,
            (TimeframeType.FOUR_HOURLY, TimeframeType.DAILY): 6,
            (TimeframeType.DAILY, TimeframeType.WEEKLY): 7
        }
    
    def get_parent_timeframes(self, timeframe: TimeframeType) -> List[TimeframeType]:
        """Get higher-level (parent) timeframes"""
        current_idx = self.hierarchy.index(timeframe)
        return self.hierarchy[current_idx + 1:]
    
    def get_child_timeframes(self, timeframe: TimeframeType) -> List[TimeframeType]:
        """Get lower-level (child) timeframes"""
        current_idx = self.hierarchy.index(timeframe)
        return self.hierarchy[:current_idx]
    
    def get_conversion_ratio(self, from_tf: TimeframeType, to_tf: TimeframeType) -> Optional[int]:
        """Get conversion ratio between timeframes"""
        return self.conversion_ratios.get((from_tf, to_tf))


class CrossTimeframeAttention(nn.Module):
    """
    Attention mechanism for fusing information across timeframes
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0
        
        # Timeframe-specific projections
        self.timeframe_query = nn.Linear(d_model, d_model)
        self.timeframe_key = nn.Linear(d_model, d_model)
        self.timeframe_value = nn.Linear(d_model, d_model)
        
        # Cross-timeframe fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Timeframe importance weights
        self.timeframe_importance = nn.Parameter(torch.ones(7))  # 7 timeframes
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, timeframe_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse features from multiple timeframes
        
        Args:
            timeframe_features: Dict mapping timeframe names to feature tensors
                               Each tensor shape: [batch, seq_len, d_model]
        Returns:
            Fused representation: [batch, seq_len, d_model]
        """
        if not timeframe_features:
            raise ValueError("No timeframe features provided")
        
        # Stack timeframe features
        tf_names = list(timeframe_features.keys())
        tf_tensors = [timeframe_features[name] for name in tf_names]
        
        # Ensure all tensors have same sequence length (use shortest)
        min_seq_len = min(tensor.shape[1] for tensor in tf_tensors)
        tf_tensors = [tensor[:, :min_seq_len, :] for tensor in tf_tensors]
        
        # Stack: [batch, n_timeframes, seq_len, d_model]
        stacked_features = torch.stack(tf_tensors, dim=1)
        batch_size, n_tf, seq_len, d_model = stacked_features.shape
        
        # Apply timeframe importance weights
        importance_weights = F.softmax(self.timeframe_importance[:n_tf], dim=0)
        weighted_features = stacked_features * importance_weights.view(1, -1, 1, 1)
        
        # Reshape for cross-attention: [batch * seq_len, n_timeframes, d_model]
        reshaped = weighted_features.transpose(1, 2).contiguous().view(
            batch_size * seq_len, n_tf, d_model
        )
        
        # Apply cross-timeframe attention
        attended, attention_weights = self.cross_attention(
            reshaped, reshaped, reshaped
        )
        
        # Aggregate across timeframes (mean pooling)
        fused = attended.mean(dim=1)  # [batch * seq_len, d_model]
        
        # Reshape back: [batch, seq_len, d_model]
        fused = fused.view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.output_proj(fused)
        output = self.dropout(output)
        
        return output


class AdaptiveTimeframeFusion(nn.Module):
    """
    Adaptive fusion that learns to weight timeframes based on market conditions
    """
    
    def __init__(self, d_model: int, n_timeframes: int = 3):
        super().__init__()
        self.d_model = d_model
        self.n_timeframes = n_timeframes
        
        # Market regime detection
        self.regime_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3),  # 3 regimes: bull, bear, sideways
            nn.Softmax(dim=-1)
        )
        
        # Regime-specific timeframe weights
        self.regime_weights = nn.Parameter(torch.ones(3, n_timeframes))
        
        # Volatility-based adaptation
        self.volatility_adaptation = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(), 
            nn.Linear(d_model // 4, n_timeframes),
            nn.Softmax(dim=-1)
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * n_timeframes, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, timeframe_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Adaptively fuse timeframe features based on market conditions
        
        Args:
            timeframe_features: List of tensors [batch, seq_len, d_model]
        Returns:
            Fused features: [batch, seq_len, d_model]
        """
        if len(timeframe_features) != self.n_timeframes:
            raise ValueError(f"Expected {self.n_timeframes} timeframes, got {len(timeframe_features)}")
        
        batch_size, seq_len, d_model = timeframe_features[0].shape
        
        # Use highest frequency timeframe for regime detection
        primary_features = timeframe_features[0]  # Assume first is highest frequency
        
        # Detect market regime
        regime_probs = self.regime_detector(primary_features.mean(dim=1))  # [batch, 3]
        
        # Get regime-specific weights
        regime_specific_weights = torch.matmul(regime_probs, self.regime_weights)  # [batch, n_timeframes]
        
        # Volatility-based adaptation
        volatility_weights = self.volatility_adaptation(primary_features.mean(dim=1))  # [batch, n_timeframes]
        
        # Combine regime and volatility weights
        combined_weights = (regime_specific_weights + volatility_weights) / 2.0
        combined_weights = F.softmax(combined_weights, dim=-1)
        
        # Apply adaptive weights to timeframe features
        weighted_features = []
        for i, features in enumerate(timeframe_features):
            weight = combined_weights[:, i].unsqueeze(1).unsqueeze(2)  # [batch, 1, 1]
            weighted = features * weight
            weighted_features.append(weighted)
        
        # Concatenate for fusion
        concatenated = torch.cat(weighted_features, dim=-1)  # [batch, seq_len, d_model * n_timeframes]
        
        # Final fusion
        fused = self.fusion_layer(concatenated)
        
        return fused


class HierarchicalPredictionReconciliation:
    """
    Reconciles predictions across timeframe hierarchy to ensure consistency
    """
    
    def __init__(self, hierarchy: TemporalHierarchy):
        self.hierarchy = hierarchy
        
    def reconcile_predictions(self, predictions: Dict[TimeframeType, np.ndarray]) -> Dict[TimeframeType, np.ndarray]:
        """
        Reconcile predictions to satisfy hierarchical constraints
        
        Args:
            predictions: Dict mapping timeframes to prediction arrays
        Returns:
            Reconciled predictions
        """
        reconciled = predictions.copy()
        
        # Sort timeframes by hierarchy (high frequency to low frequency)
        sorted_timeframes = sorted(
            predictions.keys(),
            key=lambda tf: self.hierarchy.hierarchy.index(tf)
        )
        
        # Bottom-up reconciliation
        for i in range(len(sorted_timeframes) - 1):
            child_tf = sorted_timeframes[i]
            parent_tf = sorted_timeframes[i + 1]
            
            # Get conversion ratio
            ratio = self.hierarchy.get_conversion_ratio(child_tf, parent_tf)
            
            if ratio is not None:
                # Aggregate child predictions to match parent timeframe
                child_pred = reconciled[child_tf]
                
                # Reshape and aggregate
                if len(child_pred.shape) == 1:
                    # 1D case
                    n_periods = len(child_pred) // ratio
                    reshaped = child_pred[:n_periods * ratio].reshape(n_periods, ratio)
                    aggregated = reshaped.mean(axis=1)
                else:
                    # Multi-dimensional case
                    n_periods = child_pred.shape[0] // ratio
                    reshaped = child_pred[:n_periods * ratio].reshape(n_periods, ratio, -1)
                    aggregated = reshaped.mean(axis=1)
                
                # Blend with original parent prediction
                if parent_tf in reconciled:
                    parent_pred = reconciled[parent_tf]
                    min_len = min(len(aggregated), len(parent_pred))
                    # Weighted average (70% original, 30% aggregated)
                    reconciled[parent_tf][:min_len] = (
                        0.7 * parent_pred[:min_len] + 
                        0.3 * aggregated[:min_len]
                    )
                else:
                    reconciled[parent_tf] = aggregated
        
        return reconciled


class MultiTimeframeFusionSystem:
    """
    Complete multi-timeframe fusion system
    """
    
    def __init__(self, d_model: int = 128, n_heads: int = 8):
        self.d_model = d_model
        self.hierarchy = TemporalHierarchy()
        self.reconciliation = HierarchicalPredictionReconciliation(self.hierarchy)
        
        # Neural fusion components
        self.cross_attention = CrossTimeframeAttention(d_model, n_heads)
        self.adaptive_fusion = AdaptiveTimeframeFusion(d_model, n_timeframes=3)
        
    def create_timeframe_dataset(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Create aligned dataset from multiple timeframes
        
        Args:
            data_dict: Dict mapping timeframe names to dataframes
        Returns:
            Aligned feature arrays
        """
        aligned_data = {}
        
        # Find common time range
        min_start_time = max(df['timestamp'].min() for df in data_dict.values())
        max_end_time = min(df['timestamp'].max() for df in data_dict.values())
        
        for tf_name, df in data_dict.items():
            # Filter to common time range
            mask = (df['timestamp'] >= min_start_time) & (df['timestamp'] <= max_end_time)
            filtered_df = df[mask].copy()
            
            # Extract features (exclude timestamp)
            feature_cols = [col for col in filtered_df.columns if col != 'timestamp']
            features = filtered_df[feature_cols].values
            
            aligned_data[tf_name] = features
        
        return aligned_data
    
    def fuse_timeframes_neural(self, timeframe_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Neural fusion of timeframe features
        """
        # Cross-timeframe attention
        fused_attention = self.cross_attention(timeframe_features)
        
        # If we have exactly 3 timeframes, use adaptive fusion too
        if len(timeframe_features) == 3:
            tf_list = list(timeframe_features.values())
            # Ensure same sequence length
            min_len = min(t.shape[1] for t in tf_list)
            tf_list = [t[:, :min_len, :] for t in tf_list]
            
            fused_adaptive = self.adaptive_fusion(tf_list)
            
            # Combine both fusion methods
            combined = (fused_attention[:, :min_len, :] + fused_adaptive) / 2.0
            return combined
        
        return fused_attention
    
    def generate_multiscale_predictions(self, model, data_loaders: Dict[str, torch.utils.data.DataLoader]) -> Dict[str, np.ndarray]:
        """
        Generate predictions for all timeframes
        """
        predictions = {}
        
        for tf_name, data_loader in data_loaders.items():
            tf_predictions = []
            
            model.eval()
            with torch.no_grad():
                for batch in data_loader:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    
                    # Generate prediction
                    outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    
                    # Extract prediction
                    if isinstance(outputs, tuple):
                        pred = outputs[0]
                    else:
                        pred = outputs
                    
                    tf_predictions.append(pred.cpu().numpy())
            
            # Concatenate all predictions
            predictions[tf_name] = np.concatenate(tf_predictions, axis=0)
        
        return predictions
    
    def full_multiscale_pipeline(self, train_data: Dict[str, pd.DataFrame], 
                                test_data: Dict[str, pd.DataFrame],
                                models: Dict[str, torch.nn.Module]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Complete multi-scale prediction pipeline
        """
        results = {}
        
        # Create aligned datasets
        train_aligned = self.create_timeframe_dataset(train_data)
        test_aligned = self.create_timeframe_dataset(test_data)
        
        print(f"Created aligned datasets for timeframes: {list(train_aligned.keys())}")
        
        # Train models for each timeframe
        for tf_name, model in models.items():
            print(f"Training model for {tf_name}...")
            
            # Training would happen here
            # For now, we'll simulate predictions
            
            # Generate predictions on test data
            test_features = test_aligned[tf_name]
            
            # Simulate predictions (replace with actual model inference)
            predictions = np.random.randn(test_features.shape[0], 24)  # 24-step forecast
            
            results[tf_name] = {
                'predictions': predictions,
                'features': test_features
            }
        
        # Reconcile predictions across hierarchy
        pred_dict = {tf_name: results[tf_name]['predictions'] for tf_name in results}
        
        # Convert string keys to TimeframeType (simplified mapping)
        tf_mapping = {
            '1min': TimeframeType.MINUTE,
            '5min': TimeframeType.FIVE_MINUTE,
            '1h': TimeframeType.HOURLY,
            '1D': TimeframeType.DAILY
        }
        
        reconcile_input = {}
        for tf_name, pred in pred_dict.items():
            if tf_name in tf_mapping:
                reconcile_input[tf_mapping[tf_name]] = pred
        
        if reconcile_input:
            reconciled = self.reconciliation.reconcile_predictions(reconcile_input)
            
            # Convert back to string keys
            for tf_enum, pred in reconciled.items():
                tf_str = tf_enum.value
                if tf_str in results:
                    results[tf_str]['reconciled_predictions'] = pred
        
        return results


def create_multi_timeframe_config() -> Dict[str, TimeframeConfig]:
    """
    Create default configuration for multi-timeframe system
    """
    configs = {
        '1min': TimeframeConfig(
            timeframe=TimeframeType.MINUTE,
            weight=0.1,
            seq_len=1440,  # 24 hours of minutes
            pred_len=60,   # 1 hour ahead
            features=['open', 'high', 'low', 'close', 'volume'],
            importance=0.2
        ),
        '5min': TimeframeConfig(
            timeframe=TimeframeType.FIVE_MINUTE,
            weight=0.2,
            seq_len=288,   # 24 hours of 5-minute bars
            pred_len=12,   # 1 hour ahead
            features=['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd'],
            importance=0.3
        ),
        '1h': TimeframeConfig(
            timeframe=TimeframeType.HOURLY,
            weight=0.4,
            seq_len=168,   # 1 week of hours
            pred_len=24,   # 1 day ahead
            features=['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'bb_width'],
            importance=0.6
        ),
        '1D': TimeframeConfig(
            timeframe=TimeframeType.DAILY,
            weight=0.3,
            seq_len=90,    # 3 months of days
            pred_len=7,    # 1 week ahead
            features=['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'bb_width', 'realized_vol'],
            importance=1.0
        )
    }
    
    return configs