import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import OrderedDict

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoConfig, AutoModel, AutoTokenizer
from layers.Embed import PatchEmbedding
from layers.StandardNorm import Normalize
import transformers

transformers.logging.set_verbosity_error()


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism that focuses on relevant historical periods
    """
    
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
        # Learnable positional encoding for temporal patterns
        self.temporal_encoding = nn.Parameter(torch.randn(1000, d_model))
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Add temporal encoding
        positions = torch.arange(seq_len, device=x.device)
        temporal_enc = self.temporal_encoding[positions]
        x = x + temporal_enc
        
        # Multi-head attention
        Q = self.query(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
            
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        output = self.out(attention_output)
        return output, attention_weights


class MultiTimeframeEncoder(nn.Module):
    """
    Encodes multiple timeframes (minute, hourly, daily) into unified representation
    """
    
    def __init__(self, d_model, timeframes=['1min', '1h', '1D']):
        super().__init__()
        self.timeframes = timeframes
        self.d_model = d_model
        
        # Separate encoders for each timeframe
        self.timeframe_encoders = nn.ModuleDict({
            tf: nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model)
            ) for tf in timeframes
        })
        
        # Timeframe fusion layer
        self.fusion = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Learnable timeframe embeddings
        self.timeframe_embeddings = nn.Embedding(len(timeframes), d_model)
        
    def forward(self, timeframe_data: Dict[str, torch.Tensor]):
        """
        Args:
            timeframe_data: Dict mapping timeframe names to tensors
        Returns:
            Fused multi-timeframe representation
        """
        encoded_timeframes = []
        timeframe_ids = []
        
        for i, (tf, data) in enumerate(timeframe_data.items()):
            if tf in self.timeframe_encoders:
                # Encode this timeframe
                encoded = self.timeframe_encoders[tf](data)
                
                # Add timeframe embedding
                tf_embedding = self.timeframe_embeddings(torch.tensor(i, device=data.device))
                encoded = encoded + tf_embedding.unsqueeze(0).unsqueeze(0)
                
                encoded_timeframes.append(encoded)
                timeframe_ids.append(i)
        
        if not encoded_timeframes:
            raise ValueError("No valid timeframes provided")
        
        # Stack timeframes for attention
        stacked = torch.stack(encoded_timeframes, dim=1)  # [batch, n_timeframes, seq_len, d_model]
        batch_size, n_tf, seq_len, d_model = stacked.shape
        
        # Reshape for attention: [batch * seq_len, n_timeframes, d_model]
        reshaped = stacked.transpose(1, 2).contiguous().view(batch_size * seq_len, n_tf, d_model)
        
        # Apply cross-timeframe attention
        fused, attention_weights = self.fusion(reshaped, reshaped, reshaped)
        
        # Reshape back: [batch, seq_len, d_model]
        fused = fused.mean(dim=1).view(batch_size, seq_len, d_model)
        
        return fused, attention_weights


class HierarchicalPredictor(nn.Module):
    """
    Hierarchical prediction system that combines short-term and long-term forecasts
    """
    
    def __init__(self, d_model, pred_horizons=[6, 24, 96], dropout=0.1):
        super().__init__()
        self.pred_horizons = pred_horizons
        self.d_model = d_model
        
        # Separate prediction heads for different horizons
        self.horizon_predictors = nn.ModuleDict({
            f'horizon_{h}': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, h)
            ) for h in pred_horizons
        })
        
        # Consistency regularization between horizons
        self.consistency_weight = nn.Parameter(torch.tensor(0.1))
        
        # Adaptive weighting based on prediction confidence
        self.confidence_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, len(pred_horizons)),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model] encoded representation
        Returns:
            Dict of predictions for each horizon and confidence weights
        """
        # Use last timestep for prediction
        last_hidden = x[:, -1, :]  # [batch, d_model]
        
        predictions = {}
        horizon_outputs = []
        
        # Generate predictions for each horizon
        for horizon in self.pred_horizons:
            pred_key = f'horizon_{horizon}'
            pred = self.horizon_predictors[pred_key](last_hidden)
            predictions[pred_key] = pred
            horizon_outputs.append(pred)
        
        # Estimate confidence for each horizon
        confidence_weights = self.confidence_estimator(last_hidden)
        
        # Compute consistency loss between overlapping horizons
        consistency_loss = 0.0
        if len(horizon_outputs) > 1:
            for i in range(len(horizon_outputs) - 1):
                pred_short = horizon_outputs[i]
                pred_long = horizon_outputs[i + 1]
                
                # Compare overlapping portions
                overlap_len = min(pred_short.shape[1], pred_long.shape[1])
                if overlap_len > 0:
                    consistency_loss += F.mse_loss(
                        pred_short[:, :overlap_len],
                        pred_long[:, :overlap_len]
                    )
        
        return {
            'predictions': predictions,
            'confidence_weights': confidence_weights,
            'consistency_loss': consistency_loss * self.consistency_weight
        }


class EnsembleLLMPredictor(nn.Module):
    """
    Ensemble of multiple LLM predictors with dynamic weighting
    """
    
    def __init__(self, llm_configs, d_model, pred_len):
        super().__init__()
        self.llm_configs = llm_configs
        self.d_model = d_model
        self.pred_len = pred_len
        
        # Individual LLM predictors
        self.llm_predictors = nn.ModuleDict()
        self.llm_weights = nn.ParameterDict()
        
        for name, config in llm_configs.items():
            # Create lightweight prediction head for each LLM
            self.llm_predictors[name] = nn.Sequential(
                nn.Linear(config['llm_dim'], d_model),
                nn.ReLU(),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, pred_len)
            )
            
            # Learnable weight for this LLM
            self.llm_weights[name] = nn.Parameter(torch.tensor(1.0))
        
        # Dynamic weighting network
        self.weight_network = nn.Sequential(
            nn.Linear(d_model * len(llm_configs), d_model),
            nn.ReLU(),
            nn.Linear(d_model, len(llm_configs)),
            nn.Softmax(dim=-1)
        )
        
        # Performance tracking for adaptive weighting
        self.register_buffer('performance_history', torch.ones(len(llm_configs)))
        self.register_buffer('update_count', torch.zeros(1))
        
    def forward(self, llm_outputs: Dict[str, torch.Tensor], target=None):
        """
        Args:
            llm_outputs: Dict mapping LLM names to their outputs
            target: Optional target for performance tracking
        Returns:
            Ensemble prediction and individual predictions
        """
        individual_predictions = {}
        prediction_features = []
        
        # Get predictions from each LLM
        for name, output in llm_outputs.items():
            if name in self.llm_predictors:
                pred = self.llm_predictors[name](output)
                individual_predictions[name] = pred
                prediction_features.append(pred)
        
        if not prediction_features:
            raise ValueError("No valid LLM outputs provided")
        
        # Concatenate for dynamic weighting
        concat_features = torch.cat(prediction_features, dim=-1)
        dynamic_weights = self.weight_network(concat_features)
        
        # Combine static and dynamic weights
        static_weights = torch.stack([self.llm_weights[name] for name in individual_predictions.keys()])
        static_weights = F.softmax(static_weights, dim=0)
        
        # Performance-based weighting
        perf_weights = F.softmax(self.performance_history[:len(individual_predictions)], dim=0)
        
        # Final ensemble weights (average of dynamic, static, and performance)
        final_weights = (dynamic_weights.mean(dim=0) + static_weights + perf_weights) / 3.0
        
        # Weighted ensemble prediction
        ensemble_pred = sum(
            weight * pred for weight, pred in zip(final_weights, prediction_features)
        )
        
        # Update performance tracking if target provided
        if target is not None and self.training:
            self._update_performance(individual_predictions, target)
        
        return {
            'ensemble_prediction': ensemble_pred,
            'individual_predictions': individual_predictions,
            'ensemble_weights': final_weights,
            'dynamic_weights': dynamic_weights
        }
    
    def _update_performance(self, predictions: Dict[str, torch.Tensor], target: torch.Tensor):
        """Update performance history for adaptive weighting"""
        with torch.no_grad():
            for i, (name, pred) in enumerate(predictions.items()):
                # Simple MSE-based performance metric
                mse = F.mse_loss(pred, target, reduction='mean')
                
                # Exponential moving average of performance (lower is better)
                alpha = 0.1
                self.performance_history[i] = (
                    alpha * (1.0 / (1.0 + mse)) + 
                    (1 - alpha) * self.performance_history[i]
                )
            
            self.update_count += 1


class MultiScaleTimeLLM(nn.Module):
    """
    Multi-scale Time-LLM with hierarchical forecasting and ensemble methods
    """
    
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        
        # Multi-scale components
        self.temporal_attention = TemporalAttention(
            d_model=configs.d_model,
            n_heads=8,
            dropout=configs.dropout
        )
        
        self.multi_timeframe_encoder = MultiTimeframeEncoder(
            d_model=configs.d_model,
            timeframes=getattr(configs, 'timeframes', ['1min', '1h', '1D'])
        )
        
        self.hierarchical_predictor = HierarchicalPredictor(
            d_model=configs.d_model,
            pred_horizons=getattr(configs, 'pred_horizons', [6, 24, 96]),
            dropout=configs.dropout
        )
        
        # LLM ensemble configuration
        llm_configs = {
            'llama': {'llm_dim': 4096},
            'gpt2': {'llm_dim': 768},
            'bert': {'llm_dim': 768}
        }
        
        self.ensemble_predictor = EnsembleLLMPredictor(
            llm_configs=llm_configs,
            d_model=configs.d_model,
            pred_len=self.pred_len
        )
        
        # Core components (from original TimeLLM)
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout
        )
        
        self.normalize_layers = Normalize(configs.enc_in, affine=False)
        
        # Multi-resolution patch embeddings
        self.multi_patch_lengths = getattr(configs, 'multi_patch_lengths', [8, 16, 32])
        self.multi_patch_embeddings = nn.ModuleDict({
            f'patch_{pl}': PatchEmbedding(configs.d_model, pl, pl//2, configs.dropout)
            for pl in self.multi_patch_lengths
        })
        
        # Adaptive patch selection
        self.patch_selector = nn.Sequential(
            nn.Linear(configs.d_model, len(self.multi_patch_lengths)),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return None
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        x_enc = self.normalize_layers(x_enc, 'norm')
        
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        
        # Multi-resolution patch embedding
        multi_patch_outputs = []
        patch_weights_list = []
        
        for patch_len in self.multi_patch_lengths:
            if T >= patch_len:
                patch_emb = self.multi_patch_embeddings[f'patch_{patch_len}']
                patch_out, n_vars = patch_emb(x_enc.permute(0, 2, 1))
                
                # Apply temporal attention
                attended_out, attention_weights = self.temporal_attention(patch_out)
                
                multi_patch_outputs.append(attended_out)
                
                # Compute patch selection weights
                patch_weights = self.patch_selector(attended_out.mean(dim=1))
                patch_weights_list.append(patch_weights)
        
        # Adaptive patch fusion
        if multi_patch_outputs:
            # Average patch weights across samples
            avg_patch_weights = torch.stack(patch_weights_list).mean(dim=0)
            
            # Weighted combination of multi-resolution outputs
            fused_output = sum(
                weight.unsqueeze(1).unsqueeze(2) * output 
                for weight, output in zip(avg_patch_weights.T, multi_patch_outputs)
            )
        else:
            # Fallback to single patch embedding
            fused_output, n_vars = self.patch_embedding(x_enc.permute(0, 2, 1))
            fused_output, _ = self.temporal_attention(fused_output)
        
        # Hierarchical prediction
        hierarchical_results = self.hierarchical_predictor(fused_output)
        
        # Select appropriate prediction horizon
        target_horizon = f'horizon_{self.pred_len}' if f'horizon_{self.pred_len}' in hierarchical_results['predictions'] else list(hierarchical_results['predictions'].keys())[0]
        
        dec_out = hierarchical_results['predictions'][target_horizon]
        
        # Reshape and denormalize
        dec_out = dec_out.reshape(B, N, -1).permute(0, 2, 1).contiguous()
        
        # Ensure correct prediction length
        if dec_out.shape[1] != self.pred_len:
            if dec_out.shape[1] > self.pred_len:
                dec_out = dec_out[:, :self.pred_len, :]
            else:
                # Pad if necessary
                padding = torch.zeros(B, self.pred_len - dec_out.shape[1], N, device=dec_out.device)
                dec_out = torch.cat([dec_out, padding], dim=1)
        
        dec_out = self.normalize_layers(dec_out, 'denorm')
        
        return dec_out, hierarchical_results['consistency_loss']
    
    def get_attention_maps(self, x_enc, x_mark_enc):
        """Extract attention maps for interpretability"""
        x_enc = self.normalize_layers(x_enc, 'norm')
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        
        # Get patch embedding
        patch_out, _ = self.patch_embedding(x_enc.permute(0, 2, 1))
        
        # Get attention maps
        _, attention_weights = self.temporal_attention(patch_out)
        
        return attention_weights
    
    def predict_with_uncertainty(self, x_enc, x_mark_enc, x_dec, x_mark_dec, n_samples=10):
        """Generate predictions with uncertainty estimates"""
        self.train()  # Enable dropout for uncertainty
        
        predictions = []
        for _ in range(n_samples):
            pred, _ = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # Compute statistics
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Confidence intervals
        confidence_intervals = {
            'lower_95': mean_pred - 1.96 * std_pred,
            'upper_95': mean_pred + 1.96 * std_pred,
            'lower_68': mean_pred - std_pred,
            'upper_68': mean_pred + std_pred
        }
        
        self.eval()  # Return to eval mode
        
        return {
            'mean_prediction': mean_pred,
            'std_prediction': std_pred,
            'confidence_intervals': confidence_intervals,
            'all_samples': predictions
        }