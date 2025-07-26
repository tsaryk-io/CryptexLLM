#!/usr/bin/env python3

import argparse
import torch
import pandas as pd
import numpy as np
import os
import re
import sys
from types import SimpleNamespace

from models import TimeLLM, Autoformer, DLinear
from data_provider.data_factory import data_provider
from sklearn.preprocessing import StandardScaler
from utils.tools import load_content

def parse_model_id(model_id):
    """
    Parses the model_id string to extract hyperparameters.
    Handles both adaptive and regular model naming patterns.
    Expected formats:
    - {prefix}_{llm_model}_L{layers}_{granularity}_{features}_seq{seq_len}_pred{pred_len}_p{patch_len}_s{stride}_v{num_tokens}
    """
    # Updated pattern to handle prefixes like qwen_regular_, llama31_adaptive_, etc.
    pattern = re.compile(r'(.+?)_([A-Z0-9.]+)_L(\d+)_(.+?)_([A-Z]+)_seq(\d+)_pred(\d+)_p(\d+)_s(\d+)_v(\d+)')
    match = pattern.match(model_id)
    
    if match:
        groups = match.groups()
        prefix = groups[0]  # e.g., "qwen_regular", "llama31_adaptive" 
        llm_model = groups[1]  # e.g., "QWEN", "LLAMA3.1"
        
        # Determine if it's adaptive or regular from prefix
        if 'adaptive' in prefix:
            adaptive_type = 'adaptive'
        elif 'regular' in prefix:
            adaptive_type = 'regular'
        else:
            adaptive_type = None
            
        return {
            'adaptive_type': adaptive_type,
            'llm_model': llm_model,
            'llm_layers': int(groups[2]),
            'granularity': groups[3],
            'features': groups[4],
            'seq_len': int(groups[5]),
            'pred_len': int(groups[6]),
            'patch_len': int(groups[7]),
            'stride': int(groups[8]),
            'num_tokens': int(groups[9]),
        }
    
    # Fallback: try without prefix (original format)
    fallback_pattern = re.compile(r'([A-Z0-9.]+)_L(\d+)_(.+?)_([A-Z]+)_seq(\d+)_pred(\d+)_p(\d+)_s(\d+)_v(\d+)')
    match = fallback_pattern.match(model_id)
    
    if match:
        groups = match.groups()
        return {
            'adaptive_type': None,
            'llm_model': groups[0],
            'llm_layers': int(groups[1]),
            'granularity': groups[2],
            'features': groups[3],
            'seq_len': int(groups[4]),
            'pred_len': int(groups[5]),
            'patch_len': int(groups[6]),
            'stride': int(groups[7]),
            'num_tokens': int(groups[8]),
        }
    
    print(f"Error: model_id '{model_id}' does not match expected format.")
    sys.exit(1)

def get_model_specific_config(llm_model):
    """Get model-specific configuration parameters"""
    model_configs = {
        'QWEN': {
            'llm_dim': 4096,
            'mixed_precision': False,
            'dtype': torch.float32
        },
        'GEMMA': {
            'llm_dim': 1152,
            'mixed_precision': True,
            'dtype': torch.float16
        },
        'LLAMA': {
            'llm_dim': 4096,
            'mixed_precision': True,
            'dtype': torch.float16
        },
        'LLAMA3.1': {
            'llm_dim': 4096,
            'mixed_precision': True,
            'dtype': torch.float16
        },
        'MISTRAL': {
            'llm_dim': 4096,
            'mixed_precision': True,
            'dtype': torch.float16
        },
        'DEEPSEEK': {
            'llm_dim': 4096,
            'mixed_precision': True,
            'dtype': torch.float16
        },
        'GPT2': {
            'llm_dim': 768,
            'mixed_precision': True,
            'dtype': torch.float16
        }
    }
    
    return model_configs.get(llm_model, {
        'llm_dim': 4096,
        'mixed_precision': True,
        'dtype': torch.float16
    })

def get_data_path_and_freq(granularity, use_enhanced=True):
    """Gets the enhanced data file path and frequency string based on granularity."""
    if use_enhanced:
        granularity_map = {
            'hourly': ('cryptex/candlesticks-h-enhanced.csv', 'h'),
            'minute': ('cryptex/candlesticks-Min-enhanced.csv', 't'),
            'daily': ('cryptex/candlesticks-D-enhanced.csv', 'd'),
        }
    else:
        granularity_map = {
            'hourly': ('cryptex/candlesticks-h.csv', 'h'),
            'minute': ('cryptex/candlesticks-Min.csv', 't'),
            'daily': ('cryptex/candlesticks-D.csv', 'd'),
        }
    
    if granularity not in granularity_map:
        print(f"Error: Invalid granularity '{granularity}' found in model_id.")
        sys.exit(1)
    return granularity_map[granularity]

def load_model_for_inference(model_path, args, device='cuda'):
    """Load a saved model for inference with model-specific configurations"""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        sys.exit(1)

    # Get model-specific config
    model_config = get_model_specific_config(args.llm_model)
    
    # Override llm_dim if specified in CLI
    if hasattr(args, 'llm_dim_override') and args.llm_dim_override:
        args.llm_dim = args.llm_dim_override
    else:
        args.llm_dim = model_config['llm_dim']
    
    print(f"Loading {args.llm_model} model with llm_dim={args.llm_dim}")
    
    state_dict = torch.load(model_path, map_location=device)
    
    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    # Apply model-specific dtype
    if model_config['dtype'] == torch.float16 and model_config['mixed_precision']:
        model = model.to(torch.float16)
    elif model_config['dtype'] == torch.float32:
        model = model.to(torch.float32)
    
    model.eval()
    return model

def prepare_enhanced_data(data, args, scaler=None, timestamps=None):
    """Prepare enhanced dataset (26 features) for inference"""
    # Convert data to numpy array if it's a DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    if scaler is not None:
        data = scaler.transform(data)
    
    data = torch.FloatTensor(data).unsqueeze(0)  # Add batch dimension
    
    # Create time features
    time_features = []
    for t in pd.to_datetime(timestamps):
        time_feat = [t.month, t.day, t.weekday(), t.hour]
        time_features.append(time_feat)
    time_features = torch.FloatTensor(time_features).unsqueeze(0)
    
    return data, time_features

def run_enhanced_inference(args):
    """Main inference logic for enhanced dataset models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    args.content = load_content(args)
    
    # Load enhanced dataset
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at '{args.data_path}'")
        sys.exit(1)
    
    print(f"Loading enhanced dataset from: {args.data_path}")
    df = pd.read_csv(args.data_path)
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # For enhanced dataset, exclude timestamp and datetime columns
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'datetime']]
    print(f"Using {len(feature_cols)} features: {feature_cols[:5]}..." if len(feature_cols) > 5 else f"Using {len(feature_cols)} features: {feature_cols}")
    
    # Prepare scaler for enhanced features
    scaler = StandardScaler()
    scaler.fit(df[feature_cols])
    
    # Load model with enhanced configuration
    model = load_model_for_inference(args.model_path, args, device)
    
    # Initialize results DataFrame
    results_df = df.copy()
    for i in range(1, args.pred_len + 1):
        results_df[f'close_predicted_{i}'] = np.nan
    
    print("Starting enhanced inference loop...")
    print(f"Processing {len(df) - args.seq_len} predictions...")
    
    # Start from seq_len'th datapoint
    for i in range(args.seq_len, len(df)):
        # Get sequence for prediction
        seq_data_df = df.iloc[i-args.seq_len:i].copy()
        seq_data_features = seq_data_df[feature_cols]
        seq_timestamps = seq_data_df['timestamp'].tolist()
        
        data_x, data_x_mark = prepare_enhanced_data(
            seq_data_features, args, scaler, timestamps=seq_timestamps
        )
        
        # Generate future timestamps
        last_timestamp = seq_timestamps[-1]
        freq_offset = pd.tseries.frequencies.to_offset(args.freq)
        future_timestamps = [last_timestamp + freq_offset * (j + 1) for j in range(args.pred_len)]
        
        # Prepare decoder time features
        dec_mark_list = []
        for t in future_timestamps:
            time_feat = [t.month, t.day, t.weekday(), t.hour]
            dec_mark_list.append(time_feat)
        dec_mark = torch.FloatTensor(dec_mark_list).unsqueeze(0).to(device)
        
        # Make prediction with model-specific dtype
        with torch.no_grad():
            model_config = get_model_specific_config(args.llm_model)
            dtype = model_config['dtype']
            
            data_x = data_x.to(device, dtype=dtype)
            data_x_mark = data_x_mark.to(device, dtype=dtype)
            dec_inp = torch.zeros((1, args.pred_len, data_x.shape[-1]), dtype=dtype).to(device)
            
            # Use label_len for decoder input
            dec_inp = torch.cat([data_x[:, -args.label_len:, :], dec_inp], dim=1)
            
            outputs = model(data_x, data_x_mark, dec_inp, dec_mark)
            predictions = outputs[:, -args.pred_len:, :]
        
        # Convert predictions back to float32 for processing
        predictions_np = predictions.to(torch.float32).cpu().numpy().squeeze()
        
        # Reshape if needed
        if args.pred_len == 1:
            predictions_np = predictions_np.reshape(1, -1)
        
        # Inverse transform - handle enhanced feature dimensions
        if predictions_np.shape[1] < len(feature_cols):
            padded_preds = np.zeros((predictions_np.shape[0], len(feature_cols)))
            padded_preds[:, :predictions_np.shape[1]] = predictions_np
            predictions_inv = scaler.inverse_transform(padded_preds)
        else:
            predictions_inv = scaler.inverse_transform(predictions_np)
        
        # Store close price predictions (assuming 'close' is in feature_cols)
        try:
            close_col_index = feature_cols.index('close')
        except ValueError:
            # Fallback: assume close is the 4th column (after open, high, low)
            close_col_index = min(3, len(feature_cols) - 1)
        
        for j in range(args.pred_len):
            results_df.loc[i, f'close_predicted_{j+1}'] = predictions_inv[j, close_col_index]
        
        if (i + 1) % 100 == 0:
            print(f"\rProcessed {i+1-args.seq_len}/{len(df)-args.seq_len} predictions", end="", flush=True)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Save results with timestamp conversion for trader-pred.py compatibility
    results_df['timestamp'] = results_df['timestamp'].astype(int) // 10**9  # Convert to UNIX timestamp
    results_df.to_csv(args.output_path, index=False)
    
    print(f"\nEnhanced inference complete. Results saved to {args.output_path}")
    print(f"Columns in output: {list(results_df.columns)}")
    print(f"Sample predictions:")
    pred_cols = [col for col in results_df.columns if 'predicted' in col]
    print(results_df[['timestamp', 'close'] + pred_cols].tail())

def main():
    parser = argparse.ArgumentParser(description='Enhanced TimeLLM Inference for 26-feature models')
    parser.add_argument('--model_id', type=str, required=True, 
                       help='Model ID (e.g., qwen_regular_QWEN_L6_daily_MS_seq32_pred7_p1_s1_v1000)')
    parser.add_argument('--data_path', type=str, default=None, 
                       help='Path to enhanced CSV dataset for inference')
    parser.add_argument('--llm_dim', type=int, default=None, 
                       help='Override LLM dimension (auto-detected if not specified)')
    parser.add_argument('--output_dir', type=str, default='/mnt/nfs/inference_enhanced', 
                       help='Output directory for results')
    
    cli_args = parser.parse_args()
    
    # Static configuration for enhanced models
    config = {
        'models_dir': '/mnt/nfs/models',
        'root_path': './dataset/',
        'model': 'TimeLLM',
        'target': 'close',
        'd_model': 32,
        'd_ff': 128,
        'enc_in': 26,  # Enhanced dataset has 26 features
        'dec_in': 26,
        'c_out': 1,    # Predicting single target (close)
        'factor': 3,
        'n_heads': 8,
        'd_layers': 1,
        'dropout': 0.1,
        'moving_avg': 25,
        'embed': 'timeF',
        'activation': 'gelu',
        'output_attention': False,
        'prompt_domain': 0,
    }
    
    # Parse model_id
    try:
        parsed_params = parse_model_id(cli_args.model_id)
    except Exception as e:
        print(f"Error parsing model_id: {e}")
        sys.exit(1)
    
    # Determine data path
    if cli_args.data_path:
        data_path = cli_args.data_path
        _, freq = get_data_path_and_freq(parsed_params['granularity'])
    else:
        data_file, freq = get_data_path_and_freq(parsed_params['granularity'], use_enhanced=True)
        data_path = os.path.join(config['root_path'], data_file)
    
    # Build final args namespace
    args = SimpleNamespace(
        # From CLI
        model_id=cli_args.model_id,
        llm_dim_override=cli_args.llm_dim,
        
        # From static config
        **config,
        
        # From parsed model_id
        **parsed_params,
        
        # Derived values
        data='CRYPTEX_ENHANCED',
        data_path=data_path,
        model_path=os.path.join(config['models_dir'], f"{cli_args.model_id}.pth"),
        output_path=os.path.join(cli_args.output_dir, f"enhanced_{cli_args.model_id}.csv"),
        freq=freq,
        label_len=parsed_params['seq_len'] // 2,
        task_name='long_term_forecast',
    )
    
    # Print configuration
    print("=== Enhanced Inference Configuration ===")
    print(f"Model ID: {args.model_id}")
    print(f"LLM Model: {args.llm_model}")
    print(f"Model Path: {args.model_path}")
    print(f"Data Path: {args.data_path}")
    print(f"Output Path: {args.output_path}")
    print(f"Enhanced Features: {args.enc_in}")
    print(f"Sequence Length: {args.seq_len}, Prediction Length: {args.pred_len}")
    
    # Show model-specific config
    model_config = get_model_specific_config(args.llm_model)
    print(f"Model Config: {model_config}")
    print("=" * 40)
    
    # Run enhanced inference
    run_enhanced_inference(args)

if __name__ == "__main__":
    main()