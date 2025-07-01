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
    Assumes format: {llm_model}_L{llm_layers}_{granularity}_{features}_seq{seq_len}_pred{pred_len}_p{patch_len}_s{stride}_v{num_tokens}
    """
    pattern = re.compile(r'(.+?)_L(\d+)_(.+?)_([A-Z]+)_seq(\d+)_pred(\d+)_p(\d+)_s(\d+)_v(\d+)')
    match = pattern.match(model_id)
    
    if not match:
        print(f"Error: model_id '{model_id}' does not match the expected format.")
        sys.exit(1)
        
    groups = match.groups()
    
    params = {
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
    return params

def get_data_path_and_freq(granularity):
    """Gets the data file path and frequency string based on granularity."""
    granularity_map = {
        'hourly': ('candlesticks-h.csv', 'h'),
        'minute': ('candlesticks-Min.csv', 't'),
        'daily': ('candlesticks-D.csv', 'd'),
    }
    if granularity not in granularity_map:
        print(f"Error: Invalid granularity '{granularity}' found in model_id.")
        sys.exit(1)
    return granularity_map[granularity]

def load_model_for_inference(model_path, args, device='cuda'):
    """Load a saved model for inference"""
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        sys.exit(1)

    state_dict = torch.load(model_path, map_location=device)
    
    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model

def prepare_data(data, args, scaler=None, timestamps=None):
    """Prepare input data for inference"""
    # Convert data to numpy array if it's a DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    if scaler is not None:
        data = scaler.transform(data)
    
    data = torch.FloatTensor(data).unsqueeze(0) # Add batch dimension
    
    time_features = []
    # Use pandas Timestamps to extract time features
    for t in pd.to_datetime(timestamps):
        time_feat = [t.month, t.day, t.weekday(), t.hour]
        time_features.append(time_feat)
    time_features = torch.FloatTensor(time_features).unsqueeze(0)
    
    return data, time_features

def run_inference(args):
    """Main inference logic, takes a fully populated args object."""
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

    args.content = load_content(args)
    
    # Load data
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at '{args.data_path}'")
        sys.exit(1)
    df = pd.read_csv(args.data_path)
    # Convert UNIX timestamp to pandas Timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Prepare scaler (needed?)
    scaler = StandardScaler()
    # Fit scaler on all numeric columns except the timestamp (which is a pd.Timestamp object)
    numeric_cols = df.select_dtypes(include=np.number).columns
    scaler.fit(df[numeric_cols])
    
    # Load model
    model = load_model_for_inference(args.model_path, args, device)
    model = model.to(torch.bfloat16)

    # Initialize results DataFrame
    results_df = df.copy()
    for i in range(1, args.pred_len + 1):
        results_df[f'close_predicted_{i}'] = np.nan

    print("Starting inference loop...")
    # Start from seq_len'th datapoint
    for i in range(args.seq_len, len(df)):
        # Get sequence for prediction (using previous seq_len rows)
        seq_data_df = df.iloc[i-args.seq_len:i].copy()
        seq_data_numeric = seq_data_df[numeric_cols]
        seq_timestamps = seq_data_df['timestamp'].tolist()

        data_x, data_x_mark = prepare_data(
            seq_data_numeric, args, scaler, timestamps=seq_timestamps
        )

        # Generate future timestamps for the prediction window
        last_timestamp = seq_timestamps[-1]
        freq_offset = pd.tseries.frequencies.to_offset(args.freq)
        future_timestamps = [last_timestamp + freq_offset * (j + 1) for j in range(args.pred_len)]

        # Prepare decoder time features for all future timestamps
        dec_mark_list = []
        for t in future_timestamps:
            time_feat = [t.month, t.day, t.weekday(), t.hour]
            dec_mark_list.append(time_feat)
        dec_mark = torch.FloatTensor(dec_mark_list).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            data_x = data_x.to(device) # Add argument dtype=torch.bfloat16 if supported
            data_x_mark = data_x_mark.to(device) # dtype=torch.bfloat16
            dec_inp = torch.zeros((1, args.pred_len, data_x.shape[-1])).float().to(device) # dtype=torch.bfloat16
            # Use the label_len for the decoder input
            dec_inp = torch.cat([data_x[:, -args.label_len:, :], dec_inp], dim=1)
            outputs = model(data_x, data_x_mark, dec_inp, dec_mark)
            predictions = outputs[:, -args.pred_len:, :]

        predictions_np = predictions.to(torch.float32).cpu().numpy().squeeze()
        # Reshape predictions to 2D array if pred_len is 1
        if args.pred_len == 1:
            predictions_np = predictions_np.reshape(1, -1)
        
        # Inverse transform requires a 2D array with all features
        if predictions_np.shape[1] < len(numeric_cols):
             padded_preds = np.zeros((predictions_np.shape[0], len(numeric_cols)))
             padded_preds[:, :predictions_np.shape[1]] = predictions_np
             predictions_inv = scaler.inverse_transform(padded_preds)
        else:
             predictions_inv = scaler.inverse_transform(predictions_np)

        # Store predictions in the current row
        close_col_index = numeric_cols.get_loc('close')
        for j in range(args.pred_len):
            results_df.loc[i, f'close_predicted_{j+1}'] = predictions_inv[j, close_col_index]

        if (i + 1) % 100 == 0:
            print(f"\rProcessed {i+1}/{len(df)} datapoints", end="", flush=True)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    results_df.to_csv(args.output_path, index=False)
    print(f"\nInference complete. Results saved to {args.output_path}")

def main():
    parser = argparse.ArgumentParser(description='TimeLLM Inference Launcher')
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to run inference for. (e.g., LLAMA_L8_daily_MS_seq168_pred1_p6_s4_v1000)')
    parser.add_argument('--data_path', type=str, default=None, help='Path to a custom CSV dataset for inference.')
    parser.add_argument('--llm_dim', type=int, default=4096, help='LLM model dimension.')
    cli_args = parser.parse_args()

    # --- Static Configuration ---
    config = {
        'models_dir': '/mnt/nfs/models',
        'output_dir': '/mnt/nfs/inference_ar',
        'root_path': './dataset/cryptex/',
        'model': 'TimeLLM',
        'target': 'close',
        'd_model': 32,
        'd_ff': 128,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'factor': 3,
        'n_heads': 8,
        'd_layers': 1,
        'dropout': 0.1,
        'moving_avg': 25,
        'embed': 'timeF',
        'activation': 'gelu', # Does not care?
        'output_attention': False,
        'prompt_domain': False, # Does not care?
    }

    # --- Dynamic Configuration from model_id ---
    try:
        parsed_params = parse_model_id(cli_args.model_id)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


    # Determine data_path: use CLI arg if provided, otherwise derive it
    if cli_args.data_path:
        data_path = cli_args.data_path
        _, freq = get_data_path_and_freq(parsed_params['granularity']) # Still need freq from model
    else:
        data_file, freq = get_data_path_and_freq(parsed_params['granularity'])
        data_path = os.path.join(config['root_path'], data_file)


    # --- Build final args namespace ---
    args = SimpleNamespace(
        # From CLI
        model_id=cli_args.model_id,
        llm_dim=cli_args.llm_dim,
        
        # From static config
        **config,
        
        # From parsed model_id
        **parsed_params,
        
        # Derived values
        data='CRYPTEX',
        data_path=data_path,
        model_path=os.path.join(config['models_dir'], f"{cli_args.model_id}.pth"),
        output_path=os.path.join(config['output_dir'], f"iar_{cli_args.model_id}.csv"),
        freq=freq,
        label_len=parsed_params['seq_len'] // 2,
        task_name='short_term_forecast', # Hardcoded, temporary fix, check if it affects anything vs "long_term_forecast"
    )

    # Print configuration for verification
    print("--- Inference Configuration ---")
    print(f"Model ID: {args.model_id}")
    print(f"Model Path: {args.model_path}")
    print(f"Data Path: {args.data_path}")
    print(f"Output Path: {args.output_path}")
    print(f"LLM Dimension: {args.llm_dim}")
    print(f"Sequence Length: {args.seq_len}, Prediction Length: {args.pred_len}")
    print("-----------------------------\n")

    # Run the main inference logic
    run_inference(args)

if __name__ == "__main__":
    main()
