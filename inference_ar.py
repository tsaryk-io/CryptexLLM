import argparse
import torch
import pandas as pd
import numpy as np
from models import TimeLLM, Autoformer, DLinear
from data_provider.data_factory import data_provider
from sklearn.preprocessing import StandardScaler
from utils.tools import load_content
import os

def load_model_for_inference(model_path, args, device='cuda'):
    """Load a saved model for inference"""
    # Load the saved state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Initialize the appropriate model
    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()
    
    # Load the state dict
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model

def prepare_data(data, args, scaler=None, timestamps=None):
    """Prepare input data for inference"""
    # Convert data to numpy array if it's a DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    # Scale the data if scaler is provided
    if scaler is not None:
        data = scaler.transform(data)
    
    # Convert to tensor
    data = torch.FloatTensor(data).unsqueeze(0)  # Add batch dimension
    
    # Create time features using provided timestamps (as pandas Timestamps)
    time_features = []
    for t in timestamps:
        time_feat = [
            t.month,
            t.day,
            t.weekday(),
            t.hour,
        ]
        time_features.append(time_feat)
    time_features = torch.FloatTensor(time_features).unsqueeze(0)
    
    return data, time_features

def main():
    parser = argparse.ArgumentParser(description='TimeLLM Inference')
    
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output CSV file')

    # basic config
    parser.add_argument('--model_path', type=str, required=True, help='path to saved model')
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, DLinear]')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; '
                            'M:multivariate predict multivariate, S: univariate predict univariate, '
                            'MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--loader', type=str, default='modal', help='dataset type')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, '
                            'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                            'you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=0, help='')
    parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
    parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768


    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--percent', type=int, default=100)
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.content = load_content(args)

    
    # Load data
    df = pd.read_csv(args.data_path)
    
    # Convert UNIX timestamp to pandas Timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Prepare scaler
    scaler = StandardScaler()
    scaler.fit(df.iloc[:, 1:].values)  # Fit on all data except timestamp
    
    # Load model
    model = load_model_for_inference(args.model_path, args, device)
    model = model.to(torch.bfloat16)

    # Initialize results DataFrame
    results_df = df.copy()
    results_df['close_predicted'] = np.nan

    # Start from seq_len'th datapoint
    for i in range(args.seq_len, len(df) - args.pred_len + 1):
        # Get sequence for prediction
        seq_data = df.iloc[i-args.seq_len:i].copy()
        seq_timestamps = seq_data['timestamp'].tolist()

        # Prepare data for inference
        data_x, data_x_mark = prepare_data(
            seq_data.iloc[:, 1:], args, scaler, timestamps=seq_timestamps
        )

        # Generate future timestamps for the prediction window
        last_timestamp = seq_data['timestamp'].iloc[-1]
        freq_offset = pd.tseries.frequencies.to_offset(args.freq)
        future_timestamps = [last_timestamp + freq_offset * (j + 1) for j in range(args.pred_len)]

        # Prepare decoder time features for all future timestamps
        dec_mark = []
        for t in future_timestamps:
            time_feat = [
                t.month,
                t.day,
                t.weekday(),
                t.hour,
            ]
            dec_mark.append(time_feat)
        dec_mark = torch.FloatTensor(dec_mark).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            data_x = data_x.to(device)
            data_x_mark = data_x_mark.to(device)
            dec_inp = torch.zeros((1, args.pred_len, data_x.shape[-1])).float().to(device)
            dec_inp = torch.cat([data_x[:, -args.label_len:, :], dec_inp], dim=1)
            outputs = model(data_x, data_x_mark, dec_inp, dec_mark)
            predictions = outputs[:, -args.pred_len:, :]

        # Inverse transform predictions
        predictions = predictions.to(torch.float32).cpu().numpy().squeeze()
        predictions = scaler.inverse_transform(predictions)

        # Store predictions for the next 96 timestamps
        for j in range(args.pred_len):
            if i + j < len(results_df):  # Ensure we don't exceed DataFrame length
                results_df.loc[i + j, 'close_predicted'] = predictions[j, 1]  # Index 1 for close price

        # Print progress
        if i % 100 == 0:
            print(f"Processed {i}/{len(df) - args.pred_len + 1} datapoints")

    # Save results
    results_df.to_csv(args.output_path, index=False)
    print(f"\nResults saved to {args.output_path}")

    return results_df

if __name__ == "__main__":
    main()
