#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from data_provider.enhanced_data_loader_optimized import Dataset_CRYPTEX_Enhanced_Optimized
from data_provider.data_factory import data_dict

# Register optimized dataset
data_dict['CRYPTEX_ENHANCED_OPTIMIZED'] = Dataset_CRYPTEX_Enhanced_Optimized

# Run training with optimized configuration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # Basic config
    parser.add_argument('--model_id', type=str, default='TimeLLM_Cryptex_Optimized')
    parser.add_argument('--model', type=str, default='TimeLLM')
    
    # Data config
    parser.add_argument('--data', type=str, default='CRYPTEX_ENHANCED_OPTIMIZED')
    parser.add_argument('--root_path', type=str, default='./dataset')
    parser.add_argument('--data_path', type=str, default='cryptex/candlesticks-D.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='close')
    parser.add_argument('--freq', type=str, default='d')
    parser.add_argument('--embed', type=str, default='timeF')
    
    # Model config - Optimized for speed
    parser.add_argument('--seq_len', type=int, default=21)
    parser.add_argument('--label_len', type=int, default=7)
    parser.add_argument('--pred_len', type=int, default=7)
    parser.add_argument('--enc_in', type=int, default=26)  # 6 OHLCV + 20 technical
    parser.add_argument('--dec_in', type=int, default=26)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--d_ff', type=int, default=128)
    
    # LLM config - Fast model
    parser.add_argument('--llm_model', type=str, default='GPT2')
    parser.add_argument('--llm_dim', type=int, default=768)
    parser.add_argument('--llm_layers', type=int, default=6)
    
    # Training config
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--loss', type=str, default='DLF')
    
    # System config
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--itr', type=int, default=1)
    
    args = parser.parse_args()
    
    print("Starting optimized Time-LLM-Cryptex training...")
    print(f"Configuration:")
    print(f"  Features: {args.enc_in} (optimized from 68+)")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Model: {args.llm_model}")
    print(f"  Loss function: {args.loss}")
    
    # Import and run main training
    try:
        from run_main import main
        main(args)
    except ImportError:
        print("run_main.py not found. Please ensure you have the main training script.")
        print("To run manually:")
        print(f"python run_main.py --model_id {args.model_id} --data {args.data} --seq_len {args.seq_len} --batch_size {args.batch_size} --llm_model {args.llm_model}")
