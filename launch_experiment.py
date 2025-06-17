import argparse
import subprocess
import sys
import os

from datetime import datetime

def generate_model_id(llm_model, llm_layers, granularity, features, seq_len, pred_len, patch_len, stride, num_tokens):
    """Generate comprehensive model ID including all key parameters"""
    model_id = f"{llm_model}_L{llm_layers}_{granularity}_{features}_seq{seq_len}_pred{pred_len}_p{patch_len}_s{stride}_v{num_tokens}"
    return model_id


def get_data_path(granularity):
    """Get data path based on granularity"""
    granularity_to_file = {
        'hourly': 'candlesticks-h.csv',
        'minute': 'candlesticks-Min.csv',
        'daily': 'candlesticks-D.csv'
    }
    return granularity_to_file[granularity]

# Prevent overriding
def check_model_exists(models_dir, model_id):
    """Check if model already exists in the models directory"""
    if not os.path.exists(models_dir):
        return False
    
    model_path = os.path.join(models_dir, f"{model_id}.pth")
    return os.path.exists(model_path)

def ask_override_confirmation(model_id):
    """Ask user if they want to override existing model"""
    print(f"Warning: Model '{model_id}' already exists in the models directory.")
    response = input("Do you want to override it? (y/N): ")
    return response.lower() in ['y', 'yes']

# Logging
def create_log_file(model_id, logs_dir):
    """Create log file path and ensure directory exists"""
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{model_id}_{timestamp}.log"
    log_path = os.path.join(logs_dir, log_filename)
    
    return log_path

def launch_experiment(args):
    """Launch the TimeLLM experiment with specified parameters"""
    
    # Generate dynamic parameters
    model_id = generate_model_id(args.llm_model, args.llm_layers, args.granularity, args.features, args.seq_len, args.pred_len, args.patch_len, args.stride, args.num_tokens)
    data_path = get_data_path(args.granularity)
    
    
    # Static configuration
    static_config = {
        'master_port': '29500',
        'num_process': '4',
        'd_model': '32',
        'd_ff': '128',
        'comment': 'TimeLLM-Cryptex',
        'train_epochs': '10',
        'learning_rate': '0.01',
        'enc_in': '7',
        'dec_in': '7',
        'c_out': '7',
        'factor': '3',
        'itr': '1',
        'data': 'CRYPTEX',
        'root_path': './dataset/cryptex/',
        'target': 'close',
        'batch_size': '24',
        'model': 'TimeLLM',
        'models_dir':'/mnt/nfs/models',
        'logs_dir':'/mnt/nfs/logs',
        'results_csv':'/mnt/nfs/experiment_results.csv'
    }
    
    # Build the command
    cmd = [
        'accelerate', 'launch',
        '--multi_gpu',
        '--mixed_precision', 'bf16',
        '--num_processes', static_config['num_process'],
        '--main_process_port', static_config['master_port'],
        'run_main.py',
        '--task_name', args.task_name,
        '--is_training', '1',
        '--model_id', model_id,
        '--model_comment', static_config['comment'],
        '--llm_model', args.llm_model,
        '--data', static_config['data'],
        '--root_path', static_config['root_path'],
        '--data_path', data_path,
        '--features', args.features,
        '--target', static_config['target'],
        '--seq_len', str(args.seq_len),
        '--label_len', str(args.label_len),
        '--pred_len', str(args.pred_len),
        '--patch_len', str(args.patch_len),
        '--stride', str(args.stride),
        '--enc_in', static_config['enc_in'],
        '--dec_in', static_config['dec_in'],
        '--c_out', static_config['c_out'],
        '--d_model', static_config['d_model'],
        '--d_ff', static_config['d_ff'],
        '--factor', static_config['factor'],
        '--itr', static_config['itr'],
        '--train_epochs', static_config['train_epochs'],
        '--batch_size', static_config['batch_size'],
        '--learning_rate', static_config['learning_rate'],
        '--llm_layers', str(args.llm_layers),
        '--model', static_config['model'],
        '--models_dir', static_config['models_dir'],
        '--num_tokens', str(args.num_tokens),
        '--llm_dim', str(args.llm_dim),
        '--results_csv', static_config['results_csv']
    ]
    
    
    # Always check if model already exists and ask for confirmation
    if check_model_exists(static_config['models_dir'], model_id):
        if not ask_override_confirmation(model_id):
            print("Experiment cancelled.")
            return
    
    # Print the command for verification
    print("Launching experiment with the following configuration:")
    print(f"Model ID: {model_id}")
    print(f"LLM Dimension: {args.llm_dim}")
    print(f"Task: {args.task_name}")
    print(f"Granularity: {args.granularity}")
    print(f"Features: {args.features}")
    print(f"Sequence Length: {args.seq_len}")
    print(f"Prediction Length: {args.pred_len}")
    print(f"Label Length: {args.label_len}")
    print(f"LLM Model: {args.llm_model}")
    print(f"LLM Layers: {args.llm_layers}")
    print(f"Num Tokens: {args.num_tokens}")
    print(f"Patch Length: {args.patch_len}")
    print(f"Stride: {args.stride}")
    print(f"Data Path: {data_path}")
    print()
    
    # Ask for confirmation
    if not args.auto_confirm:
        response = input("Do you want to proceed? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Experiment cancelled.")
            return
    
    # Launch the experiment with logging

    # Create log file path
    log_path = create_log_file(model_id, static_config['logs_dir'])
    
    # Write header to log file
    with open(log_path, 'w') as log_file:
        log_file.write(f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Command: {' '.join(cmd)}\n")
        log_file.write("=" * 80 + "\n\n")
    
    print(f"Logging output to: {log_path}")
    print("=" * 50)
    
    # Build the full command with pipe and tee
    cmd_str = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmd)
    full_cmd = f"{cmd_str} 2>&1 | tee -a '{log_path}'"
    
    # Run the command using shell with pipe and tee
    return_code = subprocess.run(full_cmd, shell=True, check=False).returncode
    
    # Write completion info to log
    with open(log_path, 'a') as log_file:
        log_file.write(f"\n\nExperiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"Log file saved to: {log_path}")

def main():
    parser = argparse.ArgumentParser(description='Launch TimeLLM experiments with simplified configuration')
    
    # Required arguments
    parser.add_argument('--llm_model', required=True, default='TimeLLM',
                       help='LLM model name (e.g., TimeLLM)')
    parser.add_argument('--llm_layers', type=int, required=True, default=8,
                       help='Number of LLM layers')
    parser.add_argument('--granularity', required=True, choices=['hourly', 'minute', 'daily'],
                       help='Data granularity (hourly, minute, daily)')
    parser.add_argument('--task_name', required=True, choices=['long_term_forecast', 'short_term_forecast'],
                       help='Task type')
    parser.add_argument('--features', required=True, choices=['M', 'MS', 'S'],
                       help='Feature type (M=multivariate, MS=multivariate w/ date, S=univariate)')
    parser.add_argument('--seq_len', type=int, required=True,
                       help='Input sequence length')
    parser.add_argument('--pred_len', type=int, required=True,
                       help='Prediction length')
    
    # Optional arguments with defaults
    parser.add_argument('--num_tokens', type=int, default=1000,
                       help='Number of tokens for vocabulary/mapping layer')
    parser.add_argument('--llm_dim', type=int, default='4096', 
                        help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
    parser.add_argument('--label_len', type=int, default=None,
                       help='Label length (defaults to seq_len//2 if not specified)')
    parser.add_argument('--patch_len', type=int, default=1,
                       help='Patch length (for short_term_forecast)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Stride (for short_term_forecast)')
    parser.add_argument('--auto_confirm', action='store_true',
                       help='Skip confirmation prompt and run immediately')
    
    args = parser.parse_args()
    
    # Set default label_len if not provided
    if args.label_len is None:
        args.label_len = args.seq_len // 2
    
    # Validate arguments
    if args.task_name == 'short_term_forecast' and (args.patch_len is None or args.stride is None):
        print("Error: patch_len and stride are required for short_term_forecast")
        sys.exit(1)
    
    launch_experiment(args)

if __name__ == '__main__':
    main()
