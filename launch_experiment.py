import argparse
import subprocess
import sys
import os

def generate_model_id(llm_model, llm_layers, granularity, features, seq_len, pred_len, num_tokens):
    """Generate comprehensive model ID including all key parameters"""
    granularity_map = {
        'hourly': 'h',
        'minute': 'Min', 
        'daily': 'd'
    }
    
    gran_short = granularity_map[granularity]
    

    model_id = f"{llm_model}_L{llm_layers}_{granularity}_{features}_seq{seq_len}_pred{pred_len}_V{num_tokens}"
    
    return model_id


def get_data_path(granularity):
    """Get data path based on granularity"""
    granularity_to_file = {
        'hourly': 'candlesticks-h.csv',
        'minute': 'candlesticks-Min.csv',
        'daily': 'candlesticks-D.csv'
    }
    return granularity_to_file[granularity]

def launch_experiment(args):
    """Launch the TimeLLM experiment with specified parameters"""
    
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
        'models_dir':'/mnt/data/trained_models'
    }
    
    # Generate dynamic parameters
    model_id = generate_model_id(args.llm_model, args.llm_layers, args.granularity, args.features, args.seq_len, args.pred_len, args.num_tokens)
    data_path = get_data_path(args.granularity)
    
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
        '--num_tokens', str(args.num_tokens)
    ]
    
    # Add patch_len and stride for short_term_forecast
    if args.task_name == 'short_term_forecast':
        cmd.extend(['--patch_len', str(args.patch_len)])
        cmd.extend(['--stride', str(args.stride)])
    
    # Print the command for verification
    print("Launching experiment with the following configuration:")
    print(f"Model ID: {model_id}")
    print(f"Task: {args.task_name}")
    print(f"Granularity: {args.granularity}")
    print(f"Features: {args.features}")
    print(f"Sequence Length: {args.seq_len}")
    print(f"Prediction Length: {args.pred_len}")
    print(f"Label Length: {args.label_len}")
    print(f"LLM Model: {args.llm_model}")
    print(f"LLM Layers: {args.llm_layers}")
    print(f"Num Tokens: {args.num_tokens}")
    if args.task_name == 'short_term_forecast':
        print(f"Patch Length: {args.patch_len}")
        print(f"Stride: {args.stride}")
    print(f"Data Path: {data_path}")
    print()
    print("Command:")
    ### This part is just formatting
    formatted_cmd = "accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \\\n"
    
    param_pairs = []
    for i in range(6, len(cmd)):  # Skip the accelerate launch part
        if not cmd[i].startswith('--'):
            continue
        param_name = cmd[i]
        param_value = cmd[i+1] if i+1 < len(cmd) and not cmd[i+1].startswith('--') else ''
        param_pairs.append(f"  {param_name} {param_value}")
    
    formatted_cmd += " \\\n".join(param_pairs) + " \\\n"
    
    print(formatted_cmd)
    ### End of formatting
    print()
    
    # Ask for confirmation
    if not args.auto_confirm:
        response = input("Do you want to proceed? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Experiment cancelled.")
            return
    
    # Launch the experiment
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with return code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        sys.exit(1)

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
