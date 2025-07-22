#!/usr/bin/env python3
"""
Hyperparameter optimization script using Optuna with MLFlow tracking
for Time-LLM-Cryptex with sentiment data integration
"""

import argparse
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.optuna_optimization import OptunaOptimizer
from utils.mlflow_integration import MLFlowExperimentTracker


class TimeLLMOptimizerConfig:
    """Configuration for TimeLLM optimization"""
    
    @staticmethod
    def get_crypto_optimization_config() -> Dict[str, Dict[str, Any]]:
        """
        Optimized parameter ranges specifically for crypto prediction with sentiment data
        """
        return {
            # Time series parameters - optimized for crypto volatility
            'seq_len': {'min': 96, 'max': 336, 'step': 24},  # 4 hours to 14 days (hourly data)
            'pred_len': {'min': 24, 'max': 168, 'step': 12}, # 1 day to 1 week prediction
            'patch_len': {'min': 1, 'max': 12, 'step': 1},   # Hourly to half-daily patches
            'stride': {'min': 1, 'max': 6, 'step': 1},       # Overlapping patterns
            
            # Model architecture - balanced for crypto complexity
            'd_model': {'choices': [32, 64, 128, 256]},       # Model dimension
            'd_ff': {'choices': [64, 128, 256, 512]},         # Feed-forward dimension
            'llm_layers': {'min': 6, 'max': 16, 'step': 2},  # LLM depth
            'num_tokens': {'choices': [1000, 2000, 3000, 5000]}, # Vocabulary size
            
            # Training parameters
            'learning_rate': {'min': 1e-5, 'max': 5e-3, 'log': True},
            'batch_size': {'choices': [16, 24, 32, 48, 64]},
            'train_epochs': {'min': 10, 'max': 30, 'step': 5},
            
            # Sentiment data weighting - optimize real data impact
            'sentiment_reddit_weight': {'min': 0.2, 'max': 0.5},    # Social media
            'sentiment_news_weight': {'min': 0.2, 'max': 0.5},       # News impact
            # fear_greed_weight will be 1.0 - reddit_weight - news_weight
        }
    
    @staticmethod
    def get_quick_test_config() -> Dict[str, Dict[str, Any]]:
        """Smaller parameter space for quick testing"""
        return {
            'seq_len': {'min': 96, 'max': 192, 'step': 48},
            'pred_len': {'min': 24, 'max': 48, 'step': 24},
            'd_model': {'choices': [32, 64]},
            'd_ff': {'choices': [64, 128]},
            'llm_layers': {'min': 6, 'max': 10, 'step': 2},
            'learning_rate': {'min': 1e-4, 'max': 1e-3, 'log': True},
            'batch_size': {'choices': [16, 24, 32]},
            'train_epochs': {'min': 5, 'max': 10, 'step': 5},
        }


def create_base_args(granularity: str = "hourly", task_name: str = "long_term_forecast"):
    """Create base arguments for optimization"""
    
    class BaseArgs:
        def __init__(self):
            # Fixed parameters
            self.llm_model = 'TimeLLM'
            self.llm_dim = 4096
            self.granularity = granularity
            self.task_name = task_name
            self.features = 'M'  # Multivariate with sentiment data
            self.loss = 'MSE'
            self.metric = 'MAE'
            
            # Will be optimized
            self.seq_len = 96
            self.pred_len = 24
            self.patch_len = 1
            self.stride = 1
            self.label_len = 48
            self.llm_layers = 8
            self.d_model = 32
            self.d_ff = 128
            self.num_tokens = 1000
            self.learning_rate = 0.001
            self.batch_size = 24
            self.train_epochs = 10
    
    return BaseArgs()


def run_optimization(args):
    """Run hyperparameter optimization"""
    
    print("=" * 80)
    print("TIME-LLM-CRYPTEX HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"Task: {args.task_name}")
    print(f"Granularity: {args.granularity}")
    print(f"Optimization mode: {args.optimization_mode}")
    print(f"Number of trials: {args.n_trials}")
    print(f"MLFlow tracking: {'Enabled' if args.enable_mlflow else 'Disabled'}")
    
    # Initialize MLFlow experiment tracker
    mlflow_tracker = None
    if args.enable_mlflow:
        experiment_name = f"TimeLLM-Optimization-{args.granularity}-{datetime.now().strftime('%Y%m%d')}"
        mlflow_tracker = MLFlowExperimentTracker(experiment_name)
        print(f"MLFlow experiment: {experiment_name}")
    
    # Create study name
    study_name = f"timellm_{args.granularity}_{args.optimization_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize Optuna optimizer
    optimizer = OptunaOptimizer(
        study_name=study_name,
        direction="minimize",  # Minimize MAE
        n_trials=args.n_trials
    )
    
    # Get optimization configuration
    if args.optimization_mode == "quick":
        optimization_config = TimeLLMOptimizerConfig.get_quick_test_config()
        print("Using quick test configuration (reduced parameter space)")
    else:
        optimization_config = TimeLLMOptimizerConfig.get_crypto_optimization_config()
        print("Using full crypto optimization configuration")
    
    print(f"Optimizing parameters: {list(optimization_config.keys())}")
    
    # Create base arguments
    base_args = create_base_args(args.granularity, args.task_name)
    
    # Log optimization setup to MLFlow
    if mlflow_tracker:
        mlflow_tracker.start_run(f"optimization_setup_{study_name}")
        mlflow_tracker.log_params({
            "optimization_mode": args.optimization_mode,
            "n_trials": args.n_trials,
            "study_name": study_name,
            "base_granularity": args.granularity,
            "base_task_name": args.task_name,
            "parameter_count": len(optimization_config)
        })
        mlflow_tracker.log_config({
            "optimization_config": optimization_config,
            "base_args": vars(base_args)
        })
        mlflow_tracker.end_run()
    
    # Create objective function
    objective_function = optimizer.create_objective_function(base_args, optimization_config)
    
    # Run optimization
    print(f"\nStarting optimization with {args.n_trials} trials...")
    study = optimizer.optimize(
        objective_function, 
        n_trials=args.n_trials,
        timeout=args.timeout
    )
    
    # Get results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    
    summary = optimizer.get_optimization_summary()
    print(f"Study: {summary['study_name']}")
    print(f"Total trials: {summary['n_trials']}")
    print(f"Completed: {summary['n_complete_trials']}")
    print(f"Failed: {summary['n_failed_trials']}")
    print(f"Pruned: {summary['n_pruned_trials']}")
    print(f"Best value (MAE): {summary['best_value']:.6f}")
    print(f"Best parameters:")
    for key, value in summary['best_params'].items():
        print(f"  {key}: {value}")
    
    # Save results
    results_dir = "./optimization_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Save study
    study_file = os.path.join(results_dir, f"{study_name}.json")
    optimizer.save_study(study_file)
    
    # Create visualizations
    if args.create_plots:
        plot_dir = os.path.join(results_dir, f"{study_name}_plots")
        optimizer.create_visualization(plot_dir)
        print(f"Visualization plots saved to: {plot_dir}")
    
    # Log final results to MLFlow
    if mlflow_tracker:
        mlflow_tracker.start_run(f"optimization_results_{study_name}")
        mlflow_tracker.log_params(summary['best_params'])
        mlflow_tracker.log_metrics({
            "best_mae": summary['best_value'],
            "n_trials_completed": summary['n_complete_trials'],
            "success_rate": summary['n_complete_trials'] / summary['n_trials']
        })
        mlflow_tracker.log_config({"optimization_summary": summary})
        mlflow_tracker.end_run()
    
    print(f"Optimization completed!")
    print(f"Results saved to: {study_file}")
    
    # Suggest next steps
    print(f"Next steps:")
    print(f"1. Run training with best parameters:")
    print(f"   python launch_experiment.py --adaptive optimized \\")
    for key, value in summary['best_params'].items():
        if key in ['seq_len', 'pred_len', 'patch_len', 'stride', 'llm_layers', 'num_tokens', 'batch_size', 'train_epochs']:
            print(f"     --{key} {value} \\")
        elif key == 'learning_rate':
            print(f"     --learning_rate {value:.6f} \\")
    print(f"     --granularity {args.granularity} --task_name {args.task_name} --features M")
    
    print(f"\n2. View MLFlow UI: mlflow ui")
    print(f"3. Check plots in: {plot_dir if args.create_plots else 'Enable --create_plots'}")


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for Time-LLM-Cryptex')
    
    parser.add_argument('--granularity', type=str, default='hourly', 
                       choices=['hourly', 'minute', 'daily'],
                       help='Data granularity for optimization')
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                       choices=['long_term_forecast', 'short_term_forecast'],
                       help='Task type for optimization')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of optimization trials')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout in seconds for optimization')
    parser.add_argument('--optimization_mode', type=str, default='full',
                       choices=['full', 'quick'],
                       help='Optimization mode: full or quick test')
    parser.add_argument('--enable_mlflow', action='store_true',
                       help='Enable MLFlow experiment tracking')
    parser.add_argument('--create_plots', action='store_true',
                       help='Create Optuna visualization plots')
    
    args = parser.parse_args()
    
    # Validation
    if args.n_trials < 5:
        print("Warning: Very few trials may not find good parameters. Consider n_trials >= 20")
    
    # Run optimization
    run_optimization(args)


if __name__ == "__main__":
    main()