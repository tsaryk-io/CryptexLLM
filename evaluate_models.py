#!/usr/bin/env python3
"""
Model Evaluation Script for TimeLLM Enhanced Models

This script performs comprehensive evaluation of TimeLLM models using
walk-forward validation, trading performance metrics, and benchmarking.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.validation_evaluation import ValidationManager, ValidationConfig
    from utils.tools import load_content
    from data_provider.data_factory import data_provider
    EVALUATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Evaluation components not available: {e}")
    EVALUATION_AVAILABLE = False


def load_trained_model(model_path: str, args):
    """Load a trained TimeLLM model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Import model dynamically
    if args.model == 'TimeLLM':
        from models.TimeLLM import Model
    elif args.model == 'MultiScaleTimeLLM':
        from models.MultiScaleTimeLLM import Model
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # Load model
    model = Model(args)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model


def generate_predictions(model, data_loader, args):
    """Generate predictions from a trained model"""
    predictions = []
    actuals = []
    
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            # Prepare inputs
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float()
            
            # Generate predictions
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # Extract predictions and actuals
            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            
            predictions.append(pred)
            actuals.append(true)
    
    # Concatenate all predictions
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    # Reshape to 1D for evaluation (assuming single target)
    if len(predictions.shape) > 2:
        predictions = predictions[:, -args.pred_len:, -1]  # Last prediction, target feature
        actuals = actuals[:, -args.pred_len:, -1]
    
    # Flatten to 1D
    predictions = predictions.flatten()
    actuals = actuals.flatten()
    
    return predictions, actuals


def evaluate_single_model(model_path: str, args, validation_manager: ValidationManager):
    """Evaluate a single model"""
    print(f"\nEvaluating model: {model_path}")
    
    try:
        # Load model
        model = load_trained_model(model_path, args)
        
        # Load test data
        _, test_loader = data_provider(args, 'test')
        
        # Generate predictions
        predictions, actuals = generate_predictions(model, test_loader, args)
        
        print(f"Generated {len(predictions)} predictions")
        
        # Extract model name from path
        model_name = os.path.basename(model_path).replace('.pth', '')
        
        return model_name, predictions, actuals
        
    except Exception as e:
        print(f"Error evaluating model {model_path}: {e}")
        return None, None, None


def create_evaluation_config():
    """Create evaluation configuration"""
    return ValidationConfig(
        initial_train_size=1000,
        validation_size=100,
        step_size=24,
        min_train_size=500,
        max_validations=20,
        transaction_cost=0.001,
        confidence_levels=[0.95, 0.99],
        min_sharpe_ratio=0.5,
        min_win_rate=0.45
    )


def evaluate_models_directory(models_dir: str, args):
    """Evaluate all models in a directory"""
    print(f"Evaluating models in directory: {models_dir}")
    
    if not EVALUATION_AVAILABLE:
        print("Evaluation system not available - please install required dependencies")
        return
    
    # Find all model files
    model_files = []
    for file in os.listdir(models_dir):
        if file.endswith('.pth'):
            model_files.append(os.path.join(models_dir, file))
    
    if not model_files:
        print("No model files found in directory")
        return
    
    print(f"Found {len(model_files)} model files")
    
    # Create validation manager
    config = create_evaluation_config()
    validation_manager = ValidationManager(config)
    
    # Evaluate each model
    model_predictions = {}
    
    for model_path in model_files:
        model_name, predictions, actuals = evaluate_single_model(model_path, args, validation_manager)
        
        if model_name and predictions is not None:
            model_predictions[model_name] = (predictions, actuals)
    
    if not model_predictions:
        print("No successful model evaluations")
        return
    
    print(f"\nSuccessfully evaluated {len(model_predictions)} models")
    
    # Load original data for regime information
    data_set, _ = data_provider(args, 'test')
    
    # Create dummy dataframe for evaluation (in real scenario, load actual data)
    n_points = len(list(model_predictions.values())[0][1])
    dummy_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=n_points, freq='H'),
        'close': list(model_predictions.values())[0][1]
    })
    
    # Run comprehensive evaluation
    print("\nRunning comprehensive evaluation...")
    evaluation_results = validation_manager.comprehensive_evaluation(
        model_predictions=model_predictions,
        data=dummy_data,
        target_col='close',
        timestamp_col='timestamp'
    )
    
    # Generate final report
    output_dir = os.path.join(args.checkpoints, 'evaluation_results')
    report_file = validation_manager.generate_final_report(output_dir)
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Report file: {report_file}")
    
    # Print summary
    print(f"\nModel Performance Summary:")
    sorted_models = sorted(evaluation_results.items(), 
                          key=lambda x: x[1].overall_score, reverse=True)
    
    for i, (model_name, result) in enumerate(sorted_models):
        print(f"{i+1:2d}. {model_name:<30} Score: {result.overall_score:.3f} "
              f"Sharpe: {result.trading_metrics['sharpe_ratio']:6.3f} "
              f"Win Rate: {result.trading_metrics['win_rate']:6.2%}")
    
    return evaluation_results


def setup_args_for_evaluation(model_path: str):
    """Setup arguments for model evaluation based on model path"""
    
    # Create basic args object
    class Args:
        def __init__(self):
            # Model parameters
            self.model = 'TimeLLM'
            self.model_id = 'evaluation'
            
            # Data parameters
            self.data = 'CRYPTEX'
            self.root_path = './dataset/cryptex/'
            self.data_path = 'candlesticks-h.csv'
            self.features = 'M'
            self.target = 'close'
            self.freq = 'h'
            self.checkpoints = './checkpoints/'
            
            # Model architecture
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
            self.enc_in = 6
            self.dec_in = 6
            self.c_out = 6
            self.d_model = 64
            self.d_ff = 256
            self.n_heads = 8
            self.e_layers = 2
            self.d_layers = 1
            self.factor = 3
            self.dropout = 0.1
            self.activation = 'gelu'
            
            # LLM parameters
            self.llm_model = 'LLAMA'
            self.llm_layers = 8
            self.llm_dim = 4096
            
            # TimeLLM specific
            self.patch_len = 16
            self.stride = 8
            self.num_tokens = 1000
            
            # Training parameters
            self.batch_size = 32
            self.learning_rate = 0.0001
            self.loss = 'MSE'
            
            # Other
            self.use_amp = False
            self.prompt_domain = True
            self.content = ""
            self.percent = 100
            self.num_workers = 10
            self.seasonal_patterns = None
    
    args = Args()
    
    # Infer parameters from model path
    model_path_lower = model_path.lower()
    
    # Detect model type
    if 'multiscale' in model_path_lower:
        args.model = 'MultiScaleTimeLLM'
        args.data = 'CRYPTEX_MULTISCALE'
        args.enc_in = 80
        args.dec_in = 80 
        args.c_out = 80
    elif 'external' in model_path_lower:
        args.data = 'CRYPTEX_EXTERNAL'
        args.enc_in = 100
        args.dec_in = 100
        args.c_out = 100
    elif 'enhanced' in model_path_lower:
        args.data = 'CRYPTEX_ENHANCED'
        args.enc_in = 68
        args.dec_in = 68
        args.c_out = 68
    
    # Detect sequence lengths from filename
    if '_96_24' in model_path:
        args.seq_len = 96
        args.pred_len = 24
    elif '_168_48' in model_path:
        args.seq_len = 168
        args.pred_len = 48
    elif '_192_96' in model_path:
        args.seq_len = 192
        args.pred_len = 96
    
    # Load appropriate content for prompting
    if args.prompt_domain:
        try:
            args.content = load_content(args)
        except:
            args.content = "Cryptocurrency time series forecasting dataset"
    
    return args


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate TimeLLM Enhanced Models')
    
    parser.add_argument('--models_dir', type=str, required=True,
                       help='Directory containing trained model files')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Specific model file to evaluate')
    parser.add_argument('--output_dir', type=str, default='./results/evaluation/',
                       help='Output directory for evaluation results')
    parser.add_argument('--data_path', type=str, default='./dataset/cryptex/candlesticks-h.csv',
                       help='Path to test data')
    
    args = parser.parse_args()
    
    if not EVALUATION_AVAILABLE:
        print("Evaluation system not available - please install required dependencies")
        return
    
    # Evaluate specific model or all models in directory
    if args.model_path:
        # Evaluate single model
        model_args = setup_args_for_evaluation(args.model_path)
        config = create_evaluation_config()
        validation_manager = ValidationManager(config)
        
        model_name, predictions, actuals = evaluate_single_model(args.model_path, model_args, validation_manager)
        
        if model_name:
            print(f"Model {model_name} evaluated successfully")
            
            # Quick performance summary
            from utils.validation_evaluation import quick_evaluation
            performance = quick_evaluation(predictions, actuals, model_name)
            
            print(f"Performance Summary:")
            print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
            print(f"  Win Rate: {performance['win_rate']:.2%}")
            print(f"  Max Drawdown: {performance['max_drawdown']:.2%}")
            print(f"  Annual Return: {performance['annualized_return']:.2%}")
        
    else:
        # Evaluate all models in directory
        if not os.path.exists(args.models_dir):
            print(f"Models directory not found: {args.models_dir}")
            return
        
        # Use first model to setup base args
        model_files = [f for f in os.listdir(args.models_dir) if f.endswith('.pth')]
        if not model_files:
            print("No model files found")
            return
        
        first_model = os.path.join(args.models_dir, model_files[0])
        model_args = setup_args_for_evaluation(first_model)
        model_args.checkpoints = args.output_dir
        
        # Evaluate all models
        evaluation_results = evaluate_models_directory(args.models_dir, model_args)
        
        if evaluation_results:
            print(f"\nEvaluation completed for {len(evaluation_results)} models")
        else:
            print("No models were successfully evaluated")


if __name__ == "__main__":
    main()