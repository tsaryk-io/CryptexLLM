#!/usr/bin/env python3

"""
DeepSeek Training with Adaptive Loss Functions

This script demonstrates training Time-LLM-Cryptex DeepSeek models with the new
adaptive loss system that can dynamically adjust loss function weights during training.

Features:
- Multiple loss combination strategies
- Dynamic weight adaptation based on performance
- Real-time loss monitoring and visualization
- Integration with feature selection optimization
- Comprehensive logging and checkpointing

Author: Claude (Anthropic)
Purpose: Production-ready adaptive loss training for Time-LLM-Cryptex
"""

import argparse
import os
import sys
import time
import torch
import json
import pandas as pd
from datetime import datetime


# Add project root to path
sys.path.append('.')

# Import adaptive loss system
from utils.adaptive_loss import LossSelectionManager, create_adaptive_loss
from utils.adaptive_loss_trainer import create_adaptive_trainer

# Import existing infrastructure
from data_provider.data_factory import data_provider
from data_provider.enhanced_data_loader_optimized import Dataset_CRYPTEX_Enhanced_Optimized
from data_provider.data_factory import data_dict

# Register optimized dataset
data_dict['CRYPTEX_ENHANCED_OPTIMIZED'] = Dataset_CRYPTEX_Enhanced_Optimized


def create_adaptive_training_config():
    """Create configuration for adaptive loss training"""
    parser = argparse.ArgumentParser(description='DeepSeek Adaptive Loss Training')
    
    # Basic configuration
    parser.add_argument('--model_id', type=str, default='DeepSeek_Adaptive', help='Model identifier')
    parser.add_argument('--model', type=str, default='TimeLLM', help='Model name')
    
    # Data configuration
    parser.add_argument('--data', type=str, default='CRYPTEX_ENHANCED_OPTIMIZED', help='Dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset', help='Root path of data')
    parser.add_argument('--data_path', type=str, default='cryptex/candlesticks-D.csv', help='Data file path')
    parser.add_argument('--features', type=str, default='M', help='Forecasting task [M, S, MS]')
    parser.add_argument('--target', type=str, default='close', help='Target feature')
    parser.add_argument('--freq', type=str, default='d', help='Frequency for time encoding')
    parser.add_argument('--embed', type=str, default='timeF', help='Time features encoding')
    
    # Model configuration - Optimized for performance
    parser.add_argument('--seq_len', type=int, default=21, help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=7, help='Label length')
    parser.add_argument('--pred_len', type=int, default=7, help='Prediction length')
    parser.add_argument('--enc_in', type=int, default=26, help='Encoder input size (6 OHLCV + 20 optimized features)')
    parser.add_argument('--dec_in', type=int, default=26, help='Decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='Output size')
    parser.add_argument('--d_model', type=int, default=32, help='Dimension of model')
    parser.add_argument('--d_ff', type=int, default=128, help='Dimension of fcn')
    
    # LLM configuration
    parser.add_argument('--llm_model', type=str, default='DEEPSEEK', help='LLM model type')
    parser.add_argument('--llm_dim', type=int, default=4096, help='LLM dimension')
    parser.add_argument('--llm_layers', type=int, default=8, help='Number of LLM layers')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (optimized for memory)')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--train_epochs', type=int, default=15, help='Training epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--lradj', type=str, default='type1', help='Learning rate adjustment')
    
    # Adaptive Loss Configuration
    parser.add_argument('--loss_combination', type=str, default='trading_focused', 
                       choices=['basic', 'trading_focused', 'robust_prediction', 'comprehensive', 'directional_focused'],
                       help='Loss combination strategy')
    parser.add_argument('--adaptation_strategy', type=str, default='performance_based',
                       choices=['performance_based', 'learning_based', 'hybrid'],
                       help='Weight adaptation strategy')
    parser.add_argument('--adaptation_frequency', type=int, default=50, help='Steps between adaptations')
    parser.add_argument('--performance_window', type=int, default=20, help='Window for performance evaluation')
    
    # System configuration
    parser.add_argument('--use_gpu', type=bool, default=True, help='Use GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    parser.add_argument('--num_workers', type=int, default=8, help='Data loader workers')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='Checkpoint directory')
    
    # Monitoring and visualization
    parser.add_argument('--save_loss_plots', type=bool, default=True, help='Save loss evolution plots')
    parser.add_argument('--log_frequency', type=int, default=1, help='Logging frequency (epochs)')
    parser.add_argument('--save_checkpoints', type=bool, default=True, help='Save model checkpoints')
    
    # Feature selection integration
    parser.add_argument('--feature_selection_config', type=str, 
                       default='./feature_selection_results/feature_selection_config.json',
                       help='Feature selection configuration file')
    
    return parser


def setup_model_and_data(args):
    """Setup model and data loaders"""
    print("=" * 60)
    print("SETTING UP MODEL AND DATA")
    print("=" * 60)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print(f"Loading dataset: {args.data}")
    print(f"Data path: {args.data_path}")
    print(f"Features: {args.enc_in} (optimized from 68+)")
    
    train_data, train_loader = data_provider(args, flag='train')
    val_data, val_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')
    
    print(f"Data loaded successfully:")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    print(f"  Test samples: {len(test_data)}")
    
    # Create model
    from models.TimeLLM import Model
    model = Model(args).to(device)
    
    print(f"Model created: {args.model}")
    print(f"  LLM model: {args.llm_model}")
    print(f"  LLM dimension: {args.llm_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    return model, optimizer, device, (train_loader, val_loader, test_loader)


def setup_adaptive_loss(args):
    """Setup adaptive loss system"""
    print("\n" + "=" * 60)
    print("SETTING UP ADAPTIVE LOSS SYSTEM")
    print("=" * 60)
    
    # Get loss combination info
    manager = LossSelectionManager()
    combination_info = manager.get_combination_info(args.loss_combination)
    
    print(f"Loss combination: {args.loss_combination}")
    print(f"Description: {combination_info['description']}")
    print(f"Loss functions: {combination_info['losses']}")
    print(f"Recommended for: {combination_info['recommended_for']}")
    
    print(f"\nAdaptation configuration:")
    print(f"  Strategy: {args.adaptation_strategy}")
    print(f"  Frequency: {args.adaptation_frequency} steps")
    print(f"  Performance window: {args.performance_window} steps")
    
    # Create adaptive loss
    adaptive_loss = create_adaptive_loss(
        combination=args.loss_combination,
        adaptation_strategy=args.adaptation_strategy,
        adaptation_frequency=args.adaptation_frequency,
        performance_window=args.performance_window
    )
    
    return adaptive_loss


def run_adaptive_training(args):
    """Run the complete adaptive training pipeline"""
    print("DEEPSEEK ADAPTIVE LOSS TRAINING")
    print("=" * 80)
    print(f"Model ID: {args.model_id}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    model, optimizer, device, (train_loader, val_loader, test_loader) = setup_model_and_data(args)
    adaptive_loss = setup_adaptive_loss(args)
    
    # Create adaptive trainer
    trainer = create_adaptive_trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        loss_combination=args.loss_combination,
        adaptation_strategy=args.adaptation_strategy,
        model_id=args.model_id
    )
    
    # Override with our configured adaptive loss
    trainer.adaptive_loss = adaptive_loss
    
    print(f"\n{'='*60}")
    print("STARTING ADAPTIVE TRAINING")
    print(f"{'='*60}")
    
    training_start = time.time()
    
    # Run training
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.train_epochs,
        patience=args.patience,
        save_best=args.save_checkpoints,
        log_frequency=args.log_frequency
    )
    
    training_time = time.time() - training_start
    
    # Results and analysis
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED - ANALYZING RESULTS")
    print(f"{'='*60}")
    
    # Performance analysis
    analysis = trainer.analyze_loss_performance()
    
    print(f"Training Results:")
    print(f"  Total time: {training_time/60:.2f} minutes")
    print(f"  Best validation loss: {results['best_val_loss']:.6f}")
    print(f"  Total epochs: {results['total_epochs']}")
    print(f"  Best weights: {[f'{k}:{v:.3f}' for k, v in results['best_weights'].items()]}")
    
    if analysis.get('adaptation_effectiveness'):
        eff = analysis['adaptation_effectiveness']
        print(f"\nAdaptation Effectiveness:")
        print(f"  Initial loss: {eff['initial_loss']:.6f}")
        print(f"  Final loss: {eff['final_loss']:.6f}")
        print(f"  Improvement: {eff['improvement_percent']:.2f}%")
        print(f"  Converged: {eff['converged']}")
    
    if analysis.get('best_performing_combinations'):
        print(f"\nBest Performing Weight Combinations:")
        for combo in analysis['best_performing_combinations'][:3]:
            print(f"  Rank {combo['rank']}: Loss {combo['val_loss']:.6f}")
            print(f"    Weights: {[f'{k}:{v:.3f}' for k, v in combo['weights'].items()]}")
    
    # Save results
    save_results(args, results, analysis, trainer, training_time)
    
    # Visualization
    if args.save_loss_plots:
        plot_path = f"./results/adaptive_loss_{args.model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        trainer.visualize_loss_evolution(save_path=plot_path)
    
    return results, analysis


def save_results(args, results, analysis, trainer, training_time):
    """Save comprehensive training results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"./results/adaptive_loss_{args.model_id}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Main results
    full_results = {
        'experiment_info': {
            'model_id': args.model_id,
            'timestamp': timestamp,
            'training_time_minutes': training_time / 60,
            'configuration': vars(args)
        },
        'training_results': results,
        'adaptation_analysis': analysis,
        'adaptive_loss_summary': trainer.adaptive_loss.get_performance_summary()
    }
    
    with open(f"{results_dir}/training_results.json", 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    
    # Save adaptation history
    trainer.adaptive_loss.save_adaptation_history(f"{results_dir}/adaptation_history.json")
    
    # Save training history
    if trainer.training_history:
        training_df = pd.DataFrame(trainer.training_history)
        training_df.to_csv(f"{results_dir}/training_history.csv", index=False)
    
    if trainer.validation_history:
        validation_df = pd.DataFrame(trainer.validation_history)
        validation_df.to_csv(f"{results_dir}/validation_history.csv", index=False)
    
    print(f"\nResults saved to: {results_dir}")
    print(f"  • training_results.json - Complete results")
    print(f"  • adaptation_history.json - Loss adaptation details")
    print(f"  • training_history.csv - Training metrics per epoch")
    print(f"  • validation_history.csv - Validation metrics per epoch")


def main():
    """Main training function"""
    # Parse arguments
    parser = create_adaptive_training_config()
    args = parser.parse_args()
    
    # Check feature selection config
    if not os.path.exists(args.feature_selection_config):
        print(f"Feature selection config not found: {args.feature_selection_config}")
        print("Consider running feature selection first:")
        print("  python test_feature_selection_simple.py")
        
        # Use default if file doesn't exist
        args.feature_selection_config = None
    
    # Run training
    try:
        results, analysis = run_adaptive_training(args)
        
        print(f"\nADAPTIVE LOSS TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Model trained with adaptive loss combination: {args.loss_combination}")
        print(f"Best validation loss: {results['best_val_loss']:.6f}")
        print(f"Training time: {results['training_time_minutes']:.2f} minutes")
        
        # Recommendations for next steps
        print(f"\nNEXT STEPS:")
        print(f"1. Run inference to test predictions:")
        print(f"   python inference.py --model_id {args.model_id}")
        print(f"2. Compare with baseline models")
        print(f"3. Experiment with different loss combinations:")
        print(f"   python train_deepseek_adaptive.py --loss_combination comprehensive")
        print(f"4. Analyze loss evolution plots in ./results/")
        
        return True
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    
    if not success:
        sys.exit(1)