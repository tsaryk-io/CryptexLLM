#!/usr/bin/env python3
"""
Test script for MLFlow and Optuna integration
"""

import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.mlflow_integration import MLFlowExperimentTracker
from utils.optuna_optimization import OptunaOptimizer


def test_mlflow_integration():
    """Test MLFlow experiment tracking"""
    print("Testing MLFlow Integration...")
    
    try:
        # Initialize tracker
        tracker = MLFlowExperimentTracker("test_timellm_integration")
        
        # Start run
        run_id = tracker.start_run("test_run_with_sentiment", 
                                  tags={"test": "true", "sentiment_enabled": "true"})
        
        # Log parameters
        test_params = {
            "seq_len": 96,
            "pred_len": 24,
            "llm_model": "TimeLLM",
            "learning_rate": 0.001,
            "batch_size": 32,
            "sentiment_reddit_weight": 0.3,
            "sentiment_news_weight": 0.3,
            "sentiment_fear_greed_weight": 0.4
        }
        tracker.log_params(test_params)
        
        # Log sentiment data info
        sentiment_info = {
            "reddit_enabled": True,
            "news_enabled": True,
            "fear_greed_enabled": True,
            "total_sources": 3
        }
        tracker.log_sentiment_data_info(sentiment_info)
        
        # Simulate training progress
        for epoch in range(1, 4):
            train_loss = 0.5 - epoch * 0.1
            val_loss = 0.4 - epoch * 0.08
            test_loss = 0.45 - epoch * 0.09
            test_mae = 0.35 - epoch * 0.07
            
            tracker.log_training_progress(epoch, train_loss, val_loss, test_loss, test_mae)
        
        # Log final metrics
        final_metrics = {
            "final_test_loss": 0.18,
            "final_test_mae": 0.14,
            "best_epoch": 3
        }
        tracker.log_metrics(final_metrics)
        
        # End run
        tracker.end_run()
        
        print(f"MLFlow test completed successfully!")
        print(f"   Run ID: {run_id}")
        print(f"   Check ./mlruns directory for experiment data")
        
        return True
        
    except Exception as e:
        print(f"MLFlow test failed: {e}")
        return False


def test_optuna_integration():
    """Test Optuna optimization"""
    print("Testing Optuna Integration...")
    
    try:
        # Create optimizer
        optimizer = OptunaOptimizer("test_crypto_optimization", n_trials=5)
        
        # Define test optimization config
        test_config = {
            'seq_len': {'min': 48, 'max': 96, 'step': 24},
            'learning_rate': {'min': 1e-4, 'max': 1e-2, 'log': True},
            'batch_size': {'choices': [16, 32]},
            'sentiment_reddit_weight': {'min': 0.2, 'max': 0.5}
        }
        
        # Simple objective function that simulates crypto prediction
        def crypto_objective(trial):
            # Get suggested parameters
            params = optimizer.suggest_timellm_params(trial, test_config)
            
            # Simulate model performance based on parameters
            # Better performance with larger seq_len and balanced sentiment weights
            seq_len_score = 0.3 - (params['seq_len'] - 48) / 200  # Prefer longer sequences
            lr_penalty = abs(params['learning_rate'] - 0.001) * 100  # Prefer lr around 0.001
            
            # Simulate sentiment data impact
            reddit_weight = params.get('sentiment_reddit_weight', 0.3)
            sentiment_balance = 1.0 - abs(0.35 - reddit_weight)  # Prefer balanced sentiment
            
            # Combine factors (lower is better for MAE)
            mae = seq_len_score + lr_penalty * 0.1 + (1 - sentiment_balance) * 0.1
            mae = max(0.05, mae)  # Minimum realistic MAE
            
            return mae
        
        # Run optimization
        study = optimizer.optimize(crypto_objective, n_trials=5)
        
        # Get results
        summary = optimizer.get_optimization_summary()
        
        print(f"Optuna test completed successfully!")
        print(f"   Best MAE: {summary['best_value']:.4f}")
        print(f"   Best parameters: {json.dumps(summary['best_params'], indent=2)}")
        print(f"   Completed trials: {summary['n_complete_trials']}/5")
        
        return True
        
    except Exception as e:
        print(f"Optuna test failed: {e}")
        return False


def test_integration_with_sentiment_data():
    """Test integration with sentiment data configuration"""
    print("Testing Integration with Sentiment Data...")
    
    try:
        # Check if sentiment configuration exists
        config_file = "external_data_config.json"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                sentiment_config = json.load(f)
            
            print(f"Found sentiment configuration:")
            print(f"   Reddit: Always enabled")
            print(f"   News: {'Enabled' if sentiment_config.get('sentiment', {}).get('api_key') else 'No API key'}")
            print(f"   Fear & Greed: Always enabled")
            
            # Test MLFlow with sentiment data
            tracker = MLFlowExperimentTracker("test_sentiment_integration")
            tracker.start_run("sentiment_data_test")
            
            # Log sentiment configuration
            tracker.log_config({"sentiment_config": sentiment_config})
            
            # Log external data stats (simulated)
            external_data_stats = {
                "sentiment": {
                    "reddit_posts_fetched": 150,
                    "news_articles_fetched": 89,
                    "fear_greed_days": 30,
                    "data_sources_active": 3 if sentiment_config.get('sentiment', {}).get('api_key') else 2
                }
            }
            tracker.log_external_data_stats(external_data_stats)
            
            tracker.end_run()
            
            print(f"Sentiment data integration test completed!")
            return True
        else:
            print(f"No sentiment configuration found at {config_file}")
            print(f"   Run test_full_sentiment.py first to test sentiment APIs")
            return False
            
    except Exception as e:
        print(f"Sentiment integration test failed: {e}")
        return False


def main():
    print("=" * 70)
    print("TESTING MLFLOW & OPTUNA INTEGRATION FOR TIME-LLM-CRYPTEX")
    print("=" * 70)
    
    # Run tests
    tests = [
        ("MLFlow Experiment Tracking", test_mlflow_integration),
        ("Optuna Hyperparameter Optimization", test_optuna_integration),
        ("Sentiment Data Integration", test_integration_with_sentiment_data),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    if all_passed:
        print("\nReady to run optimization!")
        print("Usage examples:")
        print("1. Quick test: python run_optimization.py --optimization_mode quick --n_trials 10 --enable_mlflow")
        print("2. Full optimization: python run_optimization.py --n_trials 50 --enable_mlflow --create_plots")
        print("3. View results: mlflow ui")


if __name__ == "__main__":
    main()