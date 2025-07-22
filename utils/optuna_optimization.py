"""
Optuna integration for Time-LLM-Cryptex hyperparameter optimization
"""

import optuna
import optuna.visualization as vis
import mlflow
import torch
import numpy as np
import json
import os
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import subprocess
import tempfile


class OptunaOptimizer:
    """
    Optuna integration for optimizing Time-LLM hyperparameters
    """
    
    def __init__(self, study_name: str = None, storage_url: str = None, 
                 direction: str = "minimize", n_trials: int = 50):
        """
        Initialize Optuna optimizer
        
        Args:
            study_name: Name of the optimization study
            storage_url: Database URL for study persistence (optional)
            direction: "minimize" or "maximize" the objective
            n_trials: Number of optimization trials to run
        """
        if not study_name:
            study_name = f"timellm_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.study_name = study_name
        self.direction = direction
        self.n_trials = n_trials
        
        # Create study
        if storage_url:
            self.study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                direction=direction,
                load_if_exists=True
            )
        else:
            self.study = optuna.create_study(
                study_name=study_name,
                direction=direction
            )
        
        self.best_params = None
        self.best_value = None
    
    def suggest_timellm_params(self, trial: optuna.Trial, optimization_config: Dict = None) -> Dict[str, Any]:
        """
        Suggest hyperparameters for Time-LLM model
        
        Args:
            trial: Optuna trial object
            optimization_config: Configuration defining parameter search spaces
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        if not optimization_config:
            optimization_config = self.get_default_optimization_config()
        
        suggested_params = {}
        
        # Model architecture parameters
        if 'seq_len' in optimization_config:
            config = optimization_config['seq_len']
            suggested_params['seq_len'] = trial.suggest_int(
                'seq_len', config['min'], config['max'], step=config.get('step', 1)
            )
        
        if 'pred_len' in optimization_config:
            config = optimization_config['pred_len']
            suggested_params['pred_len'] = trial.suggest_int(
                'pred_len', config['min'], config['max'], step=config.get('step', 1)
            )
        
        if 'patch_len' in optimization_config:
            config = optimization_config['patch_len']
            suggested_params['patch_len'] = trial.suggest_int(
                'patch_len', config['min'], config['max'], step=config.get('step', 1)
            )
        
        if 'stride' in optimization_config:
            config = optimization_config['stride']
            suggested_params['stride'] = trial.suggest_int(
                'stride', config['min'], config['max'], step=config.get('step', 1)
            )
        
        if 'd_model' in optimization_config:
            config = optimization_config['d_model']
            if config.get('choices'):
                suggested_params['d_model'] = trial.suggest_categorical('d_model', config['choices'])
            else:
                suggested_params['d_model'] = trial.suggest_int(
                    'd_model', config['min'], config['max'], step=config.get('step', 1)
                )
        
        if 'd_ff' in optimization_config:
            config = optimization_config['d_ff']
            if config.get('choices'):
                suggested_params['d_ff'] = trial.suggest_categorical('d_ff', config['choices'])
            else:
                suggested_params['d_ff'] = trial.suggest_int(
                    'd_ff', config['min'], config['max'], step=config.get('step', 1)
                )
        
        if 'llm_layers' in optimization_config:
            config = optimization_config['llm_layers']
            suggested_params['llm_layers'] = trial.suggest_int(
                'llm_layers', config['min'], config['max'], step=config.get('step', 1)
            )
        
        if 'num_tokens' in optimization_config:
            config = optimization_config['num_tokens']
            if config.get('choices'):
                suggested_params['num_tokens'] = trial.suggest_categorical('num_tokens', config['choices'])
            else:
                suggested_params['num_tokens'] = trial.suggest_int(
                    'num_tokens', config['min'], config['max'], step=config.get('step', 1)
                )
        
        # Training parameters
        if 'learning_rate' in optimization_config:
            config = optimization_config['learning_rate']
            suggested_params['learning_rate'] = trial.suggest_float(
                'learning_rate', config['min'], config['max'], log=config.get('log', True)
            )
        
        if 'batch_size' in optimization_config:
            config = optimization_config['batch_size']
            if config.get('choices'):
                suggested_params['batch_size'] = trial.suggest_categorical('batch_size', config['choices'])
            else:
                suggested_params['batch_size'] = trial.suggest_int(
                    'batch_size', config['min'], config['max'], step=config.get('step', 1)
                )
        
        if 'train_epochs' in optimization_config:
            config = optimization_config['train_epochs']
            suggested_params['train_epochs'] = trial.suggest_int(
                'train_epochs', config['min'], config['max'], step=config.get('step', 1)
            )
        
        # Sentiment data weighting (if using real sentiment APIs)
        if 'sentiment_reddit_weight' in optimization_config:
            config = optimization_config['sentiment_reddit_weight']
            suggested_params['sentiment_reddit_weight'] = trial.suggest_float(
                'sentiment_reddit_weight', config['min'], config['max']
            )
        
        if 'sentiment_news_weight' in optimization_config:
            config = optimization_config['sentiment_news_weight']
            suggested_params['sentiment_news_weight'] = trial.suggest_float(
                'sentiment_news_weight', config['min'], config['max']
            )
        
        # Ensure sentiment weights sum to 1.0 if both are being optimized
        if 'sentiment_reddit_weight' in suggested_params and 'sentiment_news_weight' in suggested_params:
            reddit_w = suggested_params['sentiment_reddit_weight']
            news_w = suggested_params['sentiment_news_weight']
            fear_greed_w = max(0.1, 1.0 - reddit_w - news_w)  # Ensure at least 10% for fear & greed
            
            # Normalize to sum to 1.0
            total = reddit_w + news_w + fear_greed_w
            suggested_params['sentiment_reddit_weight'] = reddit_w / total
            suggested_params['sentiment_news_weight'] = news_w / total
            suggested_params['sentiment_fear_greed_weight'] = fear_greed_w / total
        
        return suggested_params
    
    def get_default_optimization_config(self) -> Dict[str, Dict[str, Any]]:
        """Get default optimization configuration for Time-LLM"""
        return {
            # Sequence parameters
            'seq_len': {'min': 48, 'max': 192, 'step': 24},
            'pred_len': {'min': 24, 'max': 96, 'step': 12},
            'patch_len': {'min': 1, 'max': 16, 'step': 1},
            'stride': {'min': 1, 'max': 8, 'step': 1},
            
            # Model architecture
            'd_model': {'choices': [16, 32, 64, 128]},
            'd_ff': {'choices': [32, 64, 128, 256]},
            'llm_layers': {'min': 4, 'max': 12, 'step': 2},
            'num_tokens': {'choices': [500, 1000, 2000, 3000]},
            
            # Training parameters
            'learning_rate': {'min': 1e-5, 'max': 1e-2, 'log': True},
            'batch_size': {'choices': [8, 16, 24, 32, 48]},
            'train_epochs': {'min': 5, 'max': 20, 'step': 5},
            
            # Sentiment data weighting
            'sentiment_reddit_weight': {'min': 0.1, 'max': 0.6},
            'sentiment_news_weight': {'min': 0.1, 'max': 0.6}
        }
    
    def create_objective_function(self, base_args: Any, optimization_config: Dict = None) -> Callable:
        """
        Create objective function for Optuna optimization
        
        Args:
            base_args: Base arguments for TimeLLM training
            optimization_config: Parameter search space configuration
        
        Returns:
            Objective function for Optuna
        """
        def objective(trial):
            try:
                # Get suggested parameters
                suggested_params = self.suggest_timellm_params(trial, optimization_config)
                
                # Create modified args with suggested parameters
                modified_args = self._create_modified_args(base_args, suggested_params)
                
                # Run training with these parameters
                result = self._run_training_trial(modified_args, trial.number)
                
                # Extract the objective value (e.g., validation MAE)
                objective_value = result.get('val_mae', result.get('test_mae', float('inf')))
                
                # Log to MLFlow if available
                self._log_trial_to_mlflow(trial, suggested_params, result)
                
                return objective_value
                
            except Exception as e:
                print(f"Trial {trial.number} failed with error: {e}")
                # Return a high value for minimization problems
                return float('inf') if self.direction == "minimize" else float('-inf')
        
        return objective
    
    def _create_modified_args(self, base_args: Any, suggested_params: Dict[str, Any]) -> Any:
        """Create a modified copy of args with suggested parameters"""
        import copy
        modified_args = copy.deepcopy(base_args)
        
        # Update args with suggested parameters
        for param_name, param_value in suggested_params.items():
            if hasattr(modified_args, param_name):
                setattr(modified_args, param_name, param_value)
        
        # Create unique model ID for this trial
        timestamp = datetime.now().strftime("%H%M%S")
        modified_args.model_id = f"optuna_trial_{timestamp}_{hash(str(suggested_params)) % 10000}"
        
        return modified_args
    
    def _run_training_trial(self, args: Any, trial_number: int) -> Dict[str, float]:
        """
        Run a single training trial with given parameters
        
        Returns:
            Dictionary with training results (losses, metrics)
        """
        # This would typically call your existing training script
        # For now, we'll create a simplified version that calls launch_experiment.py
        
        try:
            # Create command for launching experiment
            cmd = [
                'python', 'launch_experiment.py',
                '--adaptive', f'optuna_trial_{trial_number}',
                '--llm_model', args.llm_model,
                '--llm_layers', str(args.llm_layers),
                '--granularity', args.granularity,
                '--task_name', args.task_name,
                '--features', args.features,
                '--seq_len', str(args.seq_len),
                '--pred_len', str(args.pred_len),
                '--num_tokens', str(args.num_tokens),
                '--llm_dim', str(args.llm_dim),
                '--label_len', str(args.label_len),
                '--patch_len', str(args.patch_len),
                '--stride', str(args.stride),
                '--loss', str(args.loss),
                '--metric', str(args.metric),
                '--auto_confirm'  # Skip confirmation prompts
            ]
            
            # Run the training
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            # Parse results from output
            if result.returncode == 0:
                # Look for FINAL_METRICS in output
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if line.startswith('FINAL_METRICS:'):
                        metrics_json = line.replace('FINAL_METRICS:', '').strip()
                        metrics = json.loads(metrics_json)
                        return {
                            'val_mae': metrics.get('mae', float('inf')),
                            'test_loss': metrics.get('mse', float('inf')),
                            'model_id': metrics.get('model_id', 'unknown')
                        }
                
                # If no FINAL_METRICS found, return high penalty
                return {'val_mae': float('inf'), 'test_loss': float('inf')}
            else:
                print(f"Training failed with return code {result.returncode}")
                print(f"Error output: {result.stderr}")
                return {'val_mae': float('inf'), 'test_loss': float('inf')}
                
        except subprocess.TimeoutExpired:
            print(f"Trial {trial_number} timed out")
            return {'val_mae': float('inf'), 'test_loss': float('inf')}
        except Exception as e:
            print(f"Error running trial {trial_number}: {e}")
            return {'val_mae': float('inf'), 'test_loss': float('inf')}
    
    def _log_trial_to_mlflow(self, trial: optuna.Trial, params: Dict[str, Any], result: Dict[str, float]):
        """Log Optuna trial to MLFlow"""
        try:
            with mlflow.start_run(run_name=f"optuna_trial_{trial.number}"):
                # Log parameters
                for key, value in params.items():
                    mlflow.log_param(key, value)
                
                # Log trial info
                mlflow.log_param("trial_number", trial.number)
                mlflow.log_param("study_name", self.study_name)
                
                # Log metrics
                for key, value in result.items():
                    if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                        mlflow.log_metric(key, value)
        except Exception as e:
            print(f"Failed to log trial {trial.number} to MLFlow: {e}")
    
    def optimize(self, objective_function: Callable, n_trials: int = None, 
                 timeout: int = None, callbacks: List = None) -> optuna.Study:
        """
        Run hyperparameter optimization
        
        Args:
            objective_function: Function to optimize
            n_trials: Number of trials (overrides default)
            timeout: Timeout in seconds
            callbacks: List of callback functions
            
        Returns:
            Completed Optuna study
        """
        if n_trials is None:
            n_trials = self.n_trials
        
        print(f"Starting Optuna optimization: {n_trials} trials")
        print(f"Study name: {self.study_name}")
        print(f"Direction: {self.direction}")
        
        # Add pruning callback
        if not callbacks:
            callbacks = []
        
        # Add median pruner to stop unpromising trials early
        self.study.sampler = optuna.samplers.TPESampler()
        self.study.pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=1
        )
        
        # Run optimization
        self.study.optimize(
            objective_function,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks,
            show_progress_bar=True
        )
        
        # Store best results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        print(f"\nOptimization completed!")
        print(f"Best value: {self.best_value}")
        print(f"Best parameters: {json.dumps(self.best_params, indent=2)}")
        
        return self.study
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results"""
        if not self.study.trials:
            return {"status": "No trials completed"}
        
        return {
            "study_name": self.study_name,
            "direction": self.direction,
            "n_trials": len(self.study.trials),
            "best_value": self.best_value,
            "best_params": self.best_params,
            "n_complete_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_failed_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]),
            "n_pruned_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])
        }
    
    def save_study(self, filepath: str):
        """Save study to file"""
        study_data = {
            "study_name": self.study_name,
            "direction": self.direction,
            "best_params": self.best_params,
            "best_value": self.best_value,
            "trials": []
        }
        
        for trial in self.study.trials:
            trial_data = {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
                "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None
            }
            study_data["trials"].append(trial_data)
        
        with open(filepath, 'w') as f:
            json.dump(study_data, f, indent=2)
        
        print(f"Study saved to: {filepath}")
    
    def create_visualization(self, save_dir: str = "./optuna_plots"):
        """Create visualization plots for the optimization study"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        try:
            # Optimization history
            fig1 = vis.plot_optimization_history(self.study)
            fig1.write_html(os.path.join(save_dir, "optimization_history.html"))
            
            # Parameter importance
            fig2 = vis.plot_param_importances(self.study)
            fig2.write_html(os.path.join(save_dir, "param_importances.html"))
            
            # Parallel coordinate plot
            fig3 = vis.plot_parallel_coordinate(self.study)
            fig3.write_html(os.path.join(save_dir, "parallel_coordinate.html"))
            
            # Slice plot
            fig4 = vis.plot_slice(self.study)
            fig4.write_html(os.path.join(save_dir, "slice_plot.html"))
            
            print(f"Visualization plots saved to: {save_dir}")
            
        except Exception as e:
            print(f"Failed to create visualizations: {e}")


if __name__ == "__main__":
    # Test the Optuna integration
    print("Testing Optuna Integration...")
    
    # Create a simple test optimization
    optimizer = OptunaOptimizer("test_study", n_trials=3)
    
    def simple_objective(trial):
        x = trial.suggest_float('x', -10, 10)
        y = trial.suggest_float('y', -10, 10)
        return (x - 2) ** 2 + (y + 5) ** 2
    
    # Run optimization
    study = optimizer.optimize(simple_objective, n_trials=3)
    
    # Get summary
    summary = optimizer.get_optimization_summary()
    print("Optimization summary:")
    print(json.dumps(summary, indent=2))
    
    print("Optuna integration test completed!")