#!/usr/bin/env python3
"""
Comprehensive Hyperparameter Optimization Framework for TimeLLM

This module provides advanced hyperparameter optimization capabilities using
Optuna, Ray Tune, and custom optimization strategies specifically designed
for TimeLLM cryptocurrency prediction models.
"""

import os
import sys
import json
import time
import logging
import traceback
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Optional dependencies
try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    # Create dummy classes for type hints
    class optuna:
        class Study:
            pass
        class Trial:
            pass

try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    from ray.tune.suggest.optuna import OptunaSearch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization"""
    
    # Optimization method
    method: str = "optuna"  # "optuna", "ray", "grid", "random"
    n_trials: int = 100
    timeout: Optional[int] = None  # seconds
    
    # Objective configuration
    primary_metric: str = "mae"  # Primary optimization metric
    direction: str = "minimize"  # "minimize" or "maximize"
    multi_objective: bool = False
    secondary_metrics: List[str] = field(default_factory=lambda: ["mse", "sharpe_ratio"])
    
    # Resource constraints
    max_concurrent_trials: int = 4
    gpu_memory_limit: float = 8.0  # GB
    max_training_time: int = 3600  # seconds per trial
    
    # Pruning and early stopping
    enable_pruning: bool = True
    patience: int = 5
    min_epochs: int = 3
    
    # Data configuration
    datasets: List[str] = field(default_factory=lambda: ["CRYPTEX_ENHANCED"])
    validation_split: float = 0.2
    
    # Output configuration
    study_name: str = "TimeLLM_HPO"
    output_dir: str = "./results/hyperparameter_optimization/"
    save_intermediate: bool = True


@dataclass 
class TrialResult:
    """Result from a single optimization trial"""
    trial_id: int
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    training_time: float
    epochs_trained: int
    status: str  # "completed", "pruned", "failed"
    error_message: Optional[str] = None


class HyperparameterSpace:
    """Defines the hyperparameter search space for TimeLLM"""
    
    def __init__(self, optimization_type: str = "comprehensive"):
        self.optimization_type = optimization_type
        self.space = self._define_search_space()
        
    def _define_search_space(self) -> Dict[str, Dict[str, Any]]:
        """Define comprehensive hyperparameter search space"""
        
        if self.optimization_type == "quick":
            return self._quick_search_space()
        elif self.optimization_type == "architecture":
            return self._architecture_search_space()
        elif self.optimization_type == "training":
            return self._training_search_space()
        else:
            return self._comprehensive_search_space()
    
    def _comprehensive_search_space(self) -> Dict[str, Dict[str, Any]]:
        """Comprehensive search space covering all major parameters"""
        return {
            # Model architecture
            'd_model': {'type': 'categorical', 'choices': [16, 32, 64, 96, 128]},
            'd_ff': {'type': 'categorical', 'choices': [32, 64, 128, 256, 512]},
            'n_heads': {'type': 'categorical', 'choices': [4, 8, 16]},
            'e_layers': {'type': 'categorical', 'choices': [1, 2, 3, 4]},
            'd_layers': {'type': 'categorical', 'choices': [1, 2]},
            'factor': {'type': 'int', 'low': 1, 'high': 3},
            'dropout': {'type': 'float', 'low': 0.0, 'high': 0.3},
            'activation': {'type': 'categorical', 'choices': ['relu', 'gelu', 'tanh']},
            
            # TimeLLM specific
            'llm_model': {'type': 'categorical', 'choices': ['LLAMA', 'GPT2', 'BERT', 'DEEPSEEK', 'QWEN']},
            'llm_layers': {'type': 'int', 'low': 4, 'high': 12},
            'patch_len': {'type': 'categorical', 'choices': [1, 4, 8, 12, 16, 24, 32]},
            'stride': {'type': 'categorical', 'choices': [1, 2, 4, 6, 8, 12, 16]},
            'num_tokens': {'type': 'categorical', 'choices': [500, 1000, 1500, 2000]},
            
            # Training parameters
            'learning_rate': {'type': 'loguniform', 'low': 1e-5, 'high': 1e-2},
            'batch_size': {'type': 'categorical', 'choices': [8, 12, 16, 24, 32, 48, 64]},
            'train_epochs': {'type': 'int', 'low': 5, 'high': 25},
            'lradj': {'type': 'categorical', 'choices': ['type1', 'type2', 'COS', 'TST']},
            'pct_start': {'type': 'float', 'low': 0.1, 'high': 0.4},
            
            # Data configuration
            'seq_len': {'type': 'categorical', 'choices': [96, 168, 192, 256, 336, 512]},
            'pred_len': {'type': 'categorical', 'choices': [24, 48, 96, 192]},
            
            # Loss function
            'loss': {'type': 'categorical', 'choices': ['MSE', 'MAE', 'MADL', 'DLF', 'ASYMMETRIC', 'TRADING_LOSS', 'SHARPE_LOSS', 'ROBUST']},
        }
    
    def _quick_search_space(self) -> Dict[str, Dict[str, Any]]:
        """Reduced search space for quick optimization"""
        return {
            'learning_rate': {'type': 'loguniform', 'low': 1e-4, 'high': 1e-2},
            'batch_size': {'type': 'categorical', 'choices': [16, 32, 64]},
            'd_model': {'type': 'categorical', 'choices': [32, 64, 96]},
            'd_ff': {'type': 'categorical', 'choices': [128, 256, 512]},
            'patch_len': {'type': 'categorical', 'choices': [8, 16, 24]},
            'llm_layers': {'type': 'int', 'low': 4, 'high': 8},
            'loss': {'type': 'categorical', 'choices': ['MAE', 'TRADING_LOSS', 'SHARPE_LOSS']},
        }
    
    def _architecture_search_space(self) -> Dict[str, Dict[str, Any]]:
        """Focus on model architecture parameters"""
        return {
            'd_model': {'type': 'categorical', 'choices': [16, 32, 64, 96, 128, 160, 192]},
            'd_ff': {'type': 'categorical', 'choices': [64, 128, 256, 384, 512, 768]},
            'n_heads': {'type': 'categorical', 'choices': [4, 8, 12, 16]},
            'e_layers': {'type': 'int', 'low': 1, 'high': 6},
            'd_layers': {'type': 'int', 'low': 1, 'high': 3},
            'llm_model': {'type': 'categorical', 'choices': ['LLAMA', 'GPT2', 'BERT', 'DEEPSEEK', 'QWEN']},
            'llm_layers': {'type': 'int', 'low': 2, 'high': 12},
            'patch_len': {'type': 'categorical', 'choices': [1, 2, 4, 8, 12, 16, 24, 32]},
            'stride': {'type': 'categorical', 'choices': [1, 2, 3, 4, 6, 8, 12, 16]},
            'dropout': {'type': 'float', 'low': 0.0, 'high': 0.4},
        }
    
    def _training_search_space(self) -> Dict[str, Dict[str, Any]]:
        """Focus on training parameters"""
        return {
            'learning_rate': {'type': 'loguniform', 'low': 5e-6, 'high': 5e-2},
            'batch_size': {'type': 'categorical', 'choices': [4, 8, 12, 16, 24, 32, 48, 64, 96]},
            'train_epochs': {'type': 'int', 'low': 3, 'high': 30},
            'lradj': {'type': 'categorical', 'choices': ['type1', 'type2', 'type3', 'COS', 'TST']},
            'pct_start': {'type': 'float', 'low': 0.05, 'high': 0.5},
            'loss': {'type': 'categorical', 'choices': ['MSE', 'MAE', 'MAPE', 'MADL', 'DLF', 'ASYMMETRIC', 'QUANTILE', 'TRADING_LOSS', 'SHARPE_LOSS', 'ROBUST']},
            'seq_len': {'type': 'categorical', 'choices': [48, 96, 168, 192, 256, 336, 512, 720]},
            'pred_len': {'type': 'categorical', 'choices': [6, 12, 24, 48, 96, 192]},
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and adjust parameters for consistency"""
        
        # Ensure label_len is reasonable
        if 'seq_len' in params:
            params['label_len'] = min(params['seq_len'] // 2, 48)
        
        # Ensure stride <= patch_len
        if 'stride' in params and 'patch_len' in params:
            params['stride'] = min(params['stride'], params['patch_len'])
        
        # Adjust dimensions for compatibility
        if 'd_model' in params and 'n_heads' in params:
            # Ensure d_model is divisible by n_heads
            if params['d_model'] % params['n_heads'] != 0:
                # Adjust d_model to nearest compatible value
                compatible_d_model = (params['d_model'] // params['n_heads']) * params['n_heads']
                if compatible_d_model == 0:
                    compatible_d_model = params['n_heads']
                params['d_model'] = compatible_d_model
        
        # Set LLM dimensions based on model type
        llm_dims = {
            'LLAMA': 4096,
            'GPT2': 768,
            'BERT': 768,
            'DEEPSEEK': 4096,
            'QWEN': 4096,
            'MISTRAL': 4096,
            'GEMMA': 2048
        }
        
        if 'llm_model' in params:
            params['llm_dim'] = llm_dims.get(params['llm_model'], 4096)
        
        return params
    
    def estimate_gpu_memory(self, params: Dict[str, Any]) -> float:
        """Estimate GPU memory requirements in GB"""
        
        # Base memory estimation formula
        d_model = params.get('d_model', 64)
        seq_len = params.get('seq_len', 96)
        batch_size = params.get('batch_size', 32)
        llm_layers = params.get('llm_layers', 6)
        
        # Rough estimation: model size + activations + gradients
        model_params = d_model * seq_len * llm_layers / 1e6  # Million parameters
        activation_memory = batch_size * seq_len * d_model * 4 / 1e9  # GB
        gradient_memory = model_params * 8 / 1e3  # GB (2x for gradients)
        
        total_memory = (model_params * 4 / 1e3) + activation_memory + gradient_memory
        
        return max(total_memory, 1.0)  # Minimum 1GB


class OptunaTuner:
    """Optuna-based hyperparameter optimization"""
    
    def __init__(self, config: OptimizationConfig, search_space: HyperparameterSpace):
        self.config = config
        self.search_space = search_space
        self.study = None
        
    def create_study(self):
        """Create Optuna study with appropriate sampler and pruner"""
        
        # Choose sampler
        if self.config.n_trials < 50:
            sampler = TPESampler()
        else:
            sampler = CmaEsSampler()
        
        # Choose pruner
        if self.config.enable_pruning:
            pruner = HyperbandPruner(min_resource=self.config.min_epochs, reduction_factor=2)
        else:
            pruner = MedianPruner()
        
        # Create study
        study = optuna.create_study(
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
            study_name=self.config.study_name
        )
        
        return study
    
    def suggest_parameters(self, trial) -> Dict[str, Any]:
        """Suggest parameters for a trial"""
        
        params = {}
        
        for param_name, param_config in self.search_space.space.items():
            param_type = param_config['type']
            
            if param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
            elif param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
            elif param_type == 'loguniform':
                params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
        
        # Validate parameters
        params = self.search_space.validate_parameters(params)
        
        return params
    
    def objective(self, trial) -> float:
        """Objective function for Optuna optimization"""
        
        # Suggest parameters
        params = self.suggest_parameters(trial)
        
        # Check GPU memory constraints
        estimated_memory = self.search_space.estimate_gpu_memory(params)
        if estimated_memory > self.config.gpu_memory_limit:
            raise optuna.TrialPruned()
        
        # Run experiment
        experiment_runner = ExperimentRunner(self.config)
        result = experiment_runner.run_trial(trial.number, params)
        
        # Report intermediate results for pruning
        if self.config.enable_pruning and result.status == "pruned":
            raise optuna.TrialPruned()
        
        if result.status == "failed":
            raise optuna.TrialPruned()
        
        # Return primary metric
        return result.metrics[self.config.primary_metric]


class ExperimentRunner:
    """Runs individual experiments for hyperparameter optimization"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
    def run_trial(self, trial_id: int, params: Dict[str, Any]) -> TrialResult:
        """Run a single hyperparameter optimization trial"""
        
        start_time = time.time()
        
        try:
            # Generate experiment configuration
            exp_config = self._generate_experiment_config(trial_id, params)
            
            # Run training
            metrics = self._run_training_experiment(exp_config)
            
            training_time = time.time() - start_time
            
            return TrialResult(
                trial_id=trial_id,
                parameters=params,
                metrics=metrics,
                training_time=training_time,
                epochs_trained=params.get('train_epochs', 10),
                status="completed"
            )
            
        except TimeoutError:
            return TrialResult(
                trial_id=trial_id,
                parameters=params,
                metrics={},
                training_time=time.time() - start_time,
                epochs_trained=0,
                status="pruned",
                error_message="Training timeout"
            )
            
        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {e}")
            return TrialResult(
                trial_id=trial_id,
                parameters=params,
                metrics={},
                training_time=time.time() - start_time,
                epochs_trained=0,
                status="failed",
                error_message=str(e)
            )
    
    def _generate_experiment_config(self, trial_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete experiment configuration"""
        
        # Base configuration
        config = {
            'task_name': 'long_term_forecast',
            'is_training': 1,
            'model_id': f'HPO_trial_{trial_id}',
            'model': 'TimeLLM',
            'data': self.config.datasets[0] if self.config.datasets else 'CRYPTEX_ENHANCED',
            'root_path': './dataset/cryptex/',
            'data_path': 'candlesticks-h.csv',
            'features': 'M',
            'target': 'close',
            'freq': 'h',
            'checkpoints': os.path.join(self.config.output_dir, 'checkpoints'),
            'prompt_domain': 1,
            'content': '',
            'percent': 100,
            'num_workers': 4,
            'use_amp': False,
            'itr': 1,
        }
        
        # Add hyperparameters
        config.update(params)
        
        # Set dataset-specific parameters
        if config['data'] == 'CRYPTEX_ENHANCED':
            config.update({'enc_in': 68, 'dec_in': 68, 'c_out': 68})
        elif config['data'] == 'CRYPTEX_EXTERNAL':
            config.update({'enc_in': 100, 'dec_in': 100, 'c_out': 100})
        elif config['data'] == 'CRYPTEX_MULTISCALE':
            config.update({'enc_in': 80, 'dec_in': 80, 'c_out': 80})
        else:
            config.update({'enc_in': 6, 'dec_in': 6, 'c_out': 6})
        
        return config
    
    def _run_training_experiment(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Run training experiment and extract metrics"""
        
        # Create command line arguments
        cmd = self._build_training_command(config)
        
        # Run training with timeout
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.config.max_training_time
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Training failed: {result.stderr}")
            
            # Parse metrics from output
            metrics = self._parse_training_metrics(result.stdout, result.stderr)
            
            return metrics
            
        except subprocess.TimeoutExpired:
            raise TimeoutError("Training exceeded maximum time limit")
    
    def _build_training_command(self, config: Dict[str, Any]) -> str:
        """Build training command from configuration"""
        
        cmd = "python run_main.py"
        
        for key, value in config.items():
            if key not in ['checkpoints']:  # Skip complex arguments
                cmd += f" --{key} {value}"
        
        return cmd
    
    def _parse_training_metrics(self, stdout: str, stderr: str) -> Dict[str, float]:
        """Parse metrics from training output"""
        
        metrics = {}
        
        # Look for final metrics in output
        lines = stdout.split('\n') + stderr.split('\n')
        
        for line in lines:
            # Look for metric patterns
            if 'mae:' in line.lower():
                try:
                    mae_val = float(line.split('mae:')[1].split()[0].strip(','))
                    metrics['mae'] = mae_val
                except:
                    pass
            
            if 'mse:' in line.lower():
                try:
                    mse_val = float(line.split('mse:')[1].split()[0].strip(','))
                    metrics['mse'] = mse_val
                except:
                    pass
            
            if 'mape:' in line.lower():
                try:
                    mape_val = float(line.split('mape:')[1].split()[0].strip(','))
                    metrics['mape'] = mape_val
                except:
                    pass
        
        # Set default values if metrics not found
        if not metrics:
            logger.warning("No metrics found in training output, using defaults")
            metrics = {'mae': 999.0, 'mse': 999.0, 'mape': 999.0}
        
        return metrics


class HyperparameterOptimizer:
    """Main hyperparameter optimization orchestrator"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.results = []
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for optimization"""
        log_file = os.path.join(self.config.output_dir, f"{self.config.study_name}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
    
    def optimize(self, 
                optimization_type: str = "comprehensive",
                datasets: List[str] = None,
                n_trials: int = None) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        
        logger.info(f"Starting hyperparameter optimization: {optimization_type}")
        
        # Update configuration
        if datasets:
            self.config.datasets = datasets
        if n_trials:
            self.config.n_trials = n_trials
        
        # Create search space
        search_space = HyperparameterSpace(optimization_type)
        
        # Choose optimization method
        if self.config.method == "optuna" and OPTUNA_AVAILABLE:
            results = self._run_optuna_optimization(search_space)
        elif self.config.method == "ray" and RAY_AVAILABLE:
            results = self._run_ray_optimization(search_space)
        elif self.config.method == "grid":
            results = self._run_grid_search(search_space)
        else:
            results = self._run_random_search(search_space)
        
        # Save results
        self._save_optimization_results(results)
        
        return results
    
    def _run_optuna_optimization(self, search_space: HyperparameterSpace) -> Dict[str, Any]:
        """Run Optuna-based optimization"""
        
        logger.info("Running Optuna optimization")
        
        tuner = OptunaTuner(self.config, search_space)
        study = tuner.create_study()
        
        # Run optimization
        study.optimize(tuner.objective, n_trials=self.config.n_trials, timeout=self.config.timeout)
        
        # Extract results
        best_trial = study.best_trial
        
        results = {
            'method': 'optuna',
            'best_parameters': best_trial.params,
            'best_value': best_trial.value,
            'n_trials': len(study.trials),
            'study': study,
            'optimization_history': [(t.number, t.value) for t in study.trials if t.value is not None]
        }
        
        logger.info(f"Optuna optimization completed. Best value: {best_trial.value}")
        
        return results
    
    def _run_random_search(self, search_space: HyperparameterSpace) -> Dict[str, Any]:
        """Run random search optimization"""
        
        logger.info("Running random search optimization")
        
        best_params = None
        best_value = float('inf') if self.config.direction == "minimize" else float('-inf')
        all_results = []
        
        experiment_runner = ExperimentRunner(self.config)
        
        for trial_id in range(self.config.n_trials):
            # Generate random parameters
            params = self._generate_random_parameters(search_space)
            
            # Run trial
            result = experiment_runner.run_trial(trial_id, params)
            all_results.append(result)
            
            # Update best result
            if result.status == "completed":
                metric_value = result.metrics.get(self.config.primary_metric, float('inf'))
                
                is_better = (
                    (self.config.direction == "minimize" and metric_value < best_value) or
                    (self.config.direction == "maximize" and metric_value > best_value)
                )
                
                if is_better:
                    best_value = metric_value
                    best_params = params.copy()
            
            logger.info(f"Trial {trial_id}/{self.config.n_trials} completed")
        
        results = {
            'method': 'random_search',
            'best_parameters': best_params,
            'best_value': best_value,
            'n_trials': self.config.n_trials,
            'all_results': all_results,
            'optimization_history': [(i, r.metrics.get(self.config.primary_metric, float('inf'))) 
                                   for i, r in enumerate(all_results) if r.status == "completed"]
        }
        
        logger.info(f"Random search completed. Best value: {best_value}")
        
        return results
    
    def _generate_random_parameters(self, search_space: HyperparameterSpace) -> Dict[str, Any]:
        """Generate random parameters from search space"""
        
        params = {}
        
        for param_name, param_config in search_space.space.items():
            param_type = param_config['type']
            
            if param_type == 'categorical':
                params[param_name] = np.random.choice(param_config['choices'])
            elif param_type == 'int':
                params[param_name] = np.random.randint(param_config['low'], param_config['high'] + 1)
            elif param_type == 'float':
                params[param_name] = np.random.uniform(param_config['low'], param_config['high'])
            elif param_type == 'loguniform':
                log_low = np.log(param_config['low'])
                log_high = np.log(param_config['high'])
                params[param_name] = np.exp(np.random.uniform(log_low, log_high))
        
        # Validate parameters
        params = search_space.validate_parameters(params)
        
        return params
    
    def _save_optimization_results(self, results: Dict[str, Any]):
        """Save optimization results to files"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main results
        results_file = os.path.join(
            self.config.output_dir, 
            f"{self.config.study_name}_results_{timestamp}.json"
        )
        
        # Prepare results for JSON serialization
        json_results = results.copy()
        if 'study' in json_results:
            del json_results['study']  # Can't serialize Optuna study
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Generate summary report
        report_file = os.path.join(
            self.config.output_dir,
            f"{self.config.study_name}_report_{timestamp}.txt"
        )
        
        with open(report_file, 'w') as f:
            f.write(self._generate_optimization_report(results))
        
        logger.info(f"Results saved to {results_file}")
        logger.info(f"Report saved to {report_file}")
    
    def _generate_optimization_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable optimization report"""
        
        report = []
        report.append("=" * 60)
        report.append("HYPERPARAMETER OPTIMIZATION REPORT")
        report.append("=" * 60)
        report.append(f"Study Name: {self.config.study_name}")
        report.append(f"Optimization Method: {results['method']}")
        report.append(f"Total Trials: {results['n_trials']}")
        report.append(f"Primary Metric: {self.config.primary_metric}")
        report.append(f"Direction: {self.config.direction}")
        
        report.append("\nBEST PARAMETERS:")
        best_params = results.get('best_parameters', {})
        for param, value in best_params.items():
            report.append(f"  {param}: {value}")
        
        report.append(f"\nBEST VALUE: {results.get('best_value', 'N/A')}")
        
        # Optimization history
        if 'optimization_history' in results:
            history = results['optimization_history']
            if history:
                report.append("\nOPTIMIZATION PROGRESS:")
                report.append("Trial\tValue")
                for trial_num, value in history[-10:]:  # Last 10 trials
                    report.append(f"{trial_num}\t{value:.6f}")
        
        # Parameter importance (if available)
        if 'study' in results and hasattr(results['study'], 'trials'):
            try:
                import optuna.importance
                importance = optuna.importance.get_param_importances(results['study'])
                
                report.append("\nPARAMETER IMPORTANCE:")
                for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
                    report.append(f"  {param}: {imp:.4f}")
            except:
                pass
        
        report.append("\nRECOMMENDATIONS:")
        report.append("1. Use the best parameters for final model training")
        report.append("2. Consider ensemble methods with top-performing configurations")
        report.append("3. Analyze parameter importance for future optimization")
        report.append("4. Validate results with extended training on best configuration")
        
        return "\n".join(report)


# Convenience functions
def quick_optimize(datasets: List[str] = None, n_trials: int = 20) -> Dict[str, Any]:
    """Quick hyperparameter optimization for immediate results"""
    
    config = OptimizationConfig(
        method="random",
        n_trials=n_trials,
        primary_metric="mae",
        datasets=datasets or ["CRYPTEX_ENHANCED"],
        study_name="Quick_HPO"
    )
    
    optimizer = HyperparameterOptimizer(config)
    return optimizer.optimize("quick")


def comprehensive_optimize(datasets: List[str] = None, n_trials: int = 100) -> Dict[str, Any]:
    """Comprehensive hyperparameter optimization"""
    
    config = OptimizationConfig(
        method="optuna" if OPTUNA_AVAILABLE else "random",
        n_trials=n_trials,
        primary_metric="mae",
        secondary_metrics=["mse", "mape"],
        datasets=datasets or ["CRYPTEX_ENHANCED", "CRYPTEX_EXTERNAL"],
        study_name="Comprehensive_HPO",
        enable_pruning=True
    )
    
    optimizer = HyperparameterOptimizer(config)
    return optimizer.optimize("comprehensive")


def architecture_search(datasets: List[str] = None, n_trials: int = 50) -> Dict[str, Any]:
    """Focus on model architecture optimization"""
    
    config = OptimizationConfig(
        method="optuna" if OPTUNA_AVAILABLE else "random",
        n_trials=n_trials,
        primary_metric="mae",
        datasets=datasets or ["CRYPTEX_ENHANCED"],
        study_name="Architecture_Search"
    )
    
    optimizer = HyperparameterOptimizer(config)
    return optimizer.optimize("architecture")


if __name__ == "__main__":
    print("TimeLLM Hyperparameter Optimization Framework")
    print(f"Optuna available: {OPTUNA_AVAILABLE}")
    print(f"Ray Tune available: {RAY_AVAILABLE}")
    
    # Example usage
    print("\nRunning quick optimization example...")
    results = quick_optimize(n_trials=5)
    print(f"Quick optimization completed. Best value: {results.get('best_value')}")