#!/usr/bin/env python3
"""
Advanced Learning Rate Schedulers and Training Optimization for TimeLLM

This module provides sophisticated learning rate scheduling strategies,
ensemble weighting optimization, and advanced training techniques specifically
designed for cryptocurrency time series prediction.
"""

import os
import sys
import numpy as np
import pandas as pd
import math
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')

# Optional dependencies for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import _LRScheduler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for type hints
    class _LRScheduler:
        def __init__(self, optimizer, last_epoch=-1):
            self.optimizer = optimizer
            self.last_epoch = last_epoch
            self.base_lrs = [0.001]
        
        def get_lr(self):
            return [0.001]
        
        def step(self, epoch=None):
            pass


@dataclass
class SchedulerConfig:
    """Configuration for learning rate schedulers"""
    
    # Scheduler type
    scheduler_type: str = "cosine_annealing_warm_restarts"  # cosine, exponential, polynomial, adaptive, custom
    
    # Base parameters
    initial_lr: float = 0.001
    min_lr: float = 1e-6
    max_lr: float = 0.01
    
    # Warmup parameters
    warmup_epochs: int = 5
    warmup_factor: float = 0.1
    
    # Cosine annealing parameters
    T_0: int = 10  # Initial restart period
    T_mult: int = 2  # Restart period multiplier
    eta_min: float = 1e-6
    
    # Exponential decay parameters
    gamma: float = 0.95
    step_size: int = 5
    
    # Polynomial decay parameters
    power: float = 1.0
    total_epochs: int = 50
    
    # Adaptive parameters
    patience: int = 5
    factor: float = 0.5
    threshold: float = 1e-4
    
    # Performance-based parameters
    metric_based: bool = True
    target_metric: str = "mae"
    improvement_threshold: float = 0.01
    
    # Multi-phase training
    phases: List[Dict[str, Any]] = field(default_factory=list)


class WarmupScheduler(_LRScheduler):
    """Learning rate scheduler with warmup period"""
    
    def __init__(self, optimizer, warmup_epochs: int, warmup_factor: float = 0.1, 
                 base_scheduler: Optional[_LRScheduler] = None, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.base_scheduler = base_scheduler
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linear increase from warmup_factor to 1.0
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Use base scheduler if provided
            if self.base_scheduler is not None:
                return self.base_scheduler.get_lr()
            else:
                return [base_lr for base_lr in self.base_lrs]
    
    def step(self, metrics=None):
        super().step()
        if self.base_scheduler is not None and self.last_epoch >= self.warmup_epochs:
            self.base_scheduler.step(metrics)


class CosineAnnealingWarmRestartsAdvanced(_LRScheduler):
    """Advanced Cosine Annealing with Warm Restarts"""
    
    def __init__(self, optimizer, T_0: int, T_mult: int = 1, eta_min: float = 0,
                 restart_decay: float = 1.0, cycle_momentum: bool = True, last_epoch: int = -1):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.restart_decay = restart_decay
        self.cycle_momentum = cycle_momentum
        self.T_cur = 0
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [self.eta_min + (base_lr * (self.restart_decay ** self._get_restart_count()) - self.eta_min) *
                (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.T_cur = self.T_cur + 1
        self.last_epoch = epoch
        
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
    
    def _get_restart_count(self):
        """Calculate number of restarts that have occurred"""
        n = 0
        t = self.T_0
        while self.last_epoch >= t:
            n += 1
            t += self.T_0 * (self.T_mult ** n)
        return n


class AdaptiveScheduler(_LRScheduler):
    """Adaptive learning rate scheduler based on performance metrics"""
    
    def __init__(self, optimizer, patience: int = 5, factor: float = 0.5,
                 threshold: float = 1e-4, min_lr: float = 1e-6, 
                 mode: str = 'min', last_epoch: int = -1):
        self.patience = patience
        self.factor = factor
        self.threshold = threshold
        self.min_lr = min_lr
        self.mode = mode
        self.best_metric = None
        self.num_bad_epochs = 0
        self.last_epoch_improved = 0
        super().__init__(optimizer, last_epoch)
    
    def step(self, metrics, epoch=None):
        current_metric = metrics
        
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        
        if self.best_metric is None:
            self.best_metric = current_metric
        
        if self._is_improvement(current_metric):
            self.best_metric = current_metric
            self.num_bad_epochs = 0
            self.last_epoch_improved = epoch
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
    
    def _is_improvement(self, current_metric):
        if self.mode == 'min':
            return current_metric < self.best_metric - self.threshold
        else:
            return current_metric > self.best_metric + self.threshold
    
    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr != new_lr:
                param_group['lr'] = new_lr
                print(f"Reducing learning rate from {old_lr:.2e} to {new_lr:.2e}")
    
    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class CyclicalLearningRate(_LRScheduler):
    """Cyclical Learning Rate (CLR) scheduler"""
    
    def __init__(self, optimizer, base_lr: float, max_lr: float, step_size: int,
                 mode: str = 'triangular', gamma: float = 1.0, scale_fn=None, 
                 scale_mode: str = 'cycle', last_epoch: int = -1):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        cycle = math.floor(1 + self.last_epoch / (2 * self.step_size))
        x = abs(self.last_epoch / self.step_size - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
        elif self.mode == 'triangular2':
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * (self.gamma ** self.last_epoch)
        else:
            lr = self.base_lr
        
        return [lr for _ in self.base_lrs]


class PerformanceBasedScheduler(_LRScheduler):
    """Performance-based learning rate scheduler with multiple strategies"""
    
    def __init__(self, optimizer, config: SchedulerConfig):
        self.config = config
        self.performance_history = []
        self.lr_history = []
        self.epoch_count = 0
        super().__init__(optimizer, -1)
    
    def step(self, metrics: Dict[str, float], epoch: int = None):
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        self.epoch_count += 1
        
        # Record performance
        target_metric = metrics.get(self.config.target_metric, float('inf'))
        self.performance_history.append(target_metric)
        
        # Calculate new learning rate
        new_lr = self._calculate_lr(metrics)
        
        # Apply learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        self.lr_history.append(new_lr)
    
    def _calculate_lr(self, metrics: Dict[str, float]) -> float:
        """Calculate learning rate based on performance metrics"""
        
        current_metric = metrics.get(self.config.target_metric, float('inf'))
        
        # Get recent performance trend
        if len(self.performance_history) < 3:
            return self.config.initial_lr
        
        recent_performance = self.performance_history[-3:]
        trend = self._calculate_trend(recent_performance)
        
        # Base learning rate calculation
        base_lr = self._get_base_lr()
        
        # Adjust based on trend
        if trend > 0:  # Improving
            adjustment = 1.0 + min(trend * 0.1, 0.2)  # Increase by up to 20%
        else:  # Degrading
            adjustment = 1.0 + max(trend * 0.2, -0.5)  # Decrease by up to 50%
        
        # Apply adjustment
        new_lr = base_lr * adjustment
        
        # Ensure bounds
        new_lr = max(min(new_lr, self.config.max_lr), self.config.min_lr)
        
        return new_lr
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate performance trend (-1 to 1)"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope (normalized)
        if len(values) > 1:
            slope = (y[-1] - y[0]) / (len(values) - 1)
            # Normalize by average value to get relative change
            avg_value = np.mean(y)
            if avg_value != 0:
                normalized_slope = -slope / avg_value  # Negative because lower is better for loss
                return max(min(normalized_slope, 1.0), -1.0)
        
        return 0.0
    
    def _get_base_lr(self) -> float:
        """Get base learning rate for current epoch"""
        return self.config.initial_lr
    
    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class MultiPhaseScheduler(_LRScheduler):
    """Multi-phase training scheduler with different strategies per phase"""
    
    def __init__(self, optimizer, phases: List[Dict[str, Any]], last_epoch: int = -1):
        self.phases = phases
        self.current_phase = 0
        self.phase_epoch = 0
        self.phase_schedulers = []
        
        # Initialize schedulers for each phase
        for phase in phases:
            scheduler = self._create_phase_scheduler(optimizer, phase)
            self.phase_schedulers.append(scheduler)
        
        super().__init__(optimizer, last_epoch)
    
    def _create_phase_scheduler(self, optimizer, phase_config: Dict[str, Any]):
        """Create scheduler for a specific phase"""
        
        scheduler_type = phase_config.get('type', 'constant')
        
        if scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=phase_config.get('epochs', 10),
                eta_min=phase_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'exponential':
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=phase_config.get('gamma', 0.95)
            )
        elif scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=phase_config.get('step_size', 5),
                gamma=phase_config.get('gamma', 0.5)
            )
        else:
            # Constant learning rate
            return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    
    def step(self, metrics=None):
        if self.current_phase < len(self.phases):
            current_phase_config = self.phases[self.current_phase]
            phase_epochs = current_phase_config.get('epochs', 10)
            
            # Check if current phase is complete
            if self.phase_epoch >= phase_epochs:
                self.current_phase += 1
                self.phase_epoch = 0
                
                # Update learning rate for new phase
                if self.current_phase < len(self.phases):
                    new_lr = self.phases[self.current_phase].get('lr', 0.001)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
            
            # Step current phase scheduler
            if self.current_phase < len(self.phase_schedulers):
                self.phase_schedulers[self.current_phase].step()
            
            self.phase_epoch += 1
        
        super().step()
    
    def get_lr(self):
        if self.current_phase < len(self.phase_schedulers):
            return self.phase_schedulers[self.current_phase].get_lr()
        else:
            return [param_group['lr'] for param_group in self.optimizer.param_groups]


class EnsembleWeightOptimizer:
    """Optimize ensemble weights for multiple TimeLLM models"""
    
    def __init__(self, models: List[str], validation_data: pd.DataFrame):
        self.models = models
        self.validation_data = validation_data
        self.weights = np.ones(len(models)) / len(models)  # Initialize equally
        self.performance_history = []
    
    def optimize_weights(self, 
                        predictions: Dict[str, np.ndarray],
                        actual: np.ndarray,
                        method: str = "differential_evolution") -> np.ndarray:
        """Optimize ensemble weights using specified method"""
        
        if method == "differential_evolution":
            return self._optimize_de(predictions, actual)
        elif method == "bayesian":
            return self._optimize_bayesian(predictions, actual)
        elif method == "genetic":
            return self._optimize_genetic(predictions, actual)
        else:
            return self._optimize_grid_search(predictions, actual)
    
    def _optimize_de(self, predictions: Dict[str, np.ndarray], actual: np.ndarray) -> np.ndarray:
        """Optimize weights using Differential Evolution"""
        
        try:
            from scipy.optimize import differential_evolution
        except ImportError:
            print("SciPy not available, falling back to grid search")
            return self._optimize_grid_search(predictions, actual)
        
        pred_matrix = np.array([predictions[model] for model in self.models]).T
        
        def objective(weights):
            weights = weights / np.sum(weights)  # Normalize
            ensemble_pred = np.dot(pred_matrix, weights)
            return np.mean((ensemble_pred - actual) ** 2)  # MSE
        
        # Constraints: weights sum to 1 and are non-negative
        bounds = [(0, 1) for _ in range(len(self.models))]
        
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=100,
            atol=1e-6
        )
        
        optimal_weights = result.x / np.sum(result.x)
        return optimal_weights
    
    def _optimize_grid_search(self, predictions: Dict[str, np.ndarray], actual: np.ndarray) -> np.ndarray:
        """Optimize weights using grid search"""
        
        pred_matrix = np.array([predictions[model] for model in self.models]).T
        
        best_weights = None
        best_score = float('inf')
        
        # Grid search over weight combinations
        if len(self.models) == 2:
            for w1 in np.arange(0, 1.1, 0.1):
                w2 = 1 - w1
                weights = np.array([w1, w2])
                
                ensemble_pred = np.dot(pred_matrix, weights)
                score = np.mean((ensemble_pred - actual) ** 2)
                
                if score < best_score:
                    best_score = score
                    best_weights = weights
        
        elif len(self.models) == 3:
            for w1 in np.arange(0, 1.1, 0.1):
                for w2 in np.arange(0, 1.1 - w1, 0.1):
                    w3 = 1 - w1 - w2
                    if w3 >= 0:
                        weights = np.array([w1, w2, w3])
                        
                        ensemble_pred = np.dot(pred_matrix, weights)
                        score = np.mean((ensemble_pred - actual) ** 2)
                        
                        if score < best_score:
                            best_score = score
                            best_weights = weights
        
        else:
            # For more models, use random sampling
            for _ in range(1000):
                weights = np.random.dirichlet(np.ones(len(self.models)))
                
                ensemble_pred = np.dot(pred_matrix, weights)
                score = np.mean((ensemble_pred - actual) ** 2)
                
                if score < best_score:
                    best_score = score
                    best_weights = weights
        
        return best_weights if best_weights is not None else np.ones(len(self.models)) / len(self.models)
    
    def calculate_ensemble_predictions(self, 
                                     predictions: Dict[str, np.ndarray],
                                     weights: np.ndarray = None) -> np.ndarray:
        """Calculate weighted ensemble predictions"""
        
        if weights is None:
            weights = self.weights
        
        pred_matrix = np.array([predictions[model] for model in self.models]).T
        return np.dot(pred_matrix, weights)
    
    def evaluate_ensemble_performance(self,
                                    predictions: Dict[str, np.ndarray],
                                    actual: np.ndarray,
                                    weights: np.ndarray = None) -> Dict[str, float]:
        """Evaluate ensemble performance with given weights"""
        
        ensemble_pred = self.calculate_ensemble_predictions(predictions, weights)
        
        mse = np.mean((ensemble_pred - actual) ** 2)
        mae = np.mean(np.abs(ensemble_pred - actual))
        mape = np.mean(np.abs((actual - ensemble_pred) / actual)) * 100
        
        # Trading metrics
        returns = np.diff(actual) / actual[:-1]
        pred_direction = np.sign(np.diff(ensemble_pred))
        actual_direction = np.sign(returns)
        
        directional_accuracy = np.mean(pred_direction == actual_direction) if len(pred_direction) > 0 else 0
        
        return {
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }


class SchedulerFactory:
    """Factory for creating learning rate schedulers"""
    
    @staticmethod
    def create_scheduler(optimizer, config: SchedulerConfig) -> _LRScheduler:
        """Create learning rate scheduler based on configuration"""
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for scheduler creation")
        
        scheduler_type = config.scheduler_type.lower()
        
        if scheduler_type == "cosine_annealing_warm_restarts":
            scheduler = CosineAnnealingWarmRestartsAdvanced(
                optimizer,
                T_0=config.T_0,
                T_mult=config.T_mult,
                eta_min=config.eta_min
            )
        
        elif scheduler_type == "adaptive":
            scheduler = AdaptiveScheduler(
                optimizer,
                patience=config.patience,
                factor=config.factor,
                threshold=config.threshold,
                min_lr=config.min_lr
            )
        
        elif scheduler_type == "cyclical":
            scheduler = CyclicalLearningRate(
                optimizer,
                base_lr=config.min_lr,
                max_lr=config.max_lr,
                step_size=config.step_size
            )
        
        elif scheduler_type == "performance_based":
            scheduler = PerformanceBasedScheduler(optimizer, config)
        
        elif scheduler_type == "multi_phase":
            scheduler = MultiPhaseScheduler(optimizer, config.phases)
        
        else:
            # Default to cosine annealing
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.total_epochs,
                eta_min=config.eta_min
            )
        
        # Add warmup if specified
        if config.warmup_epochs > 0:
            scheduler = WarmupScheduler(
                optimizer,
                warmup_epochs=config.warmup_epochs,
                warmup_factor=config.warmup_factor,
                base_scheduler=scheduler
            )
        
        return scheduler
    
    @staticmethod
    def create_multi_phase_config(phases: List[Dict[str, Any]]) -> SchedulerConfig:
        """Create configuration for multi-phase training"""
        
        return SchedulerConfig(
            scheduler_type="multi_phase",
            phases=phases
        )


# Convenience functions
def create_crypto_trading_scheduler(optimizer, total_epochs: int = 50) -> _LRScheduler:
    """Create optimized scheduler for cryptocurrency trading"""
    
    config = SchedulerConfig(
        scheduler_type="cosine_annealing_warm_restarts",
        initial_lr=0.001,
        min_lr=1e-6,
        max_lr=0.005,
        warmup_epochs=3,
        T_0=7,  # Weekly cycles
        T_mult=2,
        total_epochs=total_epochs
    )
    
    return SchedulerFactory.create_scheduler(optimizer, config)


def create_adaptive_crypto_scheduler(optimizer) -> _LRScheduler:
    """Create adaptive scheduler that responds to validation performance"""
    
    config = SchedulerConfig(
        scheduler_type="performance_based",
        initial_lr=0.001,
        min_lr=1e-6,
        max_lr=0.01,
        target_metric="mae",
        improvement_threshold=0.01
    )
    
    return SchedulerFactory.create_scheduler(optimizer, config)


def create_multi_phase_crypto_scheduler(optimizer) -> _LRScheduler:
    """Create multi-phase scheduler for cryptocurrency prediction"""
    
    phases = [
        {'type': 'constant', 'lr': 0.0005, 'epochs': 5},    # Warmup
        {'type': 'cosine', 'lr': 0.001, 'epochs': 15},     # Main training
        {'type': 'exponential', 'lr': 0.0005, 'epochs': 10, 'gamma': 0.95},  # Fine-tuning
        {'type': 'step', 'lr': 0.0001, 'epochs': 10, 'step_size': 3, 'gamma': 0.5}  # Final refinement
    ]
    
    config = SchedulerFactory.create_multi_phase_config(phases)
    return SchedulerFactory.create_scheduler(optimizer, config)


if __name__ == "__main__":
    print("Advanced Learning Rate Schedulers for TimeLLM")
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    print(f"Plotting available: {PLOTTING_AVAILABLE}")
    
    if TORCH_AVAILABLE:
        # Example usage
        print("\nCreating example schedulers...")
        
        # Create dummy optimizer
        import torch.nn as nn
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create different schedulers
        crypto_scheduler = create_crypto_trading_scheduler(optimizer)
        adaptive_scheduler = create_adaptive_crypto_scheduler(optimizer)
        
        print("Schedulers created successfully!")
    else:
        print("PyTorch not available - schedulers cannot be instantiated")