import torch
import torch.nn as nn
import numpy as np
import json
from typing import Dict, List, Optional, Union, Callable
from collections import defaultdict, deque
import warnings
from utils.metrics import get_loss_function

warnings.filterwarnings('ignore')


class AdaptiveLossFunction(nn.Module):
    """
    Adaptive loss function that combines multiple losses with dynamic weighting.
    
    Supports multiple adaptation strategies:
    - Performance-based: Adjust weights based on validation performance
    - Learning-based: Learn optimal weights as parameters
    - Schedule-based: Follow predefined weight schedules
    - Hybrid: Combination of above strategies
    """
    
    def __init__(self, 
                 loss_functions: Dict[str, str],
                 initial_weights: Optional[Dict[str, float]] = None,
                 adaptation_strategy: str = 'performance_based',
                 adaptation_frequency: int = 100,
                 performance_window: int = 10,
                 learning_rate: float = 0.01,
                 min_weight: float = 0.0,
                 max_weight: float = 1.0):
        """
        Args:
            loss_functions: Dict of {name: loss_type} e.g., {'mse': 'MSE', 'dlf': 'DLF'}
            initial_weights: Initial weights for each loss function
            adaptation_strategy: 'performance_based', 'learning_based', 'schedule_based', 'hybrid'
            adaptation_frequency: How often to adapt weights (in steps)
            performance_window: Window size for performance evaluation
            learning_rate: Learning rate for weight updates
            min_weight: Minimum weight value
            max_weight: Maximum weight value
        """
        super().__init__()
        
        self.loss_functions = {}
        self.loss_names = list(loss_functions.keys())
        
        # Initialize loss functions
        for name, loss_type in loss_functions.items():
            self.loss_functions[name] = get_loss_function(loss_type)
        
        # Initialize weights
        if initial_weights is None:
            # Equal weighting initially
            initial_weights = {name: 1.0 / len(loss_functions) for name in self.loss_names}
        
        # Normalize initial weights
        total_weight = sum(initial_weights.values())
        initial_weights = {k: v / total_weight for k, v in initial_weights.items()}
        
        # Store weights as learnable parameters if learning-based
        if adaptation_strategy in ['learning_based', 'hybrid']:
            # Use softmax to ensure weights sum to 1
            initial_logits = torch.log(torch.tensor([initial_weights[name] for name in self.loss_names]))
            self.weight_logits = nn.Parameter(initial_logits)
        else:
            # Store as regular tensors
            weight_values = torch.tensor([initial_weights[name] for name in self.loss_names])
            self.register_buffer('weights', weight_values)
        
        # Configuration
        self.adaptation_strategy = adaptation_strategy
        self.adaptation_frequency = adaptation_frequency
        self.performance_window = performance_window
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Performance tracking
        self.step_count = 0
        self.loss_history = defaultdict(lambda: deque(maxlen=performance_window))
        self.performance_metrics = defaultdict(list)
        self.weight_history = []
        
        # Adaptation state
        self.last_adaptation_step = 0
        self.best_performance = float('inf')
        self.adaptation_momentum = defaultdict(float)
        
        print(f"Initialized AdaptiveLossFunction with {len(self.loss_functions)} losses")
        print(f"Strategy: {adaptation_strategy}, Initial weights: {initial_weights}")
    
    def get_current_weights(self) -> torch.Tensor:
        """Get current loss weights"""
        if hasattr(self, 'weight_logits'):
            # Learnable weights (softmax normalization)
            return torch.softmax(self.weight_logits, dim=0)
        else:
            # Fixed/adapted weights
            return self.weights
    
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute weighted combination of losses
        
        Returns:
            Dict containing individual losses, weights, and combined loss
        """
        individual_losses = {}
        weights = self.get_current_weights()
        
        # Compute individual losses
        for i, name in enumerate(self.loss_names):
            try:
                loss_value = self.loss_functions[name](pred, true)
                individual_losses[name] = loss_value
                
                # Track loss history for adaptation
                self.loss_history[name].append(loss_value.item())
                
            except Exception as e:
                print(f"Warning: Loss {name} failed: {e}")
                individual_losses[name] = torch.tensor(0.0, device=pred.device)
        
        # Compute weighted combination
        combined_loss = torch.tensor(0.0, device=pred.device)
        for i, name in enumerate(self.loss_names):
            combined_loss += weights[i] * individual_losses[name]
        
        # Scale down loss for numerical stability in fp16 (model-specific)
        # QWEN needs more aggressive scaling due to gradient instability
        scale_factor = getattr(self, 'scale_factor', 0.1)
        combined_loss = combined_loss * scale_factor
        
        # Update step count
        self.step_count += 1
        
        # Trigger adaptation if needed
        if (self.step_count - self.last_adaptation_step) >= self.adaptation_frequency:
            self._adapt_weights()
            self.last_adaptation_step = self.step_count
        
        # Store weight history
        current_weights = {name: weights[i].item() for i, name in enumerate(self.loss_names)}
        self.weight_history.append({
            'step': self.step_count,
            'weights': current_weights,
            'combined_loss': combined_loss.item()
        })
        
        return {
            'combined_loss': combined_loss,
            'individual_losses': individual_losses,
            'weights': current_weights,
            'adaptation_info': {
                'step': self.step_count,
                'strategy': self.adaptation_strategy,
                'last_adaptation': self.last_adaptation_step
            }
        }
    
    def _adapt_weights(self):
        """Adapt loss weights based on the selected strategy"""
        if self.adaptation_strategy == 'performance_based':
            self._adapt_performance_based()
        elif self.adaptation_strategy == 'schedule_based':
            self._adapt_schedule_based()
        elif self.adaptation_strategy == 'hybrid':
            self._adapt_hybrid()
        # learning_based adaptation happens automatically through gradient descent
    
    def _adapt_performance_based(self):
        """Adapt weights based on individual loss performance"""
        if len(self.loss_history[self.loss_names[0]]) < self.performance_window:
            return  # Not enough history
        
        # Calculate performance metrics for each loss
        loss_performances = {}
        for name in self.loss_names:
            if len(self.loss_history[name]) > 0:
                recent_losses = list(self.loss_history[name])
                
                # Calculate performance metrics
                mean_loss = np.mean(recent_losses)
                loss_trend = recent_losses[-1] - recent_losses[0] if len(recent_losses) > 1 else 0
                loss_stability = np.std(recent_losses)
                
                # Combined performance score (lower is better)
                performance_score = mean_loss + 0.1 * abs(loss_trend) + 0.05 * loss_stability
                loss_performances[name] = performance_score
        
        if not loss_performances:
            return
        
        # Convert performance to weights (inverse relationship)
        # Better performance (lower score) → higher weight
        min_performance = min(loss_performances.values())
        max_performance = max(loss_performances.values())
        
        if max_performance > min_performance:
            # Normalize and invert (better performance gets higher weight)
            new_weights = []
            for name in self.loss_names:
                perf = loss_performances.get(name, max_performance)
                # Invert: lower performance score → higher weight
                inverted_perf = max_performance - perf + min_performance
                new_weights.append(inverted_perf)
            
            # Normalize to sum to 1
            total_weight = sum(new_weights)
            if total_weight > 0:
                new_weights = [w / total_weight for w in new_weights]
                
                # Apply momentum and constraints
                current_weights = self.get_current_weights()
                for i, name in enumerate(self.loss_names):
                    # Momentum update
                    momentum = self.adaptation_momentum.get(name, 0.0)
                    new_weight = 0.7 * current_weights[i].item() + 0.3 * new_weights[i] + 0.1 * momentum
                    
                    # Apply constraints
                    new_weight = max(self.min_weight, min(self.max_weight, new_weight))
                    new_weights[i] = new_weight
                    
                    # Update momentum
                    self.adaptation_momentum[name] = new_weights[i] - current_weights[i].item()
                
                # Renormalize after constraints
                total_weight = sum(new_weights)
                if total_weight > 0:
                    new_weights = [w / total_weight for w in new_weights]
                    
                    # Update weights
                    if not hasattr(self, 'weight_logits'):
                        self.weights = torch.tensor(new_weights, device=self.weights.device)
                    
                    print(f"Adapted weights at step {self.step_count}:")
                    for i, name in enumerate(self.loss_names):
                        print(f"  {name}: {new_weights[i]:.4f} (perf: {loss_performances.get(name, 0):.4f})")
    
    def _adapt_schedule_based(self):
        """Adapt weights based on predefined schedules"""
        # Example schedule: start with MSE, gradually increase DLF weight
        progress = min(1.0, self.step_count / 10000)  # Progress over 10k steps
        
        if 'mse' in self.loss_names and 'dlf' in self.loss_names:
            mse_weight = 1.0 - progress
            dlf_weight = progress
            
            # Normalize
            total = mse_weight + dlf_weight
            weights = [mse_weight / total, dlf_weight / total]
            
            if not hasattr(self, 'weight_logits'):
                self.weights = torch.tensor(weights, device=self.weights.device)
    
    def _adapt_hybrid(self):
        """Hybrid adaptation combining multiple strategies"""
        # Combine performance-based and schedule-based
        self._adapt_performance_based()
        
        # Add schedule influence
        if self.step_count % (self.adaptation_frequency * 5) == 0:
            self._adapt_schedule_based()
    
    def get_performance_summary(self) -> Dict:
        """Get summary of loss performance and adaptation"""
        summary = {
            'total_steps': self.step_count,
            'adaptation_count': self.step_count // self.adaptation_frequency,
            'current_weights': {name: self.get_current_weights()[i].item() 
                              for i, name in enumerate(self.loss_names)},
            'loss_statistics': {}
        }
        
        # Loss statistics
        for name in self.loss_names:
            if self.loss_history[name]:
                losses = list(self.loss_history[name])
                summary['loss_statistics'][name] = {
                    'mean': np.mean(losses),
                    'std': np.std(losses),
                    'min': np.min(losses),
                    'max': np.max(losses),
                    'trend': losses[-1] - losses[0] if len(losses) > 1 else 0
                }
        
        return summary
    
    def save_adaptation_history(self, filepath: str):
        """Save adaptation history for analysis"""
        history_data = {
            'config': {
                'loss_functions': {name: type(loss).__name__ for name, loss in self.loss_functions.items()},
                'adaptation_strategy': self.adaptation_strategy,
                'adaptation_frequency': self.adaptation_frequency,
                'performance_window': self.performance_window
            },
            'weight_history': self.weight_history,
            'performance_summary': self.get_performance_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"Adaptation history saved to: {filepath}")


class LossSelectionManager:
    """
    Manager for selecting and configuring adaptive loss functions
    """
    
    def __init__(self):
        self.predefined_combinations = {
            'basic': {
                'mse': 'MSE',
                'mae': 'MAE'
            },
            'trading_focused': {
                'dlf': 'DLF',
                'trading': 'TRADING_LOSS',
                'sharpe': 'SHARPE_LOSS'
            },
            'robust_prediction': {
                'mse': 'MSE',
                'dlf': 'DLF',
                'robust': 'ROBUST'
            },
            'comprehensive': {
                'mse': 'MSE',
                'dlf': 'DLF',
                'trading': 'TRADING_LOSS',
                'asymmetric': 'ASYMMETRIC'
            },
            'directional_focused': {
                'dlf': 'DLF',
                'madl': 'MADL',
                'asymmetric': 'ASYMMETRIC'
            }
        }
    
    def create_adaptive_loss(self, 
                           combination: Union[str, Dict[str, str]],
                           adaptation_strategy: str = 'performance_based',
                           **kwargs) -> AdaptiveLossFunction:
        """
        Create adaptive loss function with specified combination
        
        Args:
            combination: Either predefined name or custom dict of losses
            adaptation_strategy: Adaptation strategy to use
            **kwargs: Additional parameters for AdaptiveLossFunction
        """
        if isinstance(combination, str):
            if combination not in self.predefined_combinations:
                raise ValueError(f"Unknown combination: {combination}. Available: {list(self.predefined_combinations.keys())}")
            loss_functions = self.predefined_combinations[combination]
        else:
            loss_functions = combination
        
        return AdaptiveLossFunction(
            loss_functions=loss_functions,
            adaptation_strategy=adaptation_strategy,
            **kwargs
        )
    
    def recommend_combination(self, task_type: str, data_characteristics: Dict) -> str:
        """
        Recommend loss combination based on task and data characteristics
        
        Args:
            task_type: 'price_prediction', 'direction_prediction', 'trading_strategy'
            data_characteristics: Dict with 'volatility', 'trend', 'noise_level' etc.
        """
        volatility = data_characteristics.get('volatility', 'medium')
        has_trend = data_characteristics.get('trend', True)
        noise_level = data_characteristics.get('noise_level', 'medium')
        
        if task_type == 'trading_strategy':
            return 'trading_focused'
        elif task_type == 'direction_prediction':
            return 'directional_focused'
        elif volatility == 'high' or noise_level == 'high':
            return 'robust_prediction'
        else:
            return 'comprehensive'
    
    def get_combination_info(self, combination_name: str) -> Dict:
        """Get information about a predefined combination"""
        if combination_name not in self.predefined_combinations:
            return {}
        
        combination = self.predefined_combinations[combination_name]
        
        info = {
            'name': combination_name,
            'losses': combination,
            'description': self._get_combination_description(combination_name),
            'recommended_for': self._get_combination_use_cases(combination_name)
        }
        
        return info
    
    def _get_combination_description(self, name: str) -> str:
        descriptions = {
            'basic': 'Simple combination of MSE and MAE for general regression',
            'trading_focused': 'Optimized for trading performance with directional and profit focus',
            'robust_prediction': 'Robust to outliers and noise with directional awareness',
            'comprehensive': 'Balanced approach covering multiple aspects of prediction quality',
            'directional_focused': 'Emphasizes direction prediction accuracy over absolute values'
        }
        return descriptions.get(name, 'Custom combination')
    
    def _get_combination_use_cases(self, name: str) -> List[str]:
        use_cases = {
            'basic': ['Initial experiments', 'Baseline comparison', 'Simple regression tasks'],
            'trading_focused': ['Live trading', 'Strategy backtesting', 'Profit optimization'],
            'robust_prediction': ['Noisy data', 'High volatility periods', 'Outlier resistance'],
            'comprehensive': ['Production models', 'General forecasting', 'Multi-objective optimization'],
            'directional_focused': ['Trend prediction', 'Signal generation', 'Directional accuracy']
        }
        return use_cases.get(name, ['Custom applications'])


# Convenience functions
def create_adaptive_loss(combination: str = 'comprehensive', **kwargs) -> AdaptiveLossFunction:
    """Quick function to create adaptive loss"""
    manager = LossSelectionManager()
    return manager.create_adaptive_loss(combination, **kwargs)


def get_recommended_loss(task_type: str, volatility: str = 'medium') -> AdaptiveLossFunction:
    """Get recommended adaptive loss for specific task"""
    manager = LossSelectionManager()
    
    data_chars = {'volatility': volatility}
    combination = manager.recommend_combination(task_type, data_chars)
    
    return manager.create_adaptive_loss(combination)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Adaptive Loss Function System")
    print("=" * 50)
    
    # Create manager
    manager = LossSelectionManager()
    
    # Show available combinations
    print("Available loss combinations:")
    for name in manager.predefined_combinations:
        info = manager.get_combination_info(name)
        print(f"  {name}: {info['description']}")
    
    # Create adaptive loss
    adaptive_loss = create_adaptive_loss('trading_focused', adaptation_strategy='performance_based')
    
    # Test with dummy data
    pred = torch.randn(32, 7, 1)
    true = torch.randn(32, 7, 1)
    
    # Forward pass
    result = adaptive_loss(pred, true)
    
    print(f"\nTest forward pass:")
    print(f"Combined loss: {result['combined_loss']:.4f}")
    print(f"Weights: {result['weights']}")
    print(f"Individual losses: {[f'{k}:{v:.4f}' for k, v in result['individual_losses'].items()]}")
    
    print("\nAdaptive loss system test completed!")