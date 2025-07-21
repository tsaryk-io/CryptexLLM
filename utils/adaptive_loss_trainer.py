import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
from utils.adaptive_loss import AdaptiveLossFunction, LossSelectionManager

warnings.filterwarnings('ignore')


class AdaptiveLossTrainer:
    """
    Training integration for adaptive loss functions with monitoring and visualization
    """
    
    def __init__(self, 
                 model: nn.Module,
                 adaptive_loss: AdaptiveLossFunction,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 model_id: str = "adaptive_loss_model"):
        
        self.model = model
        self.adaptive_loss = adaptive_loss
        self.optimizer = optimizer
        self.device = device
        self.model_id = model_id
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.training_history = []
        self.validation_history = []
        self.best_val_loss = float('inf')
        self.best_weights = None
        
        # Loss monitoring
        self.loss_evolution = {
            'epochs': [],
            'weights': [],
            'individual_losses': [],
            'combined_losses': [],
            'performance_metrics': []
        }
        
        print(f"AdaptiveLossTrainer initialized for model: {model_id}")
        print(f"Loss functions: {list(adaptive_loss.loss_names)}")
        print(f"Adaptation strategy: {adaptive_loss.adaptation_strategy}")
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train one epoch with adaptive loss"""
        self.model.train()
        self.adaptive_loss.train()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'individual_losses': {name: 0.0 for name in self.adaptive_loss.loss_names},
            'num_batches': 0
        }
        
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            # Move to device
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
            
            # Extract prediction
            pred_len = self.model.pred_len
            pred = outputs[:, -pred_len:, :]
            true = batch_y[:, -pred_len:, :]
            
            # Compute adaptive loss
            loss_result = self.adaptive_loss(pred, true)
            total_loss = loss_result['combined_loss']
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_metrics['total_loss'] += total_loss.item()
            for name, loss_val in loss_result['individual_losses'].items():
                epoch_metrics['individual_losses'][name] += loss_val.item()
            epoch_metrics['num_batches'] += 1
            
            self.global_step += 1
            
            # Log progress
            if batch_idx % 100 == 0:
                current_weights = loss_result['weights']
                print(f"Epoch {self.epoch}, Batch {batch_idx}/{len(train_loader)}")
                print(f"  Total Loss: {total_loss.item():.6f}")
                print(f"  Weights: {[f'{k}:{v:.3f}' for k, v in current_weights.items()]}")
        
        # Average metrics
        for key in epoch_metrics:
            if key != 'num_batches':
                if isinstance(epoch_metrics[key], dict):
                    for subkey in epoch_metrics[key]:
                        epoch_metrics[key][subkey] /= epoch_metrics['num_batches']
                else:
                    epoch_metrics[key] /= epoch_metrics['num_batches']
        
        return epoch_metrics
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """Validate one epoch"""
        self.model.eval()
        self.adaptive_loss.eval()
        
        val_metrics = {
            'total_loss': 0.0,
            'individual_losses': {name: 0.0 for name in self.adaptive_loss.loss_names},
            'num_batches': 0
        }
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                
                pred_len = self.model.pred_len
                pred = outputs[:, -pred_len:, :]
                true = batch_y[:, -pred_len:, :]
                
                loss_result = self.adaptive_loss(pred, true)
                total_loss = loss_result['combined_loss']
                
                val_metrics['total_loss'] += total_loss.item()
                for name, loss_val in loss_result['individual_losses'].items():
                    val_metrics['individual_losses'][name] += loss_val.item()
                val_metrics['num_batches'] += 1
        
        # Average metrics
        for key in val_metrics:
            if key != 'num_batches':
                if isinstance(val_metrics[key], dict):
                    for subkey in val_metrics[key]:
                        val_metrics[key][subkey] /= val_metrics['num_batches']
                else:
                    val_metrics[key] /= val_metrics['num_batches']
        
        return val_metrics
    
    def train(self, 
              train_loader,
              val_loader,
              epochs: int,
              patience: int = 5,
              save_best: bool = True,
              log_frequency: int = 1) -> Dict:
        """
        Complete training loop with adaptive loss
        """
        print(f"Starting adaptive loss training for {epochs} epochs...")
        print(f"Patience: {patience}, Save best: {save_best}")
        
        patience_counter = 0
        training_start_time = time.time()
        
        for epoch in range(epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Record metrics
            self.training_history.append(train_metrics)
            self.validation_history.append(val_metrics)
            
            # Track loss evolution
            current_weights = self.adaptive_loss.get_current_weights()
            self.loss_evolution['epochs'].append(epoch)
            self.loss_evolution['weights'].append({
                name: current_weights[i].item() 
                for i, name in enumerate(self.adaptive_loss.loss_names)
            })
            self.loss_evolution['individual_losses'].append(val_metrics['individual_losses'])
            self.loss_evolution['combined_losses'].append(val_metrics['total_loss'])
            
            # Early stopping and best model saving
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.best_weights = {
                    name: current_weights[i].item() 
                    for i, name in enumerate(self.adaptive_loss.loss_names)
                }
                patience_counter = 0
                
                if save_best:
                    self.save_checkpoint('best')
            else:
                patience_counter += 1
            
            # Logging
            if epoch % log_frequency == 0:
                epoch_time = time.time() - epoch_start_time
                print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.2f}s)")
                print(f"  Train Loss: {train_metrics['total_loss']:.6f}")
                print(f"  Val Loss: {val_metrics['total_loss']:.6f}")
                print(f"  Current Weights: {[f'{k}:{v:.3f}' for k, v in self.loss_evolution['weights'][-1].items()]}")
                print(f"  Best Val Loss: {self.best_val_loss:.6f}")
                
                # Individual loss breakdown
                print("  Individual Val Losses:")
                for name, loss_val in val_metrics['individual_losses'].items():
                    print(f"    {name}: {loss_val:.6f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (patience: {patience})")
                break
        
        training_time = time.time() - training_start_time
        
        # Final results
        results = {
            'total_epochs': self.epoch + 1,
            'training_time_minutes': training_time / 60,
            'best_val_loss': self.best_val_loss,
            'best_weights': self.best_weights,
            'final_weights': self.loss_evolution['weights'][-1],
            'adaptation_summary': self.adaptive_loss.get_performance_summary()
        }
        
        print(f"\n{'='*60}")
        print("ADAPTIVE LOSS TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Total epochs: {results['total_epochs']}")
        print(f"Training time: {results['training_time_minutes']:.2f} minutes")
        print(f"Best validation loss: {results['best_val_loss']:.6f}")
        print(f"Best weights: {[f'{k}:{v:.3f}' for k, v in results['best_weights'].items()]}")
        
        return results
    
    def save_checkpoint(self, checkpoint_type: str = 'latest'):
        """Save model and adaptive loss state"""
        checkpoint_dir = f'./checkpoints/{self.model_id}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'adaptive_loss_weights': self.adaptive_loss.get_current_weights(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'validation_history': self.validation_history,
            'loss_evolution': self.loss_evolution
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{checkpoint_type}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save adaptive loss configuration and history
        self.adaptive_loss.save_adaptation_history(
            os.path.join(checkpoint_dir, f'adaptive_loss_history_{checkpoint_type}.json')
        )
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def visualize_loss_evolution(self, save_path: Optional[str] = None):
        """Visualize how loss weights and values evolve during training"""
        if not self.loss_evolution['epochs']:
            print("No training history to visualize")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = self.loss_evolution['epochs']
        
        # 1. Weight evolution
        for name in self.adaptive_loss.loss_names:
            weights = [w[name] for w in self.loss_evolution['weights']]
            ax1.plot(epochs, weights, label=name, marker='o', markersize=3)
        ax1.set_title('Loss Weight Evolution')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Weight')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Combined loss evolution
        ax2.plot(epochs, self.loss_evolution['combined_losses'], 'b-', linewidth=2, label='Combined Loss')
        ax2.set_title('Combined Loss Evolution')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Individual loss evolution
        for name in self.adaptive_loss.loss_names:
            individual_losses = [losses[name] for losses in self.loss_evolution['individual_losses']]
            ax3.plot(epochs, individual_losses, label=name, marker='s', markersize=2)
        ax3.set_title('Individual Loss Evolution')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Weight distribution (final)
        final_weights = self.loss_evolution['weights'][-1]
        names = list(final_weights.keys())
        weights = list(final_weights.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
        
        wedges, texts, autotexts = ax4.pie(weights, labels=names, autopct='%1.1f%%', colors=colors)
        ax4.set_title('Final Weight Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss evolution plot saved: {save_path}")
        else:
            plt.show()
    
    def analyze_loss_performance(self) -> Dict:
        """Analyze the performance of different loss combinations"""
        if not self.validation_history:
            return {}
        
        analysis = {
            'weight_statistics': {},
            'loss_correlations': {},
            'best_performing_combinations': [],
            'adaptation_effectiveness': {}
        }
        
        # Weight statistics
        for name in self.adaptive_loss.loss_names:
            weights = [w[name] for w in self.loss_evolution['weights']]
            analysis['weight_statistics'][name] = {
                'mean': np.mean(weights),
                'std': np.std(weights),
                'min': np.min(weights),
                'max': np.max(weights),
                'final': weights[-1] if weights else 0
            }
        
        # Find best performing epochs
        val_losses = [h['total_loss'] for h in self.validation_history]
        best_indices = np.argsort(val_losses)[:5]  # Top 5 epochs
        
        for i, idx in enumerate(best_indices):
            if idx < len(self.loss_evolution['weights']):
                analysis['best_performing_combinations'].append({
                    'rank': i + 1,
                    'epoch': idx,
                    'val_loss': val_losses[idx],
                    'weights': self.loss_evolution['weights'][idx]
                })
        
        # Adaptation effectiveness
        if len(self.loss_evolution['combined_losses']) > 1:
            initial_loss = self.loss_evolution['combined_losses'][0]
            final_loss = self.loss_evolution['combined_losses'][-1]
            improvement = (initial_loss - final_loss) / initial_loss * 100
            
            analysis['adaptation_effectiveness'] = {
                'initial_loss': initial_loss,
                'final_loss': final_loss,
                'improvement_percent': improvement,
                'converged': improvement > 0
            }
        
        return analysis


def create_adaptive_trainer(model: nn.Module,
                          optimizer: torch.optim.Optimizer,
                          device: torch.device,
                          loss_combination: str = 'trading_focused',
                          adaptation_strategy: str = 'performance_based',
                          model_id: str = "adaptive_model") -> AdaptiveLossTrainer:
    """
    Convenience function to create adaptive trainer
    """
    # Create adaptive loss
    manager = LossSelectionManager()
    adaptive_loss = manager.create_adaptive_loss(
        combination=loss_combination,
        adaptation_strategy=adaptation_strategy,
        adaptation_frequency=50,  # Adapt every 50 steps
        performance_window=20     # Look at last 20 steps for performance
    )
    
    # Create trainer
    trainer = AdaptiveLossTrainer(
        model=model,
        adaptive_loss=adaptive_loss,
        optimizer=optimizer,
        device=device,
        model_id=model_id
    )
    
    return trainer


if __name__ == "__main__":
    # Example usage
    print("Testing Adaptive Loss Trainer")
    print("=" * 40)
    
    # Create dummy model and data
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
            self.pred_len = 7
            
        def forward(self, x, x_mark, y, y_mark):
            return self.linear(x[:, :, :10])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DummyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create adaptive trainer
    trainer = create_adaptive_trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        loss_combination='comprehensive',
        model_id='test_adaptive'
    )
    
    print("✅ Adaptive trainer created successfully!")
    print(f"Loss functions: {trainer.adaptive_loss.loss_names}")
    print(f"Initial weights: {trainer.adaptive_loss.get_current_weights()}")