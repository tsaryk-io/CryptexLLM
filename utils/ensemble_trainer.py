import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import json
from collections import defaultdict

warnings.filterwarnings('ignore')


class MultiLLMEnsembleTrainer:
    """
    Trains ensemble of multiple LLM variants for robust predictions
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # LLM configurations to ensemble
        self.llm_configs = {
            'llama': {
                'llm_model': 'LLAMA',
                'llm_dim': 4096,
                'llm_layers': 8,
                'batch_size': 16,  # Smaller batch for larger model
                'learning_rate': 0.005
            },
            'gpt2': {
                'llm_model': 'GPT2', 
                'llm_dim': 768,
                'llm_layers': 6,
                'batch_size': 32,
                'learning_rate': 0.01
            },
            'bert': {
                'llm_model': 'BERT',
                'llm_dim': 768, 
                'llm_layers': 6,
                'batch_size': 32,
                'learning_rate': 0.01
            },
            'deepseek': {
                'llm_model': 'DEEPSEEK',
                'llm_dim': 4096,
                'llm_layers': 8,
                'batch_size': 16,
                'learning_rate': 0.005
            },
            'qwen': {
                'llm_model': 'QWEN',
                'llm_dim': 4096,
                'llm_layers': 8,
                'batch_size': 16,
                'learning_rate': 0.005
            }
        }
        
        # Training state
        self.models = {}
        self.optimizers = {}
        self.performance_history = defaultdict(list)
        self.ensemble_weights = {}
        
    def train_individual_models(self, train_loader, val_loader, epochs=10):
        """Train each LLM variant individually"""
        
        print("=" * 60)
        print("TRAINING INDIVIDUAL LLM MODELS")
        print("=" * 60)
        
        from models.TimeLLM import Model
        from utils.metrics import get_loss_function
        
        for llm_name, llm_config in self.llm_configs.items():
            print(f"\nTraining {llm_name.upper()} model...")
            
            # Create model configuration
            model_config = self._create_model_config(llm_config)
            
            try:
                # Initialize model
                model = Model(model_config)
                model = model.to(self.device)
                
                # Optimizer
                optimizer = torch.optim.Adam(
                    model.parameters(), 
                    lr=llm_config['learning_rate'],
                    weight_decay=1e-4
                )
                
                # Loss function
                criterion = get_loss_function(getattr(self.config, 'loss', 'mse'))
                
                # Training loop
                best_val_loss = float('inf')
                patience = 3
                patience_counter = 0
                
                for epoch in range(epochs):
                    # Training phase
                    model.train()
                    train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
                    
                    # Validation phase  
                    model.eval()
                    val_loss = self._validate_epoch(model, val_loader, criterion)
                    
                    print(f"  Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        torch.save(model.state_dict(), f'./trained_models/best_{llm_name}_model.pth')
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"  Early stopping at epoch {epoch+1}")
                            break
                
                # Store trained model
                model.load_state_dict(torch.load(f'./trained_models/best_{llm_name}_model.pth'))
                self.models[llm_name] = model
                self.optimizers[llm_name] = optimizer
                self.performance_history[llm_name].append(best_val_loss)
                
                print(f"  {llm_name.upper()} training completed. Best Val Loss: {best_val_loss:.6f}")
                
            except Exception as e:
                print(f"  Error training {llm_name}: {str(e)}")
                print(f"  Skipping {llm_name} model...")
                continue
    
    def _create_model_config(self, llm_config):
        """Create model configuration for specific LLM"""
        
        class ModelConfig:
            pass
        
        config = ModelConfig()
        
        # Copy base configuration
        for attr in dir(self.config):
            if not attr.startswith('_'):
                setattr(config, attr, getattr(self.config, attr))
        
        # Override with LLM-specific settings
        config.llm_model = llm_config['llm_model']
        config.llm_dim = llm_config['llm_dim'] 
        config.llm_layers = llm_config['llm_layers']
        
        return config
    
    def _train_epoch(self, model, data_loader, optimizer, criterion):
        """Train single epoch"""
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
            
            # Extract prediction
            pred_len = model.pred_len
            pred = outputs[:, -pred_len:, :]
            true = batch_y[:, -pred_len:, :]
            
            # Compute loss
            loss = criterion(pred, true)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_epoch(self, model, data_loader, criterion):
        """Validate single epoch"""
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Forward pass
                outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                
                # Extract prediction
                pred_len = model.pred_len
                pred = outputs[:, -pred_len:, :]
                true = batch_y[:, -pred_len:, :]
                
                # Compute loss
                loss = criterion(pred, true)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def compute_ensemble_weights(self, val_loader):
        """Compute optimal ensemble weights based on validation performance"""
        
        print("\n" + "=" * 60)
        print("COMPUTING ENSEMBLE WEIGHTS")
        print("=" * 60)
        
        if not self.models:
            print("No trained models available for ensemble")
            return
        
        # Collect predictions from all models
        all_predictions = {}
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Get target
                pred_len = list(self.models.values())[0].pred_len
                true = batch_y[:, -pred_len:, :].cpu().numpy()
                all_targets.append(true)
                
                # Get predictions from each model
                for model_name, model in self.models.items():
                    model.eval()
                    outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    pred = outputs[:, -pred_len:, :].cpu().numpy()
                    
                    if model_name not in all_predictions:
                        all_predictions[model_name] = []
                    all_predictions[model_name].append(pred)
        
        # Concatenate all predictions and targets
        all_targets = np.concatenate(all_targets, axis=0)
        for model_name in all_predictions:
            all_predictions[model_name] = np.concatenate(all_predictions[model_name], axis=0)
        
        # Compute individual model performance
        individual_mse = {}
        for model_name, predictions in all_predictions.items():
            mse = np.mean((predictions - all_targets) ** 2)
            individual_mse[model_name] = mse
            print(f"{model_name.upper()} MSE: {mse:.6f}")
        
        # Compute ensemble weights (inverse of MSE, normalized)
        weights = {}
        total_inverse_mse = sum(1.0 / mse for mse in individual_mse.values())
        
        for model_name, mse in individual_mse.items():
            weight = (1.0 / mse) / total_inverse_mse
            weights[model_name] = weight
            print(f"{model_name.upper()} Ensemble Weight: {weight:.4f}")
        
        self.ensemble_weights = weights
        
        # Test ensemble performance
        ensemble_pred = sum(
            weights[name] * predictions 
            for name, predictions in all_predictions.items()
        )
        ensemble_mse = np.mean((ensemble_pred - all_targets) ** 2)
        print(f"\nENSEMBLE MSE: {ensemble_mse:.6f}")
        
        # Improvement over best individual model
        best_individual_mse = min(individual_mse.values())
        improvement = (best_individual_mse - ensemble_mse) / best_individual_mse * 100
        print(f"Improvement over best individual: {improvement:.2f}%")
        
        return weights
    
    def ensemble_predict(self, data_loader):
        """Generate ensemble predictions"""
        
        if not self.models or not self.ensemble_weights:
            raise ValueError("Models must be trained and weights computed first")
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Get target
                pred_len = list(self.models.values())[0].pred_len
                true = batch_y[:, -pred_len:, :]
                all_targets.append(true.cpu().numpy())
                
                # Get weighted ensemble prediction
                ensemble_pred = None
                for model_name, model in self.models.items():
                    model.eval()
                    outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    pred = outputs[:, -pred_len:, :]
                    
                    weight = self.ensemble_weights.get(model_name, 0.0)
                    if ensemble_pred is None:
                        ensemble_pred = weight * pred
                    else:
                        ensemble_pred += weight * pred
                
                all_predictions.append(ensemble_pred.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        return predictions, targets
    
    def save_ensemble(self, save_path):
        """Save ensemble models and weights"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save individual models
        for model_name, model in self.models.items():
            torch.save(model.state_dict(), os.path.join(save_path, f'{model_name}_model.pth'))
        
        # Save ensemble configuration
        ensemble_config = {
            'llm_configs': self.llm_configs,
            'ensemble_weights': self.ensemble_weights,
            'performance_history': dict(self.performance_history)
        }
        
        with open(os.path.join(save_path, 'ensemble_config.json'), 'w') as f:
            json.dump(ensemble_config, f, indent=2)
        
        print(f"Ensemble saved to {save_path}")
    
    def load_ensemble(self, load_path):
        """Load ensemble models and weights"""
        from models.TimeLLM import Model
        
        # Load ensemble configuration
        with open(os.path.join(load_path, 'ensemble_config.json'), 'r') as f:
            ensemble_config = json.load(f)
        
        self.llm_configs = ensemble_config['llm_configs']
        self.ensemble_weights = ensemble_config['ensemble_weights']
        self.performance_history = defaultdict(list, ensemble_config['performance_history'])
        
        # Load individual models
        self.models = {}
        for model_name, llm_config in self.llm_configs.items():
            model_path = os.path.join(load_path, f'{model_name}_model.pth')
            if os.path.exists(model_path):
                model_config = self._create_model_config(llm_config)
                model = Model(model_config)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model = model.to(self.device)
                self.models[model_name] = model
        
        print(f"Ensemble loaded from {load_path}")
        print(f"Available models: {list(self.models.keys())}")


class MultiTimeframeTrainer:
    """
    Trainer for multi-timeframe datasets
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_multi_timeframe_data(self, data_loaders):
        """
        Combine data from multiple timeframes
        """
        combined_data = []
        
        timeframe_names = list(data_loaders.keys())
        print(f"Combining timeframes: {timeframe_names}")
        
        # Simple strategy: iterate through shortest timeframe
        min_length = min(len(loader) for loader in data_loaders.values())
        
        iterators = {name: iter(loader) for name, loader in data_loaders.items()}
        
        for i in range(min_length):
            timeframe_batch = {}
            
            for tf_name, iterator in iterators.items():
                try:
                    batch = next(iterator)
                    timeframe_batch[tf_name] = batch
                except StopIteration:
                    break
            
            if len(timeframe_batch) == len(timeframe_names):
                combined_data.append(timeframe_batch)
        
        print(f"Created {len(combined_data)} multi-timeframe batches")
        return combined_data
    
    def train_multi_timeframe_model(self, train_loaders, val_loaders, epochs=10):
        """
        Train model on multi-timeframe data
        """
        print("\n" + "=" * 60)
        print("TRAINING MULTI-TIMEFRAME MODEL")
        print("=" * 60)
        
        from models.MultiScaleTimeLLM import MultiScaleTimeLLM
        from utils.metrics import get_loss_function
        
        # Create model
        model = MultiScaleTimeLLM(self.config)
        model = model.to(self.device)
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = get_loss_function(getattr(self.config, 'loss', 'mse'))
        
        # Create multi-timeframe data
        train_data = self.create_multi_timeframe_data(train_loaders)
        val_data = self.create_multi_timeframe_data(val_loaders)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_data in train_data:
                optimizer.zero_grad()
                
                # Use primary timeframe for now (can be enhanced)
                primary_tf = list(batch_data.keys())[0]
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data[primary_tf]
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Forward pass
                outputs, consistency_loss = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                
                # Compute loss
                pred_len = model.pred_len
                pred = outputs[:, -pred_len:, :]
                true = batch_y[:, -pred_len:, :]
                
                main_loss = criterion(pred, true)
                total_loss = main_loss + consistency_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
            
            train_loss /= len(train_data)
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_data in val_data:
                    primary_tf = list(batch_data.keys())[0]
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data[primary_tf]
                    
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    
                    outputs, consistency_loss = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    
                    pred_len = model.pred_len
                    pred = outputs[:, -pred_len:, :]
                    true = batch_y[:, -pred_len:, :]
                    
                    main_loss = criterion(pred, true)
                    total_loss = main_loss + consistency_loss
                    
                    val_loss += total_loss.item()
            
            val_loss /= len(val_data)
            
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), './trained_models/best_multiscale_model.pth')
        
        print(f"Multi-timeframe training completed. Best Val Loss: {best_val_loss:.6f}")
        return model