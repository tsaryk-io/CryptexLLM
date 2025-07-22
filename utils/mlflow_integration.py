"""
MLFlow integration for Time-LLM-Cryptex experiment tracking
"""

import os
import json
import mlflow
import mlflow.pytorch
import torch
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import shutil


class MLFlowExperimentTracker:
    """
    MLFlow integration for tracking Time-LLM experiments
    """
    
    def __init__(self, experiment_name: str = "TimeLLM-Cryptex", tracking_uri: str = None):
        """
        Initialize MLFlow experiment tracker
        
        Args:
            experiment_name: Name of the MLFlow experiment
            tracking_uri: MLFlow tracking server URI (default: local directory)
        """
        # Set tracking URI (defaults to ./mlruns)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Use local directory for tracking
            mlruns_dir = "./mlruns"
            if not os.path.exists(mlruns_dir):
                os.makedirs(mlruns_dir)
            mlflow.set_tracking_uri(f"file://{os.path.abspath(mlruns_dir)}")
        
        # Set or create experiment
        self.experiment_name = experiment_name
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)
        
        self.run_id = None
        self.active_run = None
        
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Start a new MLFlow run"""
        if self.active_run:
            print("Warning: Previous run still active. Ending it first.")
            self.end_run()
        
        # Generate run name if not provided
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"timellm_run_{timestamp}"
        
        self.active_run = mlflow.start_run(run_name=run_name, tags=tags)
        self.run_id = self.active_run.info.run_id
        
        print(f"Started MLFlow run: {run_name} (ID: {self.run_id})")
        return self.run_id
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        if not self.active_run:
            raise ValueError("No active MLFlow run. Call start_run() first.")
        
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        if not self.active_run:
            raise ValueError("No active MLFlow run. Call start_run() first.")
        
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model(self, model: torch.nn.Module, model_name: str = "time_llm_model", 
                  artifacts: Optional[Dict[str, str]] = None):
        """Log PyTorch model and artifacts"""
        if not self.active_run:
            raise ValueError("No active MLFlow run. Call start_run() first.")
        
        # Log the model
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=model_name,
            registered_model_name=f"{self.experiment_name}_{model_name}"
        )
        
        # Log additional artifacts if provided
        if artifacts:
            for name, path in artifacts.items():
                if os.path.exists(path):
                    mlflow.log_artifact(path, artifact_path=f"artifacts/{name}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration as artifact"""
        if not self.active_run:
            raise ValueError("No active MLFlow run. Call start_run() first.")
        
        config_path = f"./temp_config_{self.run_id}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        mlflow.log_artifact(config_path, artifact_path="config")
        os.remove(config_path)  # Clean up temp file
    
    def log_sentiment_data_info(self, sentiment_info: Dict[str, Any]):
        """Log sentiment data source information"""
        if not self.active_run:
            raise ValueError("No active MLFlow run. Call start_run() first.")
        
        # Log as parameters
        for key, value in sentiment_info.items():
            mlflow.log_param(f"sentiment_{key}", value)
    
    def log_training_progress(self, epoch: int, train_loss: float, val_loss: float, 
                            test_loss: float, metric_value: float, metric_name: str = "MAE"):
        """Log training progress for each epoch"""
        if not self.active_run:
            raise ValueError("No active MLFlow run. Call start_run() first.")
        
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            f"test_{metric_name.lower()}": metric_value
        }
        
        self.log_metrics(metrics, step=epoch)
    
    def log_external_data_stats(self, external_data_stats: Dict[str, Dict[str, Any]]):
        """Log statistics about external data sources"""
        if not self.active_run:
            raise ValueError("No active MLFlow run. Call start_run() first.")
        
        for source_name, stats in external_data_stats.items():
            for stat_name, value in stats.items():
                param_name = f"data_{source_name}_{stat_name}"
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(param_name, value)
    
    def end_run(self, status: str = "FINISHED"):
        """End the current MLFlow run"""
        if self.active_run:
            mlflow.end_run(status=status)
            print(f"Ended MLFlow run: {self.run_id}")
            self.active_run = None
            self.run_id = None
    
    def get_best_run(self, metric_name: str = "test_mae", ascending: bool = True):
        """Get the best run based on a metric"""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"]
        )
        
        if not runs.empty:
            best_run = runs.iloc[0]
            return {
                'run_id': best_run['run_id'],
                'params': {k.replace('params.', ''): v for k, v in best_run.items() if k.startswith('params.')},
                'metrics': {k.replace('metrics.', ''): v for k, v in best_run.items() if k.startswith('metrics.')},
                'best_metric': best_run[f'metrics.{metric_name}']
            }
        return None
    
    def compare_runs(self, metric_names: list = None):
        """Compare all runs in the experiment"""
        if not metric_names:
            metric_names = ["test_mae", "test_loss", "val_loss"]
        
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        comparison_data = []
        for _, run in runs.iterrows():
            run_data = {
                'run_id': run['run_id'][:8],  # Short ID
                'run_name': run['tags.mlflow.runName'] if 'tags.mlflow.runName' in run else 'N/A',
                'status': run['status'],
                'start_time': run['start_time']
            }
            
            # Add metrics
            for metric in metric_names:
                metric_col = f'metrics.{metric}'
                run_data[metric] = run[metric_col] if metric_col in run else None
            
            # Add key parameters
            param_cols = [col for col in run.columns if col.startswith('params.')]
            for param_col in param_cols[:10]:  # Limit to first 10 params
                param_name = param_col.replace('params.', '')
                run_data[param_name] = run[param_col]
            
            comparison_data.append(run_data)
        
        return comparison_data


def setup_mlflow_for_timellm(args, external_data_config: Dict = None):
    """
    Setup MLFlow tracking for TimeLLM experiment based on args
    
    Args:
        args: Argument parser object with model configuration
        external_data_config: Configuration for external data sources
    
    Returns:
        MLFlowExperimentTracker instance
    """
    
    # Initialize tracker
    tracker = MLFlowExperimentTracker()
    
    # Create run name from model configuration
    run_name = f"{args.model_id}_{args.task_name}_{datetime.now().strftime('%H%M%S')}"
    
    # Start run with tags
    tags = {
        "task_type": args.task_name,
        "model_type": args.model,
        "llm_model": args.llm_model,
        "granularity": getattr(args, 'granularity', 'unknown'),
        "features": args.features
    }
    
    tracker.start_run(run_name=run_name, tags=tags)
    
    # Log all hyperparameters
    params = {
        # Model architecture
        "model_id": args.model_id,
        "model": args.model,
        "llm_model": args.llm_model,
        "llm_dim": args.llm_dim,
        "llm_layers": args.llm_layers,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_tokens": getattr(args, 'num_tokens', 'N/A'),
        
        # Data configuration
        "features": args.features,
        "seq_len": args.seq_len,
        "label_len": args.label_len,
        "pred_len": args.pred_len,
        "patch_len": getattr(args, 'patch_len', 'N/A'),
        "stride": getattr(args, 'stride', 'N/A'),
        "enc_in": args.enc_in,
        "dec_in": args.dec_in,
        "c_out": args.c_out,
        
        # Training configuration
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_epochs": args.train_epochs,
        "loss_function": args.loss,
        "metric_function": args.metric,
        "lradj": args.lradj,
        
        # Data source
        "data": args.data,
        "data_path": args.data_path,
        "target": args.target
    }
    
    tracker.log_params(params)
    
    # Log external data configuration if provided
    if external_data_config:
        tracker.log_config({"external_data_config": external_data_config})
        
        # Log sentiment data info
        sentiment_info = {
            "reddit_enabled": True,
            "news_enabled": bool(external_data_config.get('sentiment', {}).get('api_key')),
            "fear_greed_enabled": True,
            "sentiment_api_rate_limit": external_data_config.get('sentiment', {}).get('rate_limit', 1.0)
        }
        tracker.log_sentiment_data_info(sentiment_info)
    
    return tracker


if __name__ == "__main__":
    # Test the MLFlow integration
    print("Testing MLFlow Integration...")
    
    # Initialize tracker
    tracker = MLFlowExperimentTracker("test_experiment")
    
    # Start a test run
    tracker.start_run("test_run", tags={"test": "true"})
    
    # Log some test parameters and metrics
    tracker.log_params({
        "learning_rate": 0.001,
        "batch_size": 32,
        "seq_len": 96
    })
    
    tracker.log_metrics({
        "train_loss": 0.5,
        "val_loss": 0.4,
        "test_mae": 0.3
    })
    
    # End run
    tracker.end_run()
    
    print("MLFlow integration test completed!")
    print(f"Check ./mlruns directory for tracked experiment data")