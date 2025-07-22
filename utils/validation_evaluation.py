#!/usr/bin/env python3
"""
Comprehensive Validation & Evaluation System for Time Series Cryptocurrency Prediction

This module provides advanced validation techniques, trading performance metrics,
and benchmarking capabilities specifically designed for cryptocurrency time series
forecasting with TimeLLM enhancements.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import warnings
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for validation and evaluation parameters"""
    # Walk-forward validation settings
    initial_train_size: int = 1000      # Initial training window size
    validation_size: int = 100          # Size of each validation window
    step_size: int = 24                 # Step size for rolling validation
    min_train_size: int = 500           # Minimum training size
    max_validations: int = 50           # Maximum number of validation windows
    
    # Trading simulation settings
    transaction_cost: float = 0.001     # 0.1% transaction cost
    position_size_limit: float = 1.0    # Maximum position size
    leverage: float = 1.0               # Leverage for trading
    slippage: float = 0.0005           # 0.05% slippage
    
    # Risk management settings
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    var_window: int = 252               # VaR calculation window
    max_drawdown_threshold: float = 0.2  # 20% max drawdown warning
    
    # Performance thresholds
    min_sharpe_ratio: float = 0.5       # Minimum acceptable Sharpe ratio
    min_win_rate: float = 0.45          # Minimum win rate threshold
    benchmark_return: float = 0.0       # Benchmark return for comparison


@dataclass
class ValidationResult:
    """Result of a single validation window"""
    window_id: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    predictions: np.ndarray
    actual: np.ndarray
    timestamp: pd.DatetimeIndex
    
    # Performance metrics
    mse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0
    r2: float = 0.0
    
    # Trading metrics
    returns: np.ndarray = field(default_factory=lambda: np.array([]))
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall: float = 0.0


@dataclass
class BenchmarkResult:
    """Comprehensive benchmark comparison results"""
    model_name: str
    dataset_type: str
    evaluation_metrics: Dict[str, float]
    trading_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    regime_performance: Dict[str, Dict[str, float]]
    validation_windows: List[ValidationResult]
    overall_score: float = 0.0


class TimeSeriesValidator:
    """Advanced time series validation with walk-forward methodology"""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.validation_results = []
        
    def walk_forward_validation(self, 
                              data: pd.DataFrame,
                              predictions_func: callable,
                              target_col: str = 'close',
                              timestamp_col: str = 'timestamp') -> List[ValidationResult]:
        """
        Perform walk-forward validation for time series data
        
        Args:
            data: Time series data with target and features
            predictions_func: Function that takes (train_data, val_data) and returns predictions
            target_col: Name of target column
            timestamp_col: Name of timestamp column
            
        Returns:
            List of validation results for each window
        """
        logger.info(f"Starting walk-forward validation with {len(data)} data points")
        
        results = []
        data_sorted = data.sort_values(timestamp_col).reset_index(drop=True)
        
        # Ensure we have enough data
        if len(data_sorted) < self.config.initial_train_size + self.config.validation_size:
            raise ValueError(f"Insufficient data for validation. Need at least {self.config.initial_train_size + self.config.validation_size} points")
        
        train_start = 0
        train_end = self.config.initial_train_size
        validation_count = 0
        
        while (train_end + self.config.validation_size <= len(data_sorted) and 
               validation_count < self.config.max_validations):
            
            val_start = train_end
            val_end = min(val_start + self.config.validation_size, len(data_sorted))
            
            # Extract training and validation data
            train_data = data_sorted.iloc[train_start:train_end].copy()
            val_data = data_sorted.iloc[val_start:val_end].copy()
            
            logger.info(f"Validation window {validation_count + 1}: Train[{train_start}:{train_end}], Val[{val_start}:{val_end}]")
            
            try:
                # Get predictions from the provided function
                predictions = predictions_func(train_data, val_data)
                actual = val_data[target_col].values
                
                # Create timestamps for validation period
                if timestamp_col in val_data.columns:
                    timestamps = pd.to_datetime(val_data[timestamp_col])
                else:
                    timestamps = pd.date_range(start='2023-01-01', periods=len(actual), freq='H')
                
                # Create validation result
                result = ValidationResult(
                    window_id=validation_count + 1,
                    train_start=train_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    predictions=predictions,
                    actual=actual,
                    timestamp=timestamps
                )
                
                # Calculate performance metrics
                self._calculate_metrics(result)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error in validation window {validation_count + 1}: {e}")
                continue
            
            # Move to next window
            train_start += self.config.step_size
            train_end += self.config.step_size
            validation_count += 1
            
            # Ensure minimum training size
            if train_end - train_start < self.config.min_train_size:
                break
        
        logger.info(f"Completed walk-forward validation with {len(results)} windows")
        self.validation_results = results
        return results
    
    def _calculate_metrics(self, result: ValidationResult):
        """Calculate comprehensive metrics for a validation result"""
        pred = result.predictions
        actual = result.actual
        
        # Basic regression metrics
        result.mse = mean_squared_error(actual, pred)
        result.mae = mean_absolute_error(actual, pred)
        result.mape = np.mean(np.abs((actual - pred) / actual)) * 100
        result.r2 = r2_score(actual, pred)
        
        # Trading metrics
        self._calculate_trading_metrics(result)
        
        # Risk metrics
        self._calculate_risk_metrics(result)
    
    def _calculate_trading_metrics(self, result: ValidationResult):
        """Calculate trading-specific performance metrics"""
        pred = result.predictions
        actual = result.actual
        
        # Ensure arrays are the same length and handle edge cases
        min_length = min(len(pred), len(actual))
        if min_length < 2:
            return
            
        pred = pred[:min_length]
        actual = actual[:min_length]
        
        # Generate trading signals (simple strategy: buy if predicted > current, sell otherwise)
        if len(pred) > 1 and len(actual) > 1:
            # Price changes
            actual_returns = np.diff(actual) / actual[:-1]
            
            # Predicted direction vs actual direction
            pred_direction = np.sign(np.diff(pred))
            
            # Ensure arrays are same length
            min_ret_length = min(len(actual_returns), len(pred_direction))
            actual_returns = actual_returns[:min_ret_length]
            pred_direction = pred_direction[:min_ret_length]
            
            # Trading returns (assuming we trade based on predicted direction)
            trading_returns = pred_direction * actual_returns
            
            # Apply transaction costs
            transaction_costs = np.abs(np.diff(np.concatenate([[0], pred_direction]))) * self.config.transaction_cost
            # Ensure transaction costs array matches trading returns length
            transaction_costs_adjusted = transaction_costs[1:len(trading_returns)+1]
            trading_returns_net = trading_returns - transaction_costs_adjusted
            
            result.returns = trading_returns_net
            
            # Sharpe ratio (annualized, assuming hourly data)
            if len(trading_returns_net) > 0 and np.std(trading_returns_net) > 0:
                result.sharpe_ratio = np.sqrt(24 * 365) * np.mean(trading_returns_net) / np.std(trading_returns_net)
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + trading_returns_net)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            result.max_drawdown = np.min(drawdown)
            
            # Win rate
            winning_trades = trading_returns_net > 0
            result.win_rate = np.mean(winning_trades) if len(winning_trades) > 0 else 0
            
            # Profit factor
            gross_profit = np.sum(trading_returns_net[trading_returns_net > 0])
            gross_loss = abs(np.sum(trading_returns_net[trading_returns_net < 0]))
            result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    def _calculate_risk_metrics(self, result: ValidationResult):
        """Calculate risk assessment metrics"""
        if len(result.returns) == 0:
            return
        
        returns = result.returns
        
        # Value at Risk
        for conf_level in self.config.confidence_levels:
            var_value = np.percentile(returns, (1 - conf_level) * 100)
            if conf_level == 0.95:
                result.var_95 = var_value
            elif conf_level == 0.99:
                result.var_99 = var_value
        
        # Expected Shortfall (Conditional VaR)
        var_95_threshold = np.percentile(returns, 5)
        tail_losses = returns[returns <= var_95_threshold]
        result.expected_shortfall = np.mean(tail_losses) if len(tail_losses) > 0 else 0


class TradingPerformanceAnalyzer:
    """Specialized analyzer for trading performance metrics"""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
    
    def analyze_trading_performance(self, 
                                  predictions: np.ndarray, 
                                  actual: np.ndarray,
                                  timestamps: pd.DatetimeIndex = None) -> Dict[str, float]:
        """Comprehensive trading performance analysis"""
        
        if timestamps is None:
            timestamps = pd.date_range(start='2023-01-01', periods=len(actual), freq='H')
        
        # Price returns
        actual_returns = np.diff(actual) / actual[:-1]
        
        # Strategy returns (simple directional strategy)
        pred_direction = np.sign(np.diff(predictions))
        strategy_returns = pred_direction * actual_returns
        
        # Apply costs
        position_changes = np.abs(np.diff(np.concatenate([[0], pred_direction])))
        transaction_costs = position_changes * self.config.transaction_cost
        
        # Ensure arrays are same length
        min_length = min(len(strategy_returns), len(transaction_costs) - 1)
        strategy_returns = strategy_returns[:min_length]
        transaction_costs_adj = transaction_costs[1:min_length+1]
        net_returns = strategy_returns - transaction_costs_adj
        
        # Cumulative performance
        cumulative_returns = np.cumprod(1 + net_returns)
        total_return = cumulative_returns[-1] - 1
        
        # Annualized metrics (assuming hourly data)
        periods_per_year = 24 * 365
        annualized_return = (1 + total_return) ** (periods_per_year / len(net_returns)) - 1
        annualized_volatility = np.std(net_returns) * np.sqrt(periods_per_year)
        
        # Risk-adjusted metrics
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Drawdown analysis
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = net_returns[net_returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_std if downside_std > 0 else 0
        
        # Trade analysis - ensure consistent array lengths
        pred_direction_adj = pred_direction[:len(strategy_returns)]
        trades = pred_direction_adj[pred_direction_adj != 0]  # Only actual position changes
        trade_returns = strategy_returns[pred_direction_adj != 0]
        
        win_rate = np.mean(trade_returns > 0) if len(trade_returns) > 0 else 0
        avg_win = np.mean(trade_returns[trade_returns > 0]) if np.any(trade_returns > 0) else 0
        avg_loss = np.mean(trade_returns[trade_returns < 0]) if np.any(trade_returns < 0) else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Information ratio (vs benchmark)
        benchmark_return = self.config.benchmark_return
        excess_returns = net_returns - benchmark_return / periods_per_year
        tracking_error = np.std(excess_returns) * np.sqrt(periods_per_year)
        information_ratio = np.mean(excess_returns) * periods_per_year / tracking_error if tracking_error > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'information_ratio': information_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(trades),
            'transaction_cost_impact': np.sum(transaction_costs[1:])
        }
    
    def calculate_regime_performance(self, 
                                   predictions: np.ndarray,
                                   actual: np.ndarray,
                                   regimes: List[str],
                                   timestamps: pd.DatetimeIndex = None) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics by market regime"""
        
        regime_performance = {}
        unique_regimes = list(set(regimes))
        
        for regime in unique_regimes:
            # Find indices for this regime
            regime_indices = [i for i, r in enumerate(regimes) if r == regime]
            
            if len(regime_indices) > 10:  # Minimum data points for meaningful analysis
                regime_pred = predictions[regime_indices]
                regime_actual = actual[regime_indices]
                regime_timestamps = timestamps[regime_indices] if timestamps is not None else None
                
                # Calculate performance for this regime
                performance = self.analyze_trading_performance(
                    regime_pred, regime_actual, regime_timestamps
                )
                
                # Add regime-specific metrics
                performance['regime_periods'] = len(regime_indices)
                performance['regime_percentage'] = len(regime_indices) / len(regimes) * 100
                
                regime_performance[regime] = performance
        
        return regime_performance
    
    def generate_performance_report(self, 
                                  performance_metrics: Dict[str, float],
                                  regime_performance: Dict[str, Dict[str, float]] = None) -> str:
        """Generate a comprehensive performance report"""
        
        report = []
        report.append("=" * 60)
        report.append("TRADING PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Overall performance
        report.append("\nOVERALL PERFORMANCE:")
        report.append(f"Total Return: {performance_metrics['total_return']:.2%}")
        report.append(f"Annualized Return: {performance_metrics['annualized_return']:.2%}")
        report.append(f"Annualized Volatility: {performance_metrics['annualized_volatility']:.2%}")
        report.append(f"Sharpe Ratio: {performance_metrics['sharpe_ratio']:.3f}")
        report.append(f"Sortino Ratio: {performance_metrics['sortino_ratio']:.3f}")
        report.append(f"Calmar Ratio: {performance_metrics['calmar_ratio']:.3f}")
        
        # Risk metrics
        report.append("\nRISK METRICS:")
        report.append(f"Maximum Drawdown: {performance_metrics['max_drawdown']:.2%}")
        report.append(f"Information Ratio: {performance_metrics['information_ratio']:.3f}")
        
        # Trading metrics
        report.append("\nTRADING METRICS:")
        report.append(f"Win Rate: {performance_metrics['win_rate']:.2%}")
        report.append(f"Profit Factor: {performance_metrics['profit_factor']:.2f}")
        report.append(f"Average Win: {performance_metrics['avg_win']:.4f}")
        report.append(f"Average Loss: {performance_metrics['avg_loss']:.4f}")
        report.append(f"Total Trades: {performance_metrics['total_trades']}")
        report.append(f"Transaction Cost Impact: {performance_metrics['transaction_cost_impact']:.4f}")
        
        # Performance assessment
        report.append("\nPERFORMANCE ASSESSMENT:")
        if performance_metrics['sharpe_ratio'] >= self.config.min_sharpe_ratio:
            report.append("✓ Sharpe ratio meets minimum threshold")
        else:
            report.append("✗ Sharpe ratio below minimum threshold")
        
        if performance_metrics['win_rate'] >= self.config.min_win_rate:
            report.append("PASS: Win rate meets minimum threshold")
        else:
            report.append("FAIL: Win rate below minimum threshold")
        
        if abs(performance_metrics['max_drawdown']) <= self.config.max_drawdown_threshold:
            report.append("PASS: Maximum drawdown within acceptable range")
        else:
            report.append("WARNING: Maximum drawdown exceeds threshold")
        
        # Regime performance
        if regime_performance:
            report.append("\nREGIME-SPECIFIC PERFORMANCE:")
            for regime, metrics in regime_performance.items():
                report.append(f"\n{regime.upper()}:")
                report.append(f"  Periods: {metrics['regime_periods']} ({metrics['regime_percentage']:.1f}%)")
                report.append(f"  Return: {metrics['total_return']:.2%}")
                report.append(f"  Sharpe: {metrics['sharpe_ratio']:.3f}")
                report.append(f"  Max DD: {metrics['max_drawdown']:.2%}")
                report.append(f"  Win Rate: {metrics['win_rate']:.2%}")
        
        return "\n".join(report)


class BenchmarkComparator:
    """Compare multiple models and configurations"""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.benchmark_results = []
    
    def add_benchmark_result(self, 
                           model_name: str,
                           dataset_type: str,
                           predictions: np.ndarray,
                           actual: np.ndarray,
                           timestamps: pd.DatetimeIndex = None,
                           regimes: List[str] = None) -> BenchmarkResult:
        """Add a model result for benchmarking"""
        
        # Basic evaluation metrics
        mse = mean_squared_error(actual, predictions)
        mae = mean_absolute_error(actual, predictions)
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        r2 = r2_score(actual, predictions)
        
        evaluation_metrics = {
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }
        
        # Trading performance
        analyzer = TradingPerformanceAnalyzer(self.config)
        trading_metrics = analyzer.analyze_trading_performance(predictions, actual, timestamps)
        
        # Risk metrics (subset of trading metrics)
        risk_metrics = {
            'max_drawdown': trading_metrics['max_drawdown'],
            'annualized_volatility': trading_metrics['annualized_volatility'],
            'var_95': np.percentile(np.diff(actual) / actual[:-1], 5),
            'expected_shortfall': np.mean(np.diff(actual)[np.diff(actual) <= np.percentile(np.diff(actual), 5)])
        }
        
        # Regime performance
        regime_performance = {}
        if regimes:
            regime_performance = analyzer.calculate_regime_performance(
                predictions, actual, regimes, timestamps
            )
        
        # Calculate overall score (weighted combination of metrics)
        overall_score = self._calculate_overall_score(evaluation_metrics, trading_metrics, risk_metrics)
        
        result = BenchmarkResult(
            model_name=model_name,
            dataset_type=dataset_type,
            evaluation_metrics=evaluation_metrics,
            trading_metrics=trading_metrics,
            risk_metrics=risk_metrics,
            regime_performance=regime_performance,
            validation_windows=[],  # Would be filled by walk-forward validation
            overall_score=overall_score
        )
        
        self.benchmark_results.append(result)
        return result
    
    def _calculate_overall_score(self, 
                               eval_metrics: Dict[str, float],
                               trading_metrics: Dict[str, float],
                               risk_metrics: Dict[str, float]) -> float:
        """Calculate weighted overall performance score"""
        
        # Normalize metrics to 0-1 scale
        normalized_scores = []
        
        # Evaluation metrics (lower is better for error metrics)
        normalized_scores.append(max(0, 1 - eval_metrics['mape'] / 100))  # MAPE penalty
        normalized_scores.append(max(0, eval_metrics['r2']))  # R2 bonus
        
        # Trading metrics
        normalized_scores.append(min(1, max(0, trading_metrics['sharpe_ratio'] / 2)))  # Sharpe ratio
        normalized_scores.append(min(1, trading_metrics['win_rate']))  # Win rate
        normalized_scores.append(max(0, 1 + risk_metrics['max_drawdown'] / 0.5))  # Max drawdown penalty
        
        # Weighted average
        weights = [0.2, 0.2, 0.3, 0.2, 0.1]  # Emphasize trading performance
        overall_score = np.average(normalized_scores, weights=weights)
        
        return overall_score
    
    def generate_comparison_report(self) -> str:
        """Generate comprehensive comparison report"""
        
        if not self.benchmark_results:
            return "No benchmark results available for comparison."
        
        report = []
        report.append("=" * 80)
        report.append("MODEL BENCHMARK COMPARISON REPORT")
        report.append("=" * 80)
        
        # Sort by overall score
        sorted_results = sorted(self.benchmark_results, key=lambda x: x.overall_score, reverse=True)
        
        # Summary table
        report.append("\nOVERALL RANKING:")
        report.append(f"{'Rank':<5} {'Model':<25} {'Dataset':<15} {'Score':<8} {'Sharpe':<8} {'Win Rate':<10}")
        report.append("-" * 80)
        
        for i, result in enumerate(sorted_results):
            report.append(f"{i+1:<5} {result.model_name:<25} {result.dataset_type:<15} "
                         f"{result.overall_score:.3f}    {result.trading_metrics['sharpe_ratio']:.3f}    "
                         f"{result.trading_metrics['win_rate']:.2%}")
        
        # Detailed comparison
        report.append("\nDETAILED PERFORMANCE COMPARISON:")
        
        for result in sorted_results[:3]:  # Top 3 models
            report.append(f"\n{result.model_name} ({result.dataset_type}):")
            report.append(f"  Overall Score: {result.overall_score:.3f}")
            report.append(f"  MAPE: {result.evaluation_metrics['mape']:.2f}%")
            report.append(f"  R²: {result.evaluation_metrics['r2']:.3f}")
            report.append(f"  Sharpe Ratio: {result.trading_metrics['sharpe_ratio']:.3f}")
            report.append(f"  Win Rate: {result.trading_metrics['win_rate']:.2%}")
            report.append(f"  Max Drawdown: {result.trading_metrics['max_drawdown']:.2%}")
            report.append(f"  Annual Return: {result.trading_metrics['annualized_return']:.2%}")
        
        # Best model analysis
        best_model = sorted_results[0]
        report.append(f"\nBEST PERFORMING MODEL: {best_model.model_name}")
        report.append(f"Dataset: {best_model.dataset_type}")
        report.append(f"Key Strengths:")
        
        if best_model.trading_metrics['sharpe_ratio'] > 1.0:
            report.append("  • Excellent risk-adjusted returns (Sharpe > 1.0)")
        if best_model.trading_metrics['win_rate'] > 0.6:
            report.append("  • High win rate (>60%)")
        if abs(best_model.trading_metrics['max_drawdown']) < 0.1:
            report.append("  • Low maximum drawdown (<10%)")
        
        return "\n".join(report)
    
    def save_benchmark_results(self, output_dir: str = "./results/benchmarks/"):
        """Save benchmark results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results as JSON
        results_data = []
        for result in self.benchmark_results:
            result_dict = {
                'model_name': result.model_name,
                'dataset_type': result.dataset_type,
                'overall_score': result.overall_score,
                'evaluation_metrics': result.evaluation_metrics,
                'trading_metrics': result.trading_metrics,
                'risk_metrics': result.risk_metrics,
                'regime_performance': result.regime_performance
            }
            results_data.append(result_dict)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save comparison report
        report = self.generate_comparison_report()
        report_file = os.path.join(output_dir, f"benchmark_report_{timestamp}.txt")
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Benchmark results saved to {output_dir}")
        return results_file, report_file


class ValidationManager:
    """Main manager for comprehensive validation and evaluation"""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.validator = TimeSeriesValidator(self.config)
        self.analyzer = TradingPerformanceAnalyzer(self.config)
        self.comparator = BenchmarkComparator(self.config)
    
    def comprehensive_evaluation(self,
                                model_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                data: pd.DataFrame,
                                target_col: str = 'close',
                                timestamp_col: str = 'timestamp',
                                regime_col: str = None) -> Dict[str, BenchmarkResult]:
        """
        Run comprehensive evaluation across multiple models/configurations
        
        Args:
            model_predictions: Dict of {model_name: (predictions, actual)} pairs
            data: Original dataset with timestamps and regime info
            target_col: Target column name
            timestamp_col: Timestamp column name
            regime_col: Market regime column name (optional)
            
        Returns:
            Dictionary of benchmark results by model name
        """
        
        logger.info("Starting comprehensive evaluation across all models")
        
        results = {}
        timestamps = pd.to_datetime(data[timestamp_col]) if timestamp_col in data.columns else None
        regimes = data[regime_col].tolist() if regime_col and regime_col in data.columns else None
        
        for model_name, (predictions, actual) in model_predictions.items():
            logger.info(f"Evaluating {model_name}")
            
            # Determine dataset type from model name
            if 'EXTERNAL' in model_name.upper():
                dataset_type = 'CRYPTEX_EXTERNAL'
            elif 'MULTISCALE' in model_name.upper():
                dataset_type = 'CRYPTEX_MULTISCALE'
            elif 'ENHANCED' in model_name.upper():
                dataset_type = 'CRYPTEX_ENHANCED'
            else:
                dataset_type = 'CRYPTEX_BASIC'
            
            # Add to benchmark comparison
            result = self.comparator.add_benchmark_result(
                model_name=model_name,
                dataset_type=dataset_type,
                predictions=predictions,
                actual=actual,
                timestamps=timestamps,
                regimes=regimes
            )
            
            results[model_name] = result
        
        return results
    
    def generate_final_report(self, output_dir: str = "./results/validation/") -> str:
        """Generate and save final validation report"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate comparison report
        comparison_report = self.comparator.generate_comparison_report()
        
        # Save results
        results_file, report_file = self.comparator.save_benchmark_results(output_dir)
        
        logger.info(f"Final validation report saved to {report_file}")
        return report_file


# Convenience functions for easy integration
def quick_evaluation(predictions: np.ndarray, 
                    actual: np.ndarray,
                    model_name: str = "Model") -> Dict[str, float]:
    """Quick evaluation with basic metrics"""
    analyzer = TradingPerformanceAnalyzer()
    return analyzer.analyze_trading_performance(predictions, actual)


def compare_models(model_results: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> str:
    """Quick model comparison"""
    comparator = BenchmarkComparator()
    
    for model_name, (predictions, actual) in model_results.items():
        comparator.add_benchmark_result(model_name, "CRYPTEX", predictions, actual)
    
    return comparator.generate_comparison_report()


if __name__ == "__main__":
    print("Validation & Evaluation System for Cryptocurrency Time Series Prediction")
    print("Use ValidationManager for comprehensive evaluation across multiple models")