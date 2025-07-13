#!/usr/bin/env python3
"""
Main Hyperparameter Optimization Script for TimeLLM

This script provides a complete interface for running hyperparameter optimization
on TimeLLM cryptocurrency prediction models with various optimization strategies.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.hyperparameter_optimization import (
        HyperparameterOptimizer, 
        OptimizationConfig, 
        HyperparameterSpace,
        quick_optimize,
        comprehensive_optimize,
        architecture_search
    )
    HPO_AVAILABLE = True
except ImportError as e:
    print(f"Hyperparameter optimization not available: {e}")
    HPO_AVAILABLE = False

try:
    from utils.cross_validation_hpo import CrossValidationHPO, CrossValidationConfig
    CV_HPO_AVAILABLE = True
except ImportError:
    CV_HPO_AVAILABLE = False

try:
    from utils.optimization_analysis import OptimizationAnalyzer, analyze_optimization_results
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> Optional[pd.DataFrame]:
    """Load cryptocurrency data for optimization"""
    
    if not PANDAS_AVAILABLE:
        logger.error("Pandas not available - cannot load data")
        return None
    
    try:
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            data = pd.read_json(data_path)
        else:
            logger.error(f"Unsupported data format: {data_path}")
            return None
        
        logger.info(f"Loaded data with {len(data)} rows and {len(data.columns)} columns")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load data from {data_path}: {e}")
        return None


def run_quick_optimization(args) -> Dict[str, Any]:
    """Run quick hyperparameter optimization for immediate results"""
    
    logger.info("Running quick hyperparameter optimization")
    
    if not HPO_AVAILABLE:
        logger.error("Hyperparameter optimization framework not available")
        return {}
    
    # Configure datasets
    datasets = args.datasets if args.datasets else ["CRYPTEX_ENHANCED"]
    
    try:
        results = quick_optimize(
            datasets=datasets,
            n_trials=args.n_trials
        )
        
        logger.info(f"Quick optimization completed. Best value: {results.get('best_value')}")
        return results
        
    except Exception as e:
        logger.error(f"Quick optimization failed: {e}")
        return {}


def run_comprehensive_optimization(args) -> Dict[str, Any]:
    """Run comprehensive hyperparameter optimization"""
    
    logger.info("Running comprehensive hyperparameter optimization")
    
    if not HPO_AVAILABLE:
        logger.error("Hyperparameter optimization framework not available")
        return {}
    
    # Configure datasets
    datasets = args.datasets if args.datasets else ["CRYPTEX_ENHANCED", "CRYPTEX_EXTERNAL"]
    
    try:
        results = comprehensive_optimize(
            datasets=datasets,
            n_trials=args.n_trials
        )
        
        logger.info(f"Comprehensive optimization completed. Best value: {results.get('best_value')}")
        return results
        
    except Exception as e:
        logger.error(f"Comprehensive optimization failed: {e}")
        return {}


def run_architecture_search(args) -> Dict[str, Any]:
    """Run architecture-focused hyperparameter search"""
    
    logger.info("Running architecture-focused hyperparameter search")
    
    if not HPO_AVAILABLE:
        logger.error("Hyperparameter optimization framework not available")
        return {}
    
    # Configure datasets
    datasets = args.datasets if args.datasets else ["CRYPTEX_ENHANCED"]
    
    try:
        results = architecture_search(
            datasets=datasets,
            n_trials=args.n_trials
        )
        
        logger.info(f"Architecture search completed. Best value: {results.get('best_value')}")
        return results
        
    except Exception as e:
        logger.error(f"Architecture search failed: {e}")
        return {}


def run_custom_optimization(args) -> Dict[str, Any]:
    """Run custom hyperparameter optimization with user-defined parameters"""
    
    logger.info("Running custom hyperparameter optimization")
    
    if not HPO_AVAILABLE:
        logger.error("Hyperparameter optimization framework not available")
        return {}
    
    # Create custom configuration
    config = OptimizationConfig(
        method=args.method,
        n_trials=args.n_trials,
        primary_metric=args.primary_metric,
        direction="minimize" if args.primary_metric in ["mae", "mse", "mape"] else "maximize",
        datasets=args.datasets or ["CRYPTEX_ENHANCED"],
        study_name=args.study_name or f"Custom_HPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        output_dir=args.output_dir or "./results/hyperparameter_optimization/",
        enable_pruning=args.enable_pruning,
        max_concurrent_trials=args.max_concurrent_trials
    )
    
    try:
        optimizer = HyperparameterOptimizer(config)
        results = optimizer.optimize(
            optimization_type=args.optimization_type,
            datasets=config.datasets,
            n_trials=config.n_trials
        )
        
        logger.info(f"Custom optimization completed. Best value: {results.get('best_value')}")
        return results
        
    except Exception as e:
        logger.error(f"Custom optimization failed: {e}")
        return {}


def run_cv_optimization(args) -> Dict[str, Any]:
    """Run hyperparameter optimization with cross-validation"""
    
    logger.info("Running cross-validated hyperparameter optimization")
    
    if not CV_HPO_AVAILABLE:
        logger.error("Cross-validation HPO framework not available")
        return {}
    
    # Load data for CV
    if args.data_path:
        data = load_data(args.data_path)
        if data is None:
            logger.error("Failed to load data for CV optimization")
            return {}
    else:
        logger.error("Data path required for CV optimization")
        return {}
    
    # Configure CV
    cv_config = CrossValidationConfig(
        cv_strategy=args.cv_strategy,
        n_splits=args.cv_splits,
        initial_train_size=min(1000, len(data) // 2),
        test_size=min(200, len(data) // 10)
    )
    
    try:
        cv_hpo = CrossValidationHPO(cv_config)
        
        # Generate parameter combinations to test
        search_space = HyperparameterSpace(args.optimization_type)
        
        # For demonstration, create a few parameter combinations
        parameter_combinations = []
        for _ in range(min(args.n_trials, 10)):  # Limit for CV
            params = {}
            for param_name, param_config in search_space.space.items():
                if param_config['type'] == 'categorical':
                    params[param_name] = param_config['choices'][0]
                elif param_config['type'] == 'int':
                    params[param_name] = param_config['low']
                elif param_config['type'] == 'float':
                    params[param_name] = param_config['low']
                elif param_config['type'] == 'loguniform':
                    params[param_name] = param_config['low']
            
            parameter_combinations.append(params)
        
        best_cv_result = None
        best_score = float('inf')
        
        for params in parameter_combinations:
            cv_result = cv_hpo.cross_validate_parameters(data, params)
            
            # Get primary metric score
            score = cv_result.aggregated_metrics.get(args.primary_metric, float('inf'))
            
            if score < best_score:
                best_score = score
                best_cv_result = cv_result
        
        results = {
            "method": "cross_validation",
            "best_value": best_score,
            "best_cv_result": best_cv_result,
            "n_trials": len(parameter_combinations)
        }
        
        logger.info(f"CV optimization completed. Best CV score: {best_score}")
        return results
        
    except Exception as e:
        logger.error(f"CV optimization failed: {e}")
        return {}


def analyze_results(args) -> None:
    """Analyze existing optimization results"""
    
    logger.info("Analyzing optimization results")
    
    if not ANALYSIS_AVAILABLE:
        logger.error("Analysis framework not available")
        return
    
    try:
        analyzer = analyze_optimization_results(args.results_dir)
        
        if not analyzer.optimization_results:
            logger.warning("No optimization results found")
            return
        
        # Generate comprehensive report
        output_file = os.path.join(args.results_dir, "analysis_report.txt")
        report = analyzer.generate_comprehensive_report(output_file=output_file)
        
        print("\n" + "=" * 60)
        print("OPTIMIZATION ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Print key findings
        lines = report.split('\n')
        for line in lines[:50]:  # First 50 lines
            print(line)
        
        if len(lines) > 50:
            print(f"\n... (truncated, full report saved to {output_file})")
        
        # Create visualizations if requested
        if args.create_plots:
            for result in analyzer.optimization_results:
                plots = analyzer.create_optimization_visualizations(result)
                if plots:
                    logger.info(f"Created plots for {result.study_name}: {list(plots.keys())}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")


def main():
    """Main function for hyperparameter optimization"""
    
    parser = argparse.ArgumentParser(description='TimeLLM Hyperparameter Optimization')
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                       choices=['quick', 'comprehensive', 'architecture', 'custom', 'cv', 'analyze'],
                       help='Optimization mode')
    
    # General parameters
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of optimization trials')
    
    parser.add_argument('--datasets', nargs='+', 
                       choices=['CRYPTEX', 'CRYPTEX_ENHANCED', 'CRYPTEX_EXTERNAL', 
                               'CRYPTEX_MULTISCALE', 'CRYPTEX_REGIME_AWARE'],
                       help='Datasets to use for optimization')
    
    parser.add_argument('--output_dir', type=str, default='./results/hyperparameter_optimization/',
                       help='Output directory for results')
    
    # Custom optimization parameters
    parser.add_argument('--method', type=str, default='optuna',
                       choices=['optuna', 'ray', 'grid', 'random'],
                       help='Optimization method')
    
    parser.add_argument('--optimization_type', type=str, default='comprehensive',
                       choices=['quick', 'comprehensive', 'architecture', 'training'],
                       help='Type of optimization to perform')
    
    parser.add_argument('--primary_metric', type=str, default='mae',
                       choices=['mae', 'mse', 'mape', 'sharpe_ratio', 'directional_accuracy'],
                       help='Primary metric to optimize')
    
    parser.add_argument('--study_name', type=str,
                       help='Name for the optimization study')
    
    parser.add_argument('--enable_pruning', action='store_true',
                       help='Enable early trial pruning')
    
    parser.add_argument('--max_concurrent_trials', type=int, default=4,
                       help='Maximum number of concurrent trials')
    
    # Cross-validation parameters
    parser.add_argument('--data_path', type=str,
                       help='Path to data file for CV optimization')
    
    parser.add_argument('--cv_strategy', type=str, default='time_series_split',
                       choices=['time_series_split', 'blocked', 'purged'],
                       help='Cross-validation strategy')
    
    parser.add_argument('--cv_splits', type=int, default=3,
                       help='Number of CV splits')
    
    # Analysis parameters
    parser.add_argument('--results_dir', type=str, default='./results/hyperparameter_optimization/',
                       help='Directory containing optimization results')
    
    parser.add_argument('--create_plots', action='store_true',
                       help='Create visualization plots during analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run optimization based on mode
    if args.mode == 'quick':
        results = run_quick_optimization(args)
    elif args.mode == 'comprehensive':
        results = run_comprehensive_optimization(args)
    elif args.mode == 'architecture':
        results = run_architecture_search(args)
    elif args.mode == 'custom':
        results = run_custom_optimization(args)
    elif args.mode == 'cv':
        results = run_cv_optimization(args)
    elif args.mode == 'analyze':
        analyze_results(args)
        return
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return
    
    # Print results summary
    if results:
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETED")
        print("=" * 60)
        print(f"Mode: {args.mode}")
        print(f"Method: {results.get('method', 'unknown')}")
        print(f"Trials: {results.get('n_trials', 0)}")
        print(f"Best Value: {results.get('best_value', 'N/A')}")
        
        if 'best_parameters' in results:
            print("\nBest Parameters:")
            for param, value in results['best_parameters'].items():
                print(f"  {param}: {value}")
        
        print(f"\nResults saved to: {args.output_dir}")
    else:
        print("Optimization failed or returned no results")


if __name__ == "__main__":
    main()