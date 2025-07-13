#!/usr/bin/env python3
"""
Optimization Result Analysis and Management for TimeLLM

This module provides comprehensive analysis, visualization, and management
of hyperparameter optimization results for TimeLLM cryptocurrency prediction models.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

warnings.filterwarnings('ignore')

# Optional dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from scipy import stats
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Complete optimization result with metadata"""
    
    # Basic information
    study_name: str
    method: str
    start_time: datetime
    end_time: datetime
    
    # Parameters and performance
    best_parameters: Dict[str, Any]
    best_value: float
    all_parameters: List[Dict[str, Any]]
    all_values: List[float]
    
    # Optimization metadata
    n_trials: int
    successful_trials: int
    failed_trials: int
    pruned_trials: int
    
    # Statistical information
    parameter_importance: Optional[Dict[str, float]] = None
    correlation_matrix: Optional[pd.DataFrame] = None
    convergence_analysis: Optional[Dict[str, Any]] = None
    
    # Performance analysis
    performance_distribution: Optional[Dict[str, Any]] = None
    pareto_frontier: Optional[List[Dict[str, Any]]] = None


class OptimizationAnalyzer:
    """Comprehensive analysis of hyperparameter optimization results"""
    
    def __init__(self, results_dir: str = "./results/hyperparameter_optimization/"):
        self.results_dir = results_dir
        self.optimization_results = []
        
        # Create output directories
        os.makedirs(os.path.join(results_dir, "analysis"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "reports"), exist_ok=True)
    
    def load_optimization_results(self, file_pattern: str = "*_results_*.json") -> List[OptimizationResult]:
        """Load optimization results from files"""
        
        import glob
        
        result_files = glob.glob(os.path.join(self.results_dir, file_pattern))
        loaded_results = []
        
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Convert to OptimizationResult
                result = self._convert_to_optimization_result(data, file_path)
                loaded_results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        
        self.optimization_results.extend(loaded_results)
        logger.info(f"Loaded {len(loaded_results)} optimization results")
        
        return loaded_results
    
    def _convert_to_optimization_result(self, data: Dict[str, Any], file_path: str) -> OptimizationResult:
        """Convert loaded data to OptimizationResult"""
        
        # Extract basic information
        study_name = data.get('study_name', os.path.basename(file_path))
        method = data.get('method', 'unknown')
        
        # Extract parameters and values
        best_parameters = data.get('best_parameters', {})
        best_value = data.get('best_value', float('inf'))
        
        # Extract trial history
        optimization_history = data.get('optimization_history', [])
        all_parameters = []
        all_values = []
        
        if 'all_results' in data:
            for result in data['all_results']:
                if hasattr(result, 'parameters') and hasattr(result, 'metrics'):
                    all_parameters.append(result.parameters)
                    all_values.append(result.metrics.get('mae', float('inf')))
        else:
            # Extract from optimization history
            for trial_num, value in optimization_history:
                all_values.append(value)
                # We don't have parameters for each trial in history
        
        # Calculate statistics
        n_trials = data.get('n_trials', len(all_values))
        successful_trials = len([v for v in all_values if not np.isnan(v) and v != float('inf')])
        failed_trials = len([v for v in all_values if np.isnan(v) or v == float('inf')])
        pruned_trials = max(0, n_trials - successful_trials - failed_trials)
        
        # Create result object
        result = OptimizationResult(
            study_name=study_name,
            method=method,
            start_time=datetime.now() - timedelta(hours=1),  # Placeholder
            end_time=datetime.now(),  # Placeholder
            best_parameters=best_parameters,
            best_value=best_value,
            all_parameters=all_parameters,
            all_values=all_values,
            n_trials=n_trials,
            successful_trials=successful_trials,
            failed_trials=failed_trials,
            pruned_trials=pruned_trials
        )
        
        return result
    
    def analyze_convergence(self, result: OptimizationResult) -> Dict[str, Any]:
        """Analyze optimization convergence"""
        
        values = [v for v in result.all_values if not np.isnan(v) and v != float('inf')]
        
        if len(values) < 3:
            return {"status": "insufficient_data"}
        
        # Calculate running minimum
        running_min = []
        current_min = float('inf')
        
        for value in values:
            current_min = min(current_min, value)
            running_min.append(current_min)
        
        # Analyze convergence characteristics
        analysis = {
            "status": "analyzed",
            "total_trials": len(values),
            "initial_value": values[0],
            "final_value": values[-1],
            "best_value": min(values),
            "improvement_ratio": (values[0] - min(values)) / values[0] if values[0] != 0 else 0,
            "convergence_trial": self._find_convergence_point(running_min),
            "plateau_length": self._calculate_plateau_length(running_min),
            "volatility": np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
        }
        
        # Convergence rate analysis
        if SCIPY_AVAILABLE:
            analysis["convergence_rate"] = self._fit_convergence_curve(running_min)
        
        return analysis
    
    def _find_convergence_point(self, running_min: List[float], threshold: float = 0.001) -> int:
        """Find the trial where optimization converged"""
        
        if len(running_min) < 10:
            return len(running_min)
        
        # Look for point where improvement becomes minimal
        final_value = running_min[-1]
        
        for i in range(len(running_min) - 10, 0, -1):
            improvement = (running_min[i] - final_value) / running_min[i] if running_min[i] != 0 else 0
            if improvement > threshold:
                return i + 10
        
        return len(running_min)
    
    def _calculate_plateau_length(self, running_min: List[float], threshold: float = 0.0001) -> int:
        """Calculate length of plateau at the end"""
        
        if len(running_min) < 2:
            return 0
        
        final_value = running_min[-1]
        plateau_length = 0
        
        for i in range(len(running_min) - 1, -1, -1):
            relative_diff = abs(running_min[i] - final_value) / final_value if final_value != 0 else 0
            if relative_diff <= threshold:
                plateau_length += 1
            else:
                break
        
        return plateau_length
    
    def _fit_convergence_curve(self, running_min: List[float]) -> Dict[str, float]:
        """Fit exponential convergence curve"""
        
        try:
            x = np.arange(len(running_min))
            y = np.array(running_min)
            
            # Fit exponential decay: y = a * exp(-b * x) + c
            def exp_decay(x, a, b, c):
                return a * np.exp(-b * x) + c
            
            # Initial guess
            p0 = [y[0] - y[-1], 0.1, y[-1]]
            
            popt, pcov = curve_fit(exp_decay, x, y, p0=p0, maxfev=1000)
            
            # Calculate R-squared
            y_pred = exp_decay(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                "decay_constant": popt[1],
                "asymptote": popt[2],
                "r_squared": r_squared,
                "converged": r_squared > 0.8
            }
            
        except Exception as e:
            logger.warning(f"Convergence curve fitting failed: {e}")
            return {"decay_constant": 0, "asymptote": running_min[-1], "r_squared": 0, "converged": False}
    
    def analyze_parameter_importance(self, result: OptimizationResult) -> Dict[str, float]:
        """Analyze parameter importance using correlation with objective"""
        
        if not result.all_parameters or len(result.all_parameters) < 10:
            return {}
        
        # Create parameter matrix
        param_names = list(result.all_parameters[0].keys())
        param_matrix = []
        valid_values = []
        
        for i, params in enumerate(result.all_parameters):
            if i < len(result.all_values) and not np.isnan(result.all_values[i]):
                row = []
                valid = True
                
                for param_name in param_names:
                    value = params.get(param_name)
                    if isinstance(value, (int, float)):
                        row.append(value)
                    elif isinstance(value, str):
                        # Convert categorical to numeric
                        row.append(hash(value) % 1000)  # Simple hash-based encoding
                    else:
                        valid = False
                        break
                
                if valid:
                    param_matrix.append(row)
                    valid_values.append(result.all_values[i])
        
        if len(param_matrix) < 5:
            return {}
        
        # Calculate correlations
        param_matrix = np.array(param_matrix)
        valid_values = np.array(valid_values)
        
        importance = {}
        
        for i, param_name in enumerate(param_names):
            param_values = param_matrix[:, i]
            
            # Skip if parameter is constant
            if np.std(param_values) == 0:
                importance[param_name] = 0.0
                continue
            
            # Calculate correlation with objective
            correlation = np.corrcoef(param_values, valid_values)[0, 1]
            importance[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
        
        return importance
    
    def create_parameter_correlation_matrix(self, result: OptimizationResult) -> Optional[pd.DataFrame]:
        """Create correlation matrix for parameters"""
        
        if not result.all_parameters or len(result.all_parameters) < 10:
            return None
        
        # Convert parameters to DataFrame
        param_df = pd.DataFrame(result.all_parameters)
        
        # Select only numeric columns
        numeric_columns = param_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) < 2:
            return None
        
        correlation_matrix = param_df[numeric_columns].corr()
        return correlation_matrix
    
    def identify_pareto_frontier(self, results: List[OptimizationResult], 
                                metrics: List[str] = ["mae", "training_time"]) -> List[Dict[str, Any]]:
        """Identify Pareto frontier for multi-objective optimization"""
        
        pareto_points = []
        
        for result in results:
            # Extract metrics (assuming we have access to training time)
            point = {
                "parameters": result.best_parameters,
                "mae": result.best_value,
                "training_time": 100.0,  # Placeholder
                "study_name": result.study_name
            }
            pareto_points.append(point)
        
        # Find Pareto frontier
        pareto_frontier = []
        
        for i, point_a in enumerate(pareto_points):
            is_dominated = False
            
            for j, point_b in enumerate(pareto_points):
                if i != j:
                    # Check if point_a is dominated by point_b
                    dominates = True
                    for metric in metrics:
                        if metric == "mae":  # Lower is better
                            if point_a[metric] < point_b[metric]:
                                dominates = False
                                break
                        else:  # Assume higher is better for other metrics
                            if point_a[metric] > point_b[metric]:
                                dominates = False
                                break
                    
                    if dominates:
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_frontier.append(point_a)
        
        return pareto_frontier
    
    def generate_performance_distribution_analysis(self, result: OptimizationResult) -> Dict[str, Any]:
        """Analyze distribution of performance across trials"""
        
        values = [v for v in result.all_values if not np.isnan(v) and v != float('inf')]
        
        if len(values) < 3:
            return {"status": "insufficient_data"}
        
        analysis = {
            "status": "analyzed",
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "q25": np.percentile(values, 25),
            "q75": np.percentile(values, 75),
            "iqr": np.percentile(values, 75) - np.percentile(values, 25),
            "skewness": stats.skew(values) if SCIPY_AVAILABLE else 0,
            "kurtosis": stats.kurtosis(values) if SCIPY_AVAILABLE else 0
        }
        
        # Performance categories
        best_10_percent = np.percentile(values, 10)
        worst_10_percent = np.percentile(values, 90)
        
        analysis["performance_categories"] = {
            "excellent": len([v for v in values if v <= best_10_percent]),
            "good": len([v for v in values if best_10_percent < v <= np.percentile(values, 25)]),
            "average": len([v for v in values if np.percentile(values, 25) < v <= np.percentile(values, 75)]),
            "poor": len([v for v in values if np.percentile(values, 75) < v <= worst_10_percent]),
            "very_poor": len([v for v in values if v > worst_10_percent])
        }
        
        return analysis
    
    def create_optimization_visualizations(self, result: OptimizationResult, output_dir: str = None) -> Dict[str, str]:
        """Create comprehensive visualizations of optimization results"""
        
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available - skipping visualizations")
            return {}
        
        if output_dir is None:
            output_dir = os.path.join(self.results_dir, "plots")
        
        os.makedirs(output_dir, exist_ok=True)
        
        created_plots = {}
        
        # 1. Convergence plot
        convergence_file = self._create_convergence_plot(result, output_dir)
        if convergence_file:
            created_plots["convergence"] = convergence_file
        
        # 2. Parameter importance plot
        importance_file = self._create_parameter_importance_plot(result, output_dir)
        if importance_file:
            created_plots["parameter_importance"] = importance_file
        
        # 3. Performance distribution plot
        distribution_file = self._create_performance_distribution_plot(result, output_dir)
        if distribution_file:
            created_plots["performance_distribution"] = distribution_file
        
        # 4. Parameter correlation heatmap
        correlation_file = self._create_correlation_heatmap(result, output_dir)
        if correlation_file:
            created_plots["parameter_correlation"] = correlation_file
        
        return created_plots
    
    def _create_convergence_plot(self, result: OptimizationResult, output_dir: str) -> Optional[str]:
        """Create convergence plot"""
        
        values = [v for v in result.all_values if not np.isnan(v) and v != float('inf')]
        
        if len(values) < 3:
            return None
        
        plt.figure(figsize=(12, 6))
        
        # Running minimum
        running_min = []
        current_min = float('inf')
        for value in values:
            current_min = min(current_min, value)
            running_min.append(current_min)
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(values) + 1), values, 'b-', alpha=0.6, label='Trial values')
        plt.plot(range(1, len(running_min) + 1), running_min, 'r-', linewidth=2, label='Best value')
        plt.xlabel('Trial Number')
        plt.ylabel('Objective Value')
        plt.title('Optimization Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Log scale convergence
        plt.subplot(1, 2, 2)
        plt.semilogy(range(1, len(running_min) + 1), running_min, 'r-', linewidth=2)
        plt.xlabel('Trial Number')
        plt.ylabel('Best Value (log scale)')
        plt.title('Convergence (Log Scale)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f"{result.study_name}_convergence.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _create_parameter_importance_plot(self, result: OptimizationResult, output_dir: str) -> Optional[str]:
        """Create parameter importance plot"""
        
        importance = self.analyze_parameter_importance(result)
        
        if not importance:
            return None
        
        # Sort by importance
        sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        plt.figure(figsize=(10, 6))
        
        params, values = zip(*sorted_params)
        
        bars = plt.bar(range(len(params)), values, color='skyblue', alpha=0.7)
        plt.xlabel('Parameters')
        plt.ylabel('Importance (|Correlation|)')
        plt.title('Parameter Importance Analysis')
        plt.xticks(range(len(params)), params, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f"{result.study_name}_parameter_importance.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _create_performance_distribution_plot(self, result: OptimizationResult, output_dir: str) -> Optional[str]:
        """Create performance distribution plot"""
        
        values = [v for v in result.all_values if not np.isnan(v) and v != float('inf')]
        
        if len(values) < 3:
            return None
        
        plt.figure(figsize=(12, 8))
        
        # Histogram
        plt.subplot(2, 2, 1)
        plt.hist(values, bins=min(30, len(values) // 3), alpha=0.7, color='lightblue', edgecolor='black')
        plt.axvline(result.best_value, color='red', linestyle='--', linewidth=2, label=f'Best: {result.best_value:.4f}')
        plt.xlabel('Objective Value')
        plt.ylabel('Frequency')
        plt.title('Performance Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(2, 2, 2)
        plt.boxplot(values, vert=True)
        plt.ylabel('Objective Value')
        plt.title('Performance Box Plot')
        plt.grid(True, alpha=0.3)
        
        # Q-Q plot
        if SCIPY_AVAILABLE:
            plt.subplot(2, 2, 3)
            stats.probplot(values, dist="norm", plot=plt)
            plt.title('Q-Q Plot (Normal Distribution)')
            plt.grid(True, alpha=0.3)
        
        # Cumulative distribution
        plt.subplot(2, 2, 4)
        sorted_values = np.sort(values)
        cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        plt.plot(sorted_values, cumulative, 'b-', linewidth=2)
        plt.axvline(result.best_value, color='red', linestyle='--', linewidth=2, label=f'Best: {result.best_value:.4f}')
        plt.xlabel('Objective Value')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f"{result.study_name}_performance_distribution.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _create_correlation_heatmap(self, result: OptimizationResult, output_dir: str) -> Optional[str]:
        """Create parameter correlation heatmap"""
        
        correlation_matrix = self.create_parameter_correlation_matrix(result)
        
        if correlation_matrix is None or correlation_matrix.empty:
            return None
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        
        plt.title('Parameter Correlation Matrix')
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f"{result.study_name}_parameter_correlation.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def generate_comprehensive_report(self, results: List[OptimizationResult] = None, 
                                    output_file: str = None) -> str:
        """Generate comprehensive analysis report"""
        
        if results is None:
            results = self.optimization_results
        
        if not results:
            return "No optimization results available for analysis."
        
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE HYPERPARAMETER OPTIMIZATION ANALYSIS")
        report.append("=" * 80)
        
        # Summary statistics
        report.append(f"\nTOTAL OPTIMIZATION STUDIES: {len(results)}")
        report.append(f"TOTAL TRIALS ACROSS ALL STUDIES: {sum(r.n_trials for r in results)}")
        
        # Best results across all studies
        best_result = min(results, key=lambda r: r.best_value)
        report.append(f"\nOVERALL BEST PERFORMANCE:")
        report.append(f"Study: {best_result.study_name}")
        report.append(f"Method: {best_result.method}")
        report.append(f"Best Value: {best_result.best_value:.6f}")
        report.append(f"Parameters:")
        for param, value in best_result.best_parameters.items():
            report.append(f"  {param}: {value}")
        
        # Analysis for each study
        for result in results:
            report.append(f"\n{'-' * 60}")
            report.append(f"STUDY: {result.study_name}")
            report.append(f"{'-' * 60}")
            
            # Basic statistics
            report.append(f"Method: {result.method}")
            report.append(f"Total Trials: {result.n_trials}")
            report.append(f"Successful: {result.successful_trials}")
            report.append(f"Failed: {result.failed_trials}")
            report.append(f"Pruned: {result.pruned_trials}")
            report.append(f"Success Rate: {result.successful_trials / result.n_trials * 100:.1f}%")
            
            # Convergence analysis
            convergence = self.analyze_convergence(result)
            if convergence["status"] == "analyzed":
                report.append(f"\nCONVERGENCE ANALYSIS:")
                report.append(f"Improvement Ratio: {convergence['improvement_ratio']:.2%}")
                report.append(f"Convergence Trial: {convergence['convergence_trial']}")
                report.append(f"Plateau Length: {convergence['plateau_length']}")
                report.append(f"Volatility: {convergence['volatility']:.4f}")
            
            # Parameter importance
            importance = self.analyze_parameter_importance(result)
            if importance:
                report.append(f"\nTOP PARAMETER IMPORTANCE:")
                sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                for param, imp in sorted_importance[:5]:
                    report.append(f"  {param}: {imp:.4f}")
            
            # Performance distribution
            perf_dist = self.generate_performance_distribution_analysis(result)
            if perf_dist["status"] == "analyzed":
                report.append(f"\nPERFORMANCE STATISTICS:")
                report.append(f"Mean: {perf_dist['mean']:.6f}")
                report.append(f"Median: {perf_dist['median']:.6f}")
                report.append(f"Std: {perf_dist['std']:.6f}")
                report.append(f"Min: {perf_dist['min']:.6f}")
                report.append(f"Max: {perf_dist['max']:.6f}")
        
        # Cross-study comparison
        if len(results) > 1:
            report.append(f"\n{'-' * 60}")
            report.append("CROSS-STUDY COMPARISON")
            report.append(f"{'-' * 60}")
            
            # Method comparison
            method_performance = {}
            for result in results:
                method = result.method
                if method not in method_performance:
                    method_performance[method] = []
                method_performance[method].append(result.best_value)
            
            report.append("\nMETHOD PERFORMANCE:")
            for method, values in method_performance.items():
                report.append(f"{method}: {np.mean(values):.6f} ± {np.std(values):.6f} (n={len(values)})")
        
        # Recommendations
        report.append(f"\n{'-' * 60}")
        report.append("RECOMMENDATIONS")
        report.append(f"{'-' * 60}")
        
        report.append("1. Use the overall best parameters for final model training")
        report.append("2. Consider ensemble methods combining top-performing configurations")
        report.append("3. Focus future optimization on parameters with highest importance")
        
        if best_result.successful_trials / best_result.n_trials < 0.8:
            report.append("4. ⚠️  Consider relaxing parameter constraints - high failure rate detected")
        
        # Save report
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_file}")
        
        return report_text


# Convenience functions
def analyze_optimization_results(results_dir: str = "./results/hyperparameter_optimization/") -> OptimizationAnalyzer:
    """Analyze all optimization results in directory"""
    
    analyzer = OptimizationAnalyzer(results_dir)
    analyzer.load_optimization_results()
    
    return analyzer


def quick_analysis_report(results_dir: str = "./results/hyperparameter_optimization/") -> str:
    """Generate quick analysis report"""
    
    analyzer = analyze_optimization_results(results_dir)
    
    if not analyzer.optimization_results:
        return "No optimization results found in directory."
    
    return analyzer.generate_comprehensive_report()


if __name__ == "__main__":
    print("Hyperparameter Optimization Analysis for TimeLLM")
    print(f"Plotting available: {PLOTTING_AVAILABLE}")
    print(f"Plotly available: {PLOTLY_AVAILABLE}")
    print(f"SciPy available: {SCIPY_AVAILABLE}")
    
    # Example usage
    print("\nExample analysis functionality available:")
    print("• Convergence analysis")
    print("• Parameter importance analysis")
    print("• Performance distribution analysis")
    print("• Cross-study comparison")
    print("• Comprehensive reporting")
    print("• Visualization creation")