#!/usr/bin/env python3
"""
Comprehensive Data Quality Enhancement System for Time-LLM-Cryptex

This module provides advanced data validation, quality scoring, profiling,
and monitoring capabilities for cryptocurrency time series data.
"""

import os
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import logging
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataQualityConfig:
    """Configuration for data quality parameters"""
    # Validation thresholds
    max_missing_ratio: float = 0.05  # Max 5% missing data
    max_outlier_ratio: float = 0.02  # Max 2% outliers
    min_data_points: int = 1000      # Minimum required data points
    
    # Price validation
    max_price_change: float = 0.5    # Max 50% price change
    min_volume: float = 0            # Minimum volume
    max_volume_spike: float = 10     # Max 10x volume spike
    
    # Time series validation
    max_gap_hours: int = 24          # Max gap in hours
    expected_frequency: str = 'H'    # Expected data frequency
    
    # External data validation
    max_external_lag_days: int = 7   # Max lag for external data
    min_correlation_threshold: float = 0.01  # Min correlation for usefulness
    
    # Quality scoring weights
    completeness_weight: float = 0.3
    consistency_weight: float = 0.25
    accuracy_weight: float = 0.25
    timeliness_weight: float = 0.2


@dataclass
class ValidationResult:
    """Result of data validation check"""
    check_name: str
    passed: bool
    score: float  # 0-1 score
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # info, warning, error, critical


@dataclass
class DataQualityReport:
    """Comprehensive data quality report"""
    timestamp: datetime
    overall_score: float
    validation_results: List[ValidationResult]
    data_profile: Dict[str, Any]
    recommendations: List[str]
    dataset_info: Dict[str, Any]


class DataValidator:
    """Comprehensive data validation system"""
    
    def __init__(self, config: DataQualityConfig = None):
        self.config = config or DataQualityConfig()
        self.validation_checks = []
        self._register_validation_checks()
    
    def _register_validation_checks(self):
        """Register all available validation checks"""
        self.validation_checks = [
            self._validate_completeness,
            self._validate_consistency,
            self._validate_price_integrity,
            self._validate_volume_integrity,
            self._validate_temporal_integrity,
            self._validate_ohlc_relationships,
            self._validate_statistical_properties,
            self._validate_external_data_alignment
        ]
    
    def validate_dataset(self, df: pd.DataFrame, 
                        dataset_type: str = "crypto") -> List[ValidationResult]:
        """Run all validation checks on dataset"""
        results = []
        
        logger.info(f"Running validation on {dataset_type} dataset with {len(df)} records")
        
        for check_func in self.validation_checks:
            try:
                result = check_func(df)
                results.append(result)
                
                if result.severity in ["error", "critical"]:
                    logger.error(f"Validation failed: {result.check_name} - {result.message}")
                elif result.severity == "warning":
                    logger.warning(f"Validation warning: {result.check_name} - {result.message}")
                    
            except Exception as e:
                error_result = ValidationResult(
                    check_name=check_func.__name__,
                    passed=False,
                    score=0.0,
                    message=f"Validation check failed with error: {str(e)}",
                    severity="error"
                )
                results.append(error_result)
                logger.error(f"Validation check {check_func.__name__} failed: {e}")
        
        return results
    
    def _validate_completeness(self, df: pd.DataFrame) -> ValidationResult:
        """Validate data completeness"""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        missing_ratio = missing_cells / total_cells if total_cells > 0 else 1.0
        
        passed = missing_ratio <= self.config.max_missing_ratio
        score = max(0, 1 - (missing_ratio / self.config.max_missing_ratio))
        
        # Analyze missing patterns by column
        missing_by_col = df.isnull().sum()
        problematic_cols = missing_by_col[missing_by_col > len(df) * 0.1].to_dict()
        
        severity = "info" if passed else ("warning" if missing_ratio < 0.1 else "error")
        
        return ValidationResult(
            check_name="Data Completeness",
            passed=passed,
            score=score,
            message=f"Missing data ratio: {missing_ratio:.3f} (threshold: {self.config.max_missing_ratio})",
            details={
                "total_cells": total_cells,
                "missing_cells": missing_cells,
                "missing_ratio": missing_ratio,
                "problematic_columns": problematic_cols
            },
            severity=severity
        )
    
    def _validate_consistency(self, df: pd.DataFrame) -> ValidationResult:
        """Validate data consistency"""
        issues = []
        score = 1.0
        
        # Check for duplicate timestamps
        if 'timestamp' in df.columns:
            duplicates = df['timestamp'].duplicated().sum()
            if duplicates > 0:
                issues.append(f"{duplicates} duplicate timestamps")
                score -= 0.3
        
        # Check for negative volumes
        if 'volume' in df.columns:
            negative_volumes = (df['volume'] < 0).sum()
            if negative_volumes > 0:
                issues.append(f"{negative_volumes} negative volume values")
                score -= 0.2
        
        # Check for zero prices
        price_cols = [col for col in df.columns if col in ['open', 'high', 'low', 'close']]
        for col in price_cols:
            zero_prices = (df[col] <= 0).sum()
            if zero_prices > 0:
                issues.append(f"{zero_prices} zero/negative prices in {col}")
                score -= 0.1
        
        score = max(0, score)
        passed = len(issues) == 0
        severity = "info" if passed else ("warning" if score > 0.5 else "error")
        
        return ValidationResult(
            check_name="Data Consistency",
            passed=passed,
            score=score,
            message="No consistency issues found" if passed else f"Found issues: {'; '.join(issues)}",
            details={"issues": issues},
            severity=severity
        )
    
    def _validate_price_integrity(self, df: pd.DataFrame) -> ValidationResult:
        """Validate price data integrity"""
        price_cols = [col for col in df.columns if col in ['open', 'high', 'low', 'close']]
        
        if not price_cols:
            return ValidationResult("Price Integrity", True, 1.0, "No price columns found", severity="info")
        
        issues = []
        score = 1.0
        
        for col in price_cols:
            if col in df.columns:
                # Check for extreme price changes
                price_changes = df[col].pct_change().abs()
                extreme_changes = (price_changes > self.config.max_price_change).sum()
                
                if extreme_changes > 0:
                    issues.append(f"{extreme_changes} extreme price changes (>{self.config.max_price_change*100}%) in {col}")
                    score -= 0.2
                
                # Check for price spikes (outliers using IQR method)
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR))).sum()
                
                if outliers > len(df) * 0.01:  # More than 1% outliers
                    issues.append(f"{outliers} price outliers detected in {col}")
                    score -= 0.1
        
        score = max(0, score)
        passed = len(issues) == 0
        severity = "info" if passed else ("warning" if score > 0.5 else "error")
        
        return ValidationResult(
            check_name="Price Integrity",
            passed=passed,
            score=score,
            message="Price data integrity validated" if passed else f"Issues found: {'; '.join(issues)}",
            details={"issues": issues},
            severity=severity
        )
    
    def _validate_volume_integrity(self, df: pd.DataFrame) -> ValidationResult:
        """Validate volume data integrity"""
        if 'volume' not in df.columns:
            return ValidationResult("Volume Integrity", True, 1.0, "No volume column found", severity="info")
        
        issues = []
        score = 1.0
        
        # Check for volume spikes
        volume_changes = df['volume'].pct_change().abs()
        volume_spikes = (volume_changes > self.config.max_volume_spike).sum()
        
        if volume_spikes > 0:
            issues.append(f"{volume_spikes} volume spikes (>{self.config.max_volume_spike}x change)")
            score -= 0.3
        
        # Check for suspiciously low volume periods
        median_volume = df['volume'].median()
        low_volume_periods = (df['volume'] < median_volume * 0.01).sum()
        
        if low_volume_periods > len(df) * 0.05:  # More than 5% of periods
            issues.append(f"{low_volume_periods} periods with suspiciously low volume")
            score -= 0.2
        
        # Check volume distribution (should not be too uniform)
        volume_std = df['volume'].std()
        volume_mean = df['volume'].mean()
        cv = volume_std / volume_mean if volume_mean > 0 else 0
        
        if cv < 0.1:  # Coefficient of variation too low
            issues.append("Volume distribution appears too uniform (possible data issue)")
            score -= 0.2
        
        score = max(0, score)
        passed = len(issues) == 0
        severity = "info" if passed else ("warning" if score > 0.5 else "error")
        
        return ValidationResult(
            check_name="Volume Integrity",
            passed=passed,
            score=score,
            message="Volume data integrity validated" if passed else f"Issues found: {'; '.join(issues)}",
            details={"issues": issues, "volume_cv": cv},
            severity=severity
        )
    
    def _validate_temporal_integrity(self, df: pd.DataFrame) -> ValidationResult:
        """Validate temporal data integrity"""
        if 'timestamp' not in df.columns:
            return ValidationResult("Temporal Integrity", False, 0.0, "No timestamp column found", severity="error")
        
        # Convert timestamp to datetime if needed
        timestamps = pd.to_datetime(df['timestamp'], unit='s' if df['timestamp'].dtype in ['int64', 'float64'] else None)
        
        issues = []
        score = 1.0
        
        # Check for time gaps
        time_diffs = timestamps.diff().dt.total_seconds() / 3600  # Convert to hours
        max_gap = time_diffs.max()
        
        if max_gap > self.config.max_gap_hours:
            large_gaps = (time_diffs > self.config.max_gap_hours).sum()
            issues.append(f"{large_gaps} time gaps larger than {self.config.max_gap_hours} hours (max: {max_gap:.1f}h)")
            score -= 0.3
        
        # Check for temporal ordering
        if not timestamps.is_monotonic_increasing:
            issues.append("Timestamps are not in chronological order")
            score -= 0.4
        
        # Check frequency consistency
        if len(timestamps) > 2:
            expected_freq = pd.infer_freq(timestamps)
            if expected_freq is None:
                issues.append("Cannot infer consistent frequency from timestamps")
                score -= 0.2
        
        score = max(0, score)
        passed = len(issues) == 0
        severity = "info" if passed else ("warning" if score > 0.5 else "error")
        
        return ValidationResult(
            check_name="Temporal Integrity",
            passed=passed,
            score=score,
            message="Temporal integrity validated" if passed else f"Issues found: {'; '.join(issues)}",
            details={
                "issues": issues,
                "max_gap_hours": max_gap,
                "total_timespan_days": (timestamps.max() - timestamps.min()).days
            },
            severity=severity
        )
    
    def _validate_ohlc_relationships(self, df: pd.DataFrame) -> ValidationResult:
        """Validate OHLC price relationships"""
        ohlc_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in ohlc_cols if col not in df.columns]
        
        if missing_cols:
            return ValidationResult(
                "OHLC Relationships", 
                True, 
                1.0, 
                f"OHLC validation skipped - missing columns: {missing_cols}", 
                severity="info"
            )
        
        issues = []
        score = 1.0
        
        # High should be >= max(open, close)
        high_violations = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
        if high_violations > 0:
            issues.append(f"{high_violations} cases where high < max(open, close)")
            score -= 0.3
        
        # Low should be <= min(open, close)
        low_violations = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
        if low_violations > 0:
            issues.append(f"{low_violations} cases where low > min(open, close)")
            score -= 0.3
        
        # High should be >= low
        hl_violations = (df['high'] < df['low']).sum()
        if hl_violations > 0:
            issues.append(f"{hl_violations} cases where high < low")
            score -= 0.4
        
        score = max(0, score)
        passed = len(issues) == 0
        severity = "info" if passed else ("warning" if score > 0.5 else "error")
        
        return ValidationResult(
            check_name="OHLC Relationships",
            passed=passed,
            score=score,
            message="OHLC relationships validated" if passed else f"Violations found: {'; '.join(issues)}",
            details={"issues": issues},
            severity=severity
        )
    
    def _validate_statistical_properties(self, df: pd.DataFrame) -> ValidationResult:
        """Validate statistical properties of the data"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        issues = []
        score = 1.0
        
        for col in numeric_cols:
            # Check for constant values (no variance)
            if df[col].nunique() == 1:
                issues.append(f"Column {col} has constant values (no variance)")
                score -= 0.2
                continue
            
            # Check for extreme skewness
            skewness = abs(df[col].skew())
            if skewness > 5:  # Very skewed
                issues.append(f"Column {col} is highly skewed (skewness: {skewness:.2f})")
                score -= 0.1
            
            # Check for infinite values
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                issues.append(f"Column {col} contains {inf_count} infinite values")
                score -= 0.3
        
        score = max(0, score)
        passed = len(issues) == 0
        severity = "info" if passed else ("warning" if score > 0.7 else "error")
        
        return ValidationResult(
            check_name="Statistical Properties",
            passed=passed,
            score=score,
            message="Statistical properties validated" if passed else f"Issues found: {'; '.join(issues)}",
            details={"issues": issues},
            severity=severity
        )
    
    def _validate_external_data_alignment(self, df: pd.DataFrame) -> ValidationResult:
        """Validate external data alignment and relevance"""
        # Identify external data columns
        external_cols = []
        for col in df.columns:
            if any(suffix in col for suffix in ['_sentiment', '_macro', '_onchain']):
                external_cols.append(col)
        
        if not external_cols:
            return ValidationResult(
                "External Data Alignment", 
                True, 
                1.0, 
                "No external data columns found", 
                severity="info"
            )
        
        issues = []
        score = 1.0
        
        # Check for excessive missing data in external columns
        for col in external_cols:
            missing_ratio = df[col].isnull().sum() / len(df)
            if missing_ratio > 0.2:  # More than 20% missing
                issues.append(f"External column {col} has {missing_ratio:.1%} missing data")
                score -= 0.1
        
        # Check correlation with target (if available)
        target_cols = [col for col in df.columns if col in ['close', 'target']]
        if target_cols:
            target_col = target_cols[0]
            low_corr_cols = []
            
            for col in external_cols:
                if df[col].dtype in [np.number] and not df[col].isnull().all():
                    corr = abs(df[col].corr(df[target_col]))
                    if not np.isnan(corr) and corr < self.config.min_correlation_threshold:
                        low_corr_cols.append((col, corr))
            
            if low_corr_cols:
                issues.append(f"{len(low_corr_cols)} external features have very low correlation with target")
                score -= 0.2
        
        score = max(0, score)
        passed = len(issues) == 0
        severity = "info" if passed else ("warning" if score > 0.6 else "error")
        
        return ValidationResult(
            check_name="External Data Alignment",
            passed=passed,
            score=score,
            message="External data alignment validated" if passed else f"Issues found: {'; '.join(issues)}",
            details={
                "issues": issues,
                "external_columns_count": len(external_cols)
            },
            severity=severity
        )


class DataProfiler:
    """Advanced data profiling and drift detection"""
    
    def __init__(self):
        self.baseline_profile = None
    
    def create_profile(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Dict[str, Any]:
        """Create comprehensive data profile"""
        profile = {
            "dataset_name": dataset_name,
            "creation_time": datetime.now().isoformat(),
            "basic_info": self._get_basic_info(df),
            "column_profiles": self._get_column_profiles(df),
            "correlations": self._get_correlation_analysis(df),
            "time_series_properties": self._get_time_series_properties(df),
            "data_quality_indicators": self._get_quality_indicators(df)
        }
        
        return profile
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information"""
        return {
            "shape": df.shape,
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "dtypes": df.dtypes.value_counts().to_dict(),
            "missing_cells": df.isnull().sum().sum(),
            "duplicate_rows": df.duplicated().sum()
        }
    
    def _get_column_profiles(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Get detailed profile for each column"""
        profiles = {}
        
        for col in df.columns:
            col_profile = {
                "dtype": str(df[col].dtype),
                "missing_count": df[col].isnull().sum(),
                "missing_percentage": df[col].isnull().sum() / len(df) * 100,
                "unique_count": df[col].nunique(),
                "unique_percentage": df[col].nunique() / len(df) * 100 if len(df) > 0 else 0
            }
            
            if df[col].dtype in [np.number]:
                col_profile.update({
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "median": df[col].median(),
                    "skewness": df[col].skew(),
                    "kurtosis": df[col].kurtosis(),
                    "outliers_iqr": self._count_outliers_iqr(df[col]),
                    "zeros_count": (df[col] == 0).sum()
                })
            
            profiles[col] = col_profile
        
        return profiles
    
    def _count_outliers_iqr(self, series: pd.Series) -> int:
        """Count outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()
        return outliers
    
    def _get_correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between features"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return {"message": "Insufficient numeric columns for correlation analysis"}
        
        corr_matrix = numeric_df.corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.8:  # High correlation threshold
                    high_corr_pairs.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": corr_val
                    })
        
        return {
            "correlation_matrix_shape": corr_matrix.shape,
            "high_correlation_pairs": high_corr_pairs,
            "max_correlation": corr_matrix.abs().max().max(),
            "mean_correlation": corr_matrix.abs().mean().mean()
        }
    
    def _get_time_series_properties(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze time series specific properties"""
        if 'timestamp' not in df.columns:
            return {"message": "No timestamp column found"}
        
        timestamps = pd.to_datetime(df['timestamp'], unit='s' if df['timestamp'].dtype in ['int64', 'float64'] else None)
        
        # Calculate frequency
        if len(timestamps) > 1:
            time_diffs = timestamps.diff().dropna()
            mode_diff = time_diffs.mode()
            freq_seconds = mode_diff[0].total_seconds() if len(mode_diff) > 0 else None
        else:
            freq_seconds = None
        
        return {
            "time_range": {
                "start": timestamps.min().isoformat(),
                "end": timestamps.max().isoformat(),
                "duration_days": (timestamps.max() - timestamps.min()).days
            },
            "frequency_seconds": freq_seconds,
            "total_periods": len(timestamps),
            "time_gaps": {
                "max_gap_hours": time_diffs.max().total_seconds() / 3600 if len(time_diffs) > 0 else 0,
                "gaps_over_2x_normal": (time_diffs > mode_diff[0] * 2).sum() if len(mode_diff) > 0 else 0
            }
        }
    
    def _get_quality_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall data quality indicators"""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        
        # Calculate quality metrics
        completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        
        # Consistency (based on duplicates and data type consistency)
        duplicate_ratio = df.duplicated().sum() / len(df) if len(df) > 0 else 0
        consistency = 1 - duplicate_ratio
        
        # Validity (based on data constraints)
        validity_issues = 0
        total_checks = 0
        
        # Check negative volumes
        if 'volume' in df.columns:
            validity_issues += (df['volume'] < 0).sum()
            total_checks += len(df)
        
        # Check zero prices
        price_cols = [col for col in df.columns if col in ['open', 'high', 'low', 'close']]
        for col in price_cols:
            validity_issues += (df[col] <= 0).sum()
            total_checks += len(df)
        
        validity = 1 - (validity_issues / total_checks) if total_checks > 0 else 1
        
        return {
            "completeness": completeness,
            "consistency": consistency,
            "validity": validity,
            "overall_quality": (completeness + consistency + validity) / 3
        }
    
    def detect_drift(self, current_df: pd.DataFrame, baseline_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Detect data drift compared to baseline"""
        current_profile = self.create_profile(current_df, "current")
        drift_results = {}
        
        # Compare basic statistics for numeric columns
        for col in current_df.select_dtypes(include=[np.number]).columns:
            if col in baseline_profile.get("column_profiles", {}):
                baseline_stats = baseline_profile["column_profiles"][col]
                current_stats = current_profile["column_profiles"][col]
                
                # Calculate statistical distance
                drift_score = self._calculate_drift_score(baseline_stats, current_stats)
                
                drift_results[col] = {
                    "drift_score": drift_score,
                    "significant_drift": drift_score > 0.1,  # Threshold for significant drift
                    "baseline_mean": baseline_stats.get("mean"),
                    "current_mean": current_stats.get("mean"),
                    "baseline_std": baseline_stats.get("std"),
                    "current_std": current_stats.get("std")
                }
        
        return {
            "drift_results": drift_results,
            "overall_drift_score": np.mean([r["drift_score"] for r in drift_results.values()]),
            "columns_with_drift": [col for col, r in drift_results.items() if r["significant_drift"]]
        }
    
    def _calculate_drift_score(self, baseline_stats: Dict, current_stats: Dict) -> float:
        """Calculate drift score between baseline and current statistics"""
        # Simple drift detection based on mean and std changes
        try:
            mean_change = abs(baseline_stats["mean"] - current_stats["mean"]) / abs(baseline_stats["mean"]) if baseline_stats["mean"] != 0 else 0
            std_change = abs(baseline_stats["std"] - current_stats["std"]) / abs(baseline_stats["std"]) if baseline_stats["std"] != 0 else 0
            
            return (mean_change + std_change) / 2
        except (KeyError, TypeError, ZeroDivisionError):
            return 0.0


class DataQualityEnhancer:
    """Advanced data cleaning and enhancement"""
    
    def __init__(self, config: DataQualityConfig = None):
        self.config = config or DataQualityConfig()
    
    def enhance_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply comprehensive data quality enhancements"""
        enhanced_df = df.copy()
        enhancement_log = {"operations": [], "statistics": {}}
        
        # 1. Handle missing data
        enhanced_df, missing_log = self._handle_missing_data(enhanced_df)
        enhancement_log["operations"].append(missing_log)
        
        # 2. Detect and handle outliers
        enhanced_df, outlier_log = self._handle_outliers(enhanced_df)
        enhancement_log["operations"].append(outlier_log)
        
        # 3. Fix data inconsistencies
        enhanced_df, consistency_log = self._fix_inconsistencies(enhanced_df)
        enhancement_log["operations"].append(consistency_log)
        
        # 4. Enhance temporal data
        enhanced_df, temporal_log = self._enhance_temporal_data(enhanced_df)
        enhancement_log["operations"].append(temporal_log)
        
        # 5. Add quality indicators
        enhanced_df, quality_log = self._add_quality_indicators(enhanced_df)
        enhancement_log["operations"].append(quality_log)
        
        # Calculate final statistics
        enhancement_log["statistics"] = {
            "original_shape": df.shape,
            "enhanced_shape": enhanced_df.shape,
            "data_quality_improvement": self._calculate_quality_improvement(df, enhanced_df)
        }
        
        return enhanced_df, enhancement_log
    
    def _handle_missing_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Advanced missing data handling"""
        enhanced_df = df.copy()
        log = {"operation": "missing_data_handling", "details": {}}
        
        missing_before = enhanced_df.isnull().sum().sum()
        
        # Strategy 1: Forward fill for time series data
        if 'timestamp' in enhanced_df.columns:
            enhanced_df = enhanced_df.sort_values('timestamp')
            
            # Forward fill price data (most recent valid observation)
            price_cols = [col for col in enhanced_df.columns if col in ['open', 'high', 'low', 'close']]
            for col in price_cols:
                enhanced_df[col] = enhanced_df[col].fillna(method='ffill')
            
            # For external data, use more sophisticated imputation
            external_cols = [col for col in enhanced_df.columns 
                           if any(suffix in col for suffix in ['_sentiment', '_macro', '_onchain'])]
            
            if external_cols:
                # Use KNN imputation for external data
                knn_imputer = KNNImputer(n_neighbors=5)
                enhanced_df[external_cols] = knn_imputer.fit_transform(enhanced_df[external_cols])
        
        # Strategy 2: Backward fill remaining gaps
        enhanced_df = enhanced_df.fillna(method='bfill')
        
        # Strategy 3: Interpolation for remaining numeric columns
        numeric_cols = enhanced_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if enhanced_df[col].isnull().any():
                enhanced_df[col] = enhanced_df[col].interpolate(method='linear')
        
        # Strategy 4: Fill remaining with column median
        for col in enhanced_df.columns:
            if enhanced_df[col].isnull().any():
                if enhanced_df[col].dtype in [np.number]:
                    enhanced_df[col] = enhanced_df[col].fillna(enhanced_df[col].median())
                else:
                    enhanced_df[col] = enhanced_df[col].fillna(enhanced_df[col].mode().iloc[0] if not enhanced_df[col].mode().empty else "unknown")
        
        missing_after = enhanced_df.isnull().sum().sum()
        
        log["details"] = {
            "missing_before": missing_before,
            "missing_after": missing_after,
            "improvement": missing_before - missing_after
        }
        
        return enhanced_df, log
    
    def _handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Advanced outlier detection and handling"""
        enhanced_df = df.copy()
        log = {"operation": "outlier_handling", "details": {}}
        
        outliers_handled = 0
        
        # Use Isolation Forest for multivariate outlier detection
        numeric_cols = enhanced_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = isolation_forest.fit_predict(enhanced_df[numeric_cols].fillna(0))
            
            # Handle outliers by capping at percentiles
            outlier_indices = np.where(outlier_labels == -1)[0]
            
            for col in numeric_cols:
                col_outliers = enhanced_df.iloc[outlier_indices][col]
                if len(col_outliers) > 0:
                    # Cap at 5th and 95th percentiles
                    p5 = enhanced_df[col].quantile(0.05)
                    p95 = enhanced_df[col].quantile(0.95)
                    
                    outlier_mask = enhanced_df.index.isin(outlier_indices)
                    enhanced_df.loc[outlier_mask & (enhanced_df[col] < p5), col] = p5
                    enhanced_df.loc[outlier_mask & (enhanced_df[col] > p95), col] = p95
                    
                    outliers_handled += outlier_mask.sum()
        
        log["details"] = {
            "outliers_detected_and_handled": outliers_handled,
            "detection_method": "Isolation Forest + Percentile Capping"
        }
        
        return enhanced_df, log
    
    def _fix_inconsistencies(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fix data inconsistencies"""
        enhanced_df = df.copy()
        log = {"operation": "consistency_fixes", "details": {}}
        
        fixes_applied = []
        
        # Remove duplicate timestamps
        if 'timestamp' in enhanced_df.columns:
            before_dedup = len(enhanced_df)
            enhanced_df = enhanced_df.drop_duplicates(subset=['timestamp'], keep='first')
            after_dedup = len(enhanced_df)
            
            if before_dedup != after_dedup:
                fixes_applied.append(f"Removed {before_dedup - after_dedup} duplicate timestamps")
        
        # Fix negative volumes
        if 'volume' in enhanced_df.columns:
            negative_volumes = (enhanced_df['volume'] < 0).sum()
            enhanced_df['volume'] = enhanced_df['volume'].clip(lower=0)
            if negative_volumes > 0:
                fixes_applied.append(f"Fixed {negative_volumes} negative volume values")
        
        # Ensure OHLC relationships
        ohlc_cols = ['open', 'high', 'low', 'close']
        if all(col in enhanced_df.columns for col in ohlc_cols):
            # Ensure high >= max(open, close)
            enhanced_df['high'] = enhanced_df[['high', 'open', 'close']].max(axis=1)
            # Ensure low <= min(open, close)
            enhanced_df['low'] = enhanced_df[['low', 'open', 'close']].min(axis=1)
            fixes_applied.append("Enforced OHLC price relationships")
        
        log["details"] = {"fixes_applied": fixes_applied}
        
        return enhanced_df, log
    
    def _enhance_temporal_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Enhance temporal aspects of the data"""
        enhanced_df = df.copy()
        log = {"operation": "temporal_enhancement", "details": {}}
        
        if 'timestamp' not in enhanced_df.columns:
            log["details"]["message"] = "No timestamp column found"
            return enhanced_df, log
        
        # Ensure temporal ordering
        enhanced_df = enhanced_df.sort_values('timestamp').reset_index(drop=True)
        
        # Add temporal features
        timestamps = pd.to_datetime(enhanced_df['timestamp'], unit='s' if enhanced_df['timestamp'].dtype in ['int64', 'float64'] else None)
        
        enhanced_df['hour'] = timestamps.dt.hour
        enhanced_df['day_of_week'] = timestamps.dt.dayofweek
        enhanced_df['month'] = timestamps.dt.month
        enhanced_df['quarter'] = timestamps.dt.quarter
        
        # Add lag features for time series analysis
        if 'close' in enhanced_df.columns:
            enhanced_df['price_lag_1'] = enhanced_df['close'].shift(1)
            enhanced_df['price_change'] = enhanced_df['close'].pct_change()
            enhanced_df['price_volatility_24h'] = enhanced_df['close'].rolling(24).std()
        
        log["details"] = {
            "temporal_features_added": ["hour", "day_of_week", "month", "quarter"],
            "lag_features_added": ["price_lag_1", "price_change", "price_volatility_24h"] if 'close' in enhanced_df.columns else []
        }
        
        return enhanced_df, log
    
    def _add_quality_indicators(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Add data quality indicator columns"""
        enhanced_df = df.copy()
        log = {"operation": "quality_indicators", "details": {}}
        
        # Add data freshness indicator
        if 'timestamp' in enhanced_df.columns:
            timestamps = pd.to_datetime(enhanced_df['timestamp'], unit='s' if enhanced_df['timestamp'].dtype in ['int64', 'float64'] else None)
            enhanced_df['data_age_hours'] = (datetime.now() - timestamps).dt.total_seconds() / 3600
        
        # Add missing data indicator per row
        enhanced_df['missing_data_ratio'] = enhanced_df.isnull().sum(axis=1) / len(enhanced_df.columns)
        
        # Add data quality score per row
        quality_factors = []
        
        # Factor 1: No missing data
        quality_factors.append((enhanced_df['missing_data_ratio'] == 0).astype(int))
        
        # Factor 2: Volume data available (if volume column exists)
        if 'volume' in enhanced_df.columns:
            quality_factors.append((enhanced_df['volume'] > 0).astype(int))
        
        # Factor 3: Price data consistency (if OHLC available)
        if all(col in enhanced_df.columns for col in ['open', 'high', 'low', 'close']):
            price_consistent = (
                (enhanced_df['high'] >= enhanced_df[['open', 'close']].max(axis=1)) &
                (enhanced_df['low'] <= enhanced_df[['open', 'close']].min(axis=1))
            ).astype(int)
            quality_factors.append(price_consistent)
        
        if quality_factors:
            enhanced_df['data_quality_score'] = np.mean(quality_factors, axis=0)
        else:
            enhanced_df['data_quality_score'] = 1.0
        
        log["details"] = {
            "quality_indicators_added": ["data_age_hours", "missing_data_ratio", "data_quality_score"],
            "quality_factors_count": len(quality_factors)
        }
        
        return enhanced_df, log
    
    def _calculate_quality_improvement(self, original_df: pd.DataFrame, enhanced_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate quality improvement metrics"""
        original_missing = original_df.isnull().sum().sum()
        enhanced_missing = enhanced_df.isnull().sum().sum()
        
        original_total = original_df.size
        enhanced_total = enhanced_df.size
        
        return {
            "missing_data_reduction": (original_missing - enhanced_missing) / original_missing if original_missing > 0 else 0,
            "data_completeness_improvement": (1 - enhanced_missing / enhanced_total) - (1 - original_missing / original_total) if original_total > 0 else 0,
            "additional_features_ratio": (enhanced_df.shape[1] - original_df.shape[1]) / original_df.shape[1] if original_df.shape[1] > 0 else 0
        }


class DataQualityManager:
    """Main manager for comprehensive data quality operations"""
    
    def __init__(self, config: DataQualityConfig = None):
        self.config = config or DataQualityConfig()
        self.validator = DataValidator(self.config)
        self.profiler = DataProfiler()
        self.enhancer = DataQualityEnhancer(self.config)
    
    def generate_quality_report(self, df: pd.DataFrame, dataset_name: str = "dataset") -> DataQualityReport:
        """Generate comprehensive data quality report"""
        # Run validation
        validation_results = self.validator.validate_dataset(df, dataset_name)
        
        # Create data profile
        data_profile = self.profiler.create_profile(df, dataset_name)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(validation_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validation_results, data_profile)
        
        # Dataset info
        dataset_info = {
            "name": dataset_name,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        return DataQualityReport(
            timestamp=datetime.now(),
            overall_score=overall_score,
            validation_results=validation_results,
            data_profile=data_profile,
            recommendations=recommendations,
            dataset_info=dataset_info
        )
    
    def _calculate_overall_score(self, validation_results: List[ValidationResult]) -> float:
        """Calculate weighted overall quality score"""
        if not validation_results:
            return 0.0
        
        weights = {
            "Data Completeness": self.config.completeness_weight,
            "Data Consistency": self.config.consistency_weight,
            "Price Integrity": self.config.accuracy_weight,
            "Temporal Integrity": self.config.timeliness_weight
        }
        
        weighted_scores = []
        for result in validation_results:
            weight = weights.get(result.check_name, 0.1)  # Default weight for other checks
            weighted_scores.append(result.score * weight)
        
        return sum(weighted_scores) / sum(weights.values()) if weighted_scores else 0.0
    
    def _generate_recommendations(self, validation_results: List[ValidationResult], 
                                data_profile: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Analyze validation results
        for result in validation_results:
            if not result.passed:
                if result.check_name == "Data Completeness":
                    recommendations.append("Consider implementing advanced missing data imputation strategies")
                elif result.check_name == "Data Consistency":
                    recommendations.append("Review data collection and preprocessing pipelines for consistency issues")
                elif result.check_name == "Price Integrity":
                    recommendations.append("Implement real-time price validation and anomaly detection")
                elif result.check_name == "Temporal Integrity":
                    recommendations.append("Add data freshness monitoring and gap detection alerts")
        
        # Analyze data profile for additional recommendations
        quality_indicators = data_profile.get("data_quality_indicators", {})
        overall_quality = quality_indicators.get("overall_quality", 1.0)
        
        if overall_quality < 0.8:
            recommendations.append("Overall data quality is below optimal - consider comprehensive data enhancement")
        
        # Check for high correlation
        corr_analysis = data_profile.get("correlations", {})
        high_corr_pairs = corr_analysis.get("high_correlation_pairs", [])
        if len(high_corr_pairs) > 5:
            recommendations.append("Consider feature selection to reduce multicollinearity")
        
        # Check external data coverage
        column_profiles = data_profile.get("column_profiles", {})
        external_cols = [col for col in column_profiles.keys() 
                        if any(suffix in col for suffix in ['_sentiment', '_macro', '_onchain'])]
        if len(external_cols) == 0:
            recommendations.append("Consider integrating external data sources for improved predictions")
        
        return recommendations
    
    def save_quality_report(self, report: DataQualityReport, output_dir: str = "./results/data_quality"):
        """Save quality report to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON report
        report_dict = {
            "timestamp": report.timestamp.isoformat(),
            "overall_score": report.overall_score,
            "dataset_info": report.dataset_info,
            "validation_results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "score": r.score,
                    "message": r.message,
                    "severity": r.severity,
                    "details": r.details
                }
                for r in report.validation_results
            ],
            "data_profile": report.data_profile,
            "recommendations": report.recommendations
        }
        
        report_file = os.path.join(output_dir, f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info(f"Data quality report saved to {report_file}")
        
        return report_file


# Convenience function for quick quality assessment
def assess_data_quality(df: pd.DataFrame, dataset_name: str = "dataset", 
                       config: DataQualityConfig = None) -> DataQualityReport:
    """Quick function to assess data quality"""
    manager = DataQualityManager(config)
    return manager.generate_quality_report(df, dataset_name)


def enhance_data_quality(df: pd.DataFrame, config: DataQualityConfig = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Quick function to enhance data quality"""
    enhancer = DataQualityEnhancer(config)
    return enhancer.enhance_dataset(df)


if __name__ == "__main__":
    # Example usage
    print("Data Quality Enhancement System initialized")
    print("Use assess_data_quality() and enhance_data_quality() functions for quick operations")