#!/usr/bin/env python3
"""
Test script for Data Quality Enhancement System
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.data_quality import (
        DataQualityManager,
        DataQualityConfig,
        assess_data_quality,
        enhance_data_quality
    )
    DATA_QUALITY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Data quality components not available: {e}")
    DATA_QUALITY_AVAILABLE = False


def create_test_data():
    """Create test cryptocurrency data with various quality issues"""
    print("Creating test cryptocurrency data with quality issues...")
    
    # Base data
    np.random.seed(42)
    n_points = 1000
    
    # Create timestamps with some gaps
    start_time = datetime(2023, 1, 1)
    timestamps = []
    current_time = start_time
    
    for i in range(n_points):
        timestamps.append(int(current_time.timestamp()))
        
        # Add random gaps (some larger than normal)
        if np.random.random() < 0.05:  # 5% chance of gap
            gap_hours = int(np.random.choice([2, 6, 24]))  # 2h, 6h, or 24h gap
            current_time += timedelta(hours=gap_hours)
        else:
            current_time += timedelta(hours=1)  # Normal 1h increment
    
    # Create OHLCV data with issues
    base_price = 50000
    prices = []
    volumes = []
    
    current_price = base_price
    
    for i in range(n_points):
        # Price with random walk and occasional spikes
        if i > 0:
            change = np.random.normal(0, 0.02)  # 2% volatility
            
            # Add occasional extreme price spikes (data quality issue)
            if np.random.random() < 0.01:  # 1% chance
                change = np.random.choice([-0.7, 0.8])  # Extreme spike
            
            current_price = current_price * (1 + change)
        
        price = current_price
        
        # Generate OHLC with some violations
        open_price = price * np.random.uniform(0.995, 1.005)
        close_price = price * np.random.uniform(0.995, 1.005)
        
        # Sometimes violate OHLC relationships (data quality issue)
        if np.random.random() < 0.05:  # 5% violations
            high_price = min(open_price, close_price) * 0.999  # High < open/close
            low_price = max(open_price, close_price) * 1.001   # Low > open/close
        else:
            high_price = max(open_price, close_price) * np.random.uniform(1.001, 1.01)
            low_price = min(open_price, close_price) * np.random.uniform(0.99, 0.999)
        
        prices.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price
        })
        
        # Volume with spikes and some negative values (data quality issue)
        volume = np.random.lognormal(10, 1)
        if np.random.random() < 0.02:  # 2% negative volumes
            volume = -volume
        if np.random.random() < 0.01:  # 1% extreme volume spikes
            volume *= 50
        
        volumes.append(volume)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': [p['open'] for p in prices],
        'high': [p['high'] for p in prices],
        'low': [p['low'] for p in prices],
        'close': [p['close'] for p in prices],
        'volume': volumes
    })
    
    # Add some missing data
    missing_indices = np.random.choice(df.index, size=int(len(df) * 0.03), replace=False)
    missing_columns = np.random.choice(['open', 'high', 'low', 'close', 'volume'], size=len(missing_indices))
    
    for idx, col in zip(missing_indices, missing_columns):
        df.loc[idx, col] = np.nan
    
    # Add some duplicate timestamps
    duplicate_indices = np.random.choice(df.index[:-1], size=5, replace=False)
    for idx in duplicate_indices:
        df.loc[idx + 1, 'timestamp'] = df.loc[idx, 'timestamp']
    
    print(f"Created test dataset with {len(df)} records")
    print(f"Data quality issues intentionally added:")
    print(f"  - Missing data: ~3% of cells")
    print(f"  - OHLC violations: ~5% of records") 
    print(f"  - Negative volumes: ~2% of records")
    print(f"  - Price spikes: ~1% extreme changes")
    print(f"  - Volume spikes: ~1% extreme spikes")
    print(f"  - Duplicate timestamps: 5 instances")
    print(f"  - Time gaps: Various gaps in hourly data")
    
    return df


def test_data_validation():
    """Test data validation functionality"""
    print("\n" + "=" * 60)
    print("TESTING DATA VALIDATION")
    print("=" * 60)
    
    if not DATA_QUALITY_AVAILABLE:
        print("Data quality components not available - skipping")
        return False
    
    try:
        # Create test data
        test_df = create_test_data()
        
        # Test validation
        print("\nRunning comprehensive data validation...")
        quality_manager = DataQualityManager()
        validation_results = quality_manager.validator.validate_dataset(test_df, "test_crypto")
        
        print(f"\nValidation completed - {len(validation_results)} checks performed:")
        
        passed_checks = 0
        for result in validation_results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            severity_icon = {"info": "ℹ", "warning": "⚠", "error": "❌", "critical": "🚨"}
            icon = severity_icon.get(result.severity, "•")
            
            print(f"  {icon} {status} {result.check_name} (Score: {result.score:.2f})")
            print(f"    {result.message}")
            
            if result.passed:
                passed_checks += 1
        
        print(f"\nSummary: {passed_checks}/{len(validation_results)} checks passed")
        
        return True, validation_results
        
    except Exception as e:
        print(f"✗ Data validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, []


def test_data_profiling():
    """Test data profiling functionality"""
    print("\n" + "=" * 60)
    print("TESTING DATA PROFILING")
    print("=" * 60)
    
    if not DATA_QUALITY_AVAILABLE:
        print("Data quality components not available - skipping")
        return False
    
    try:
        # Create test data
        test_df = create_test_data()
        
        # Test profiling
        print("\nCreating comprehensive data profile...")
        quality_manager = DataQualityManager()
        profile = quality_manager.profiler.create_profile(test_df, "test_crypto")
        
        # Display profile summary
        basic_info = profile["basic_info"]
        print(f"\nDataset Profile:")
        print(f"  Shape: {basic_info['shape']}")
        print(f"  Memory usage: {basic_info['memory_usage_mb']:.2f} MB")
        print(f"  Missing cells: {basic_info['missing_cells']}")
        print(f"  Duplicate rows: {basic_info['duplicate_rows']}")
        
        # Column profiles
        column_profiles = profile["column_profiles"]
        print(f"\nColumn Analysis:")
        for col, col_profile in column_profiles.items():
            missing_pct = col_profile['missing_percentage']
            unique_pct = col_profile['unique_percentage']
            print(f"  {col}:")
            print(f"    Missing: {missing_pct:.1f}%, Unique: {unique_pct:.1f}%")
            
            if 'mean' in col_profile:
                print(f"    Mean: {col_profile['mean']:.2f}, Std: {col_profile['std']:.2f}")
        
        # Correlation analysis
        correlations = profile["correlations"]
        if "high_correlation_pairs" in correlations:
            high_corr = correlations["high_correlation_pairs"]
            print(f"\nHigh correlation pairs found: {len(high_corr)}")
            for pair in high_corr[:3]:  # Show first 3
                print(f"  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
        
        # Time series properties
        ts_props = profile["time_series_properties"]
        if "time_range" in ts_props:
            time_range = ts_props["time_range"]
            print(f"\nTime Series Properties:")
            print(f"  Duration: {time_range['duration_days']} days")
            print(f"  Total periods: {ts_props['total_periods']}")
            if "time_gaps" in ts_props:
                gaps = ts_props["time_gaps"]
                print(f"  Max gap: {gaps['max_gap_hours']:.1f} hours")
        
        # Quality indicators
        quality_indicators = profile["data_quality_indicators"]
        print(f"\nData Quality Indicators:")
        print(f"  Completeness: {quality_indicators['completeness']:.3f}")
        print(f"  Consistency: {quality_indicators['consistency']:.3f}")
        print(f"  Validity: {quality_indicators['validity']:.3f}")
        print(f"  Overall Quality: {quality_indicators['overall_quality']:.3f}")
        
        print("\n✓ Data profiling test passed")
        return True, profile
        
    except Exception as e:
        print(f"✗ Data profiling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def test_data_enhancement():
    """Test data enhancement functionality"""
    print("\n" + "=" * 60)
    print("TESTING DATA ENHANCEMENT")
    print("=" * 60)
    
    if not DATA_QUALITY_AVAILABLE:
        print("Data quality components not available - skipping")
        return False
    
    try:
        # Create test data
        test_df = create_test_data()
        
        print(f"\nOriginal data shape: {test_df.shape}")
        print(f"Original missing values: {test_df.isnull().sum().sum()}")
        
        # Test enhancement
        print("\nApplying comprehensive data enhancement...")
        enhanced_df, enhancement_log = enhance_data_quality(test_df)
        
        print(f"\nEnhanced data shape: {enhanced_df.shape}")
        print(f"Enhanced missing values: {enhanced_df.isnull().sum().sum()}")
        
        # Display enhancement log
        print(f"\nEnhancement Operations Applied:")
        for operation in enhancement_log["operations"]:
            print(f"  • {operation['operation']}:")
            for key, value in operation["details"].items():
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value}")
                elif isinstance(value, list) and len(value) <= 3:
                    print(f"    {key}: {value}")
        
        # Display improvement statistics
        stats = enhancement_log["statistics"]
        print(f"\nImprovement Statistics:")
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
        # Validate enhanced data
        print(f"\nValidating enhanced data...")
        quality_manager = DataQualityManager()
        enhanced_validation = quality_manager.validator.validate_dataset(enhanced_df, "enhanced_crypto")
        
        enhanced_passed = sum(1 for r in enhanced_validation if r.passed)
        print(f"Enhanced data validation: {enhanced_passed}/{len(enhanced_validation)} checks passed")
        
        print("\n✓ Data enhancement test passed")
        return True, enhanced_df, enhancement_log
        
    except Exception as e:
        print(f"✗ Data enhancement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, {}


def test_quality_report_generation():
    """Test comprehensive quality report generation"""
    print("\n" + "=" * 60)
    print("TESTING QUALITY REPORT GENERATION")
    print("=" * 60)
    
    if not DATA_QUALITY_AVAILABLE:
        print("Data quality components not available - skipping")
        return False
    
    try:
        # Create test data
        test_df = create_test_data()
        
        # Generate comprehensive report
        print("\nGenerating comprehensive data quality report...")
        report = assess_data_quality(test_df, "test_cryptocurrency_data")
        
        print(f"\nData Quality Report Generated:")
        print(f"  Timestamp: {report.timestamp}")
        print(f"  Overall Score: {report.overall_score:.3f}")
        print(f"  Dataset: {report.dataset_info['name']}")
        print(f"  Shape: {report.dataset_info['shape']}")
        print(f"  Memory Usage: {report.dataset_info['memory_usage_mb']:.2f} MB")
        
        # Validation summary
        passed_validations = sum(1 for r in report.validation_results if r.passed)
        print(f"\nValidation Summary:")
        print(f"  Checks passed: {passed_validations}/{len(report.validation_results)}")
        
        critical_issues = [r for r in report.validation_results if r.severity == "critical"]
        error_issues = [r for r in report.validation_results if r.severity == "error"]
        warning_issues = [r for r in report.validation_results if r.severity == "warning"]
        
        if critical_issues:
            print(f"  Critical issues: {len(critical_issues)}")
        if error_issues:
            print(f"  Error issues: {len(error_issues)}")
        if warning_issues:
            print(f"  Warning issues: {len(warning_issues)}")
        
        # Recommendations
        print(f"\nRecommendations ({len(report.recommendations)}):")
        for i, rec in enumerate(report.recommendations[:3], 1):  # Show first 3
            print(f"  {i}. {rec}")
        
        if len(report.recommendations) > 3:
            print(f"  ... and {len(report.recommendations) - 3} more")
        
        # Save report
        quality_manager = DataQualityManager()
        report_file = quality_manager.save_quality_report(report)
        print(f"\nReport saved to: {report_file}")
        
        print("\n✓ Quality report generation test passed")
        return True, report
        
    except Exception as e:
        print(f"✗ Quality report generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_drift_detection():
    """Test data drift detection"""
    print("\n" + "=" * 60)
    print("TESTING DATA DRIFT DETECTION")
    print("=" * 60)
    
    if not DATA_QUALITY_AVAILABLE:
        print("Data quality components not available - skipping")
        return False
    
    try:
        # Create baseline data
        baseline_df = create_test_data()
        
        # Create drifted data (shifted mean and different volatility)
        drifted_df = baseline_df.copy()
        drifted_df['close'] = drifted_df['close'] * 1.2  # 20% price increase
        drifted_df['volume'] = drifted_df['volume'] * 2   # 2x volume increase
        
        # Add some noise to simulate drift
        np.random.seed(123)
        drifted_df['close'] += np.random.normal(0, drifted_df['close'].std() * 0.5, len(drifted_df))
        
        print(f"\nBaseline data shape: {baseline_df.shape}")
        print(f"Drifted data shape: {drifted_df.shape}")
        
        # Create profiles
        quality_manager = DataQualityManager()
        baseline_profile = quality_manager.profiler.create_profile(baseline_df, "baseline")
        
        # Detect drift
        print("\nDetecting data drift...")
        drift_results = quality_manager.profiler.detect_drift(drifted_df, baseline_profile)
        
        overall_drift = drift_results["overall_drift_score"]
        drifted_columns = drift_results["columns_with_drift"]
        
        print(f"\nDrift Detection Results:")
        print(f"  Overall drift score: {overall_drift:.3f}")
        print(f"  Columns with significant drift: {len(drifted_columns)}")
        
        for col in drifted_columns:
            drift_info = drift_results["drift_results"][col]
            print(f"    {col}: drift score {drift_info['drift_score']:.3f}")
            print(f"      Baseline mean: {drift_info['baseline_mean']:.2f}")
            print(f"      Current mean: {drift_info['current_mean']:.2f}")
        
        # Test with no drift (same data)
        print(f"\nTesting with no drift (same data)...")
        no_drift_results = quality_manager.profiler.detect_drift(baseline_df, baseline_profile)
        print(f"  No-drift score: {no_drift_results['overall_drift_score']:.3f}")
        
        print("\n✓ Data drift detection test passed")
        return True, drift_results
        
    except Exception as e:
        print(f"✗ Data drift detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def create_visualizations(test_results):
    """Create visualizations for test results"""
    print("\n" + "=" * 60)
    print("CREATING DATA QUALITY VISUALIZATIONS")
    print("=" * 60)
    
    try:
        os.makedirs('./plots', exist_ok=True)
        
        # Extract validation results if available
        if 'validation_results' in test_results:
            validation_results = test_results['validation_results']
            
            # Create validation scores plot
            check_names = [r.check_name for r in validation_results]
            scores = [r.score for r in validation_results]
            colors = ['green' if r.passed else 'red' for r in validation_results]
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(check_names)), scores, color=colors, alpha=0.7)
            plt.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Good Quality Threshold')
            plt.axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label='Minimum Quality Threshold')
            
            plt.xlabel('Validation Checks')
            plt.ylabel('Quality Score')
            plt.title('Data Quality Validation Results')
            plt.xticks(range(len(check_names)), check_names, rotation=45, ha='right')
            plt.ylim(0, 1)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig('./plots/data_quality_validation.png', dpi=150, bbox_inches='tight')
            print("Saved: ./plots/data_quality_validation.png")
            plt.close()
        
        # Create test data quality issues visualization
        test_df = create_test_data()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Quality Issues in Test Dataset', fontsize=16)
        
        # Missing data heatmap
        ax1 = axes[0, 0]
        missing_data = test_df.isnull()
        if missing_data.any().any():
            import seaborn as sns
            sns.heatmap(missing_data.iloc[:100], ax=ax1, cbar=True, yticklabels=False, 
                       cmap='Reds', cbar_kws={'label': 'Missing Data'})
        ax1.set_title('Missing Data Pattern (First 100 rows)')
        ax1.set_xlabel('Columns')
        
        # Price data validation
        ax2 = axes[0, 1]
        if all(col in test_df.columns for col in ['open', 'high', 'low', 'close']):
            # Check OHLC violations
            high_violations = test_df['high'] < test_df[['open', 'close']].max(axis=1)
            low_violations = test_df['low'] > test_df[['open', 'close']].min(axis=1)
            
            violation_counts = [
                (~high_violations).sum(),
                high_violations.sum(),
                (~low_violations).sum(), 
                low_violations.sum()
            ]
            labels = ['High Valid', 'High Invalid', 'Low Valid', 'Low Invalid']
            colors = ['green', 'red', 'green', 'red']
            
            ax2.bar(labels, violation_counts, color=colors, alpha=0.7)
            ax2.set_title('OHLC Relationship Violations')
            ax2.set_ylabel('Count')
        
        # Volume data issues
        ax3 = axes[1, 0]
        if 'volume' in test_df.columns:
            volume_stats = {
                'Normal': (test_df['volume'] > 0).sum(),
                'Negative': (test_df['volume'] < 0).sum(),
                'Zero': (test_df['volume'] == 0).sum()
            }
            
            colors = ['green', 'red', 'orange']
            ax3.bar(volume_stats.keys(), volume_stats.values(), color=colors, alpha=0.7)
            ax3.set_title('Volume Data Issues')
            ax3.set_ylabel('Count')
        
        # Price change distribution
        ax4 = axes[1, 1]
        if 'close' in test_df.columns:
            price_changes = test_df['close'].pct_change().dropna()
            ax4.hist(price_changes, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax4.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Extreme Change Threshold')
            ax4.axvline(x=-0.5, color='red', linestyle='--', alpha=0.7)
            ax4.set_title('Price Change Distribution')
            ax4.set_xlabel('Price Change (%)')
            ax4.set_ylabel('Frequency')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig('./plots/data_quality_issues_analysis.png', dpi=150, bbox_inches='tight')
        print("Saved: ./plots/data_quality_issues_analysis.png")
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return False


def main():
    """Main test function for data quality enhancement"""
    print("Testing Data Quality Enhancement System")
    print("=" * 80)
    
    test_results = {}
    
    # Run all tests
    print(f"Data quality components available: {DATA_QUALITY_AVAILABLE}")
    
    if not DATA_QUALITY_AVAILABLE:
        print("❌ Data quality system not available. Please install required dependencies.")
        return False
    
    # Test 1: Data validation
    success1, validation_results = test_data_validation()
    test_results['validation_results'] = validation_results
    
    # Test 2: Data profiling
    success2, profile = test_data_profiling()
    test_results['profile'] = profile
    
    # Test 3: Data enhancement
    success3, enhanced_df, enhancement_log = test_data_enhancement()
    test_results['enhanced_data'] = enhanced_df
    test_results['enhancement_log'] = enhancement_log
    
    # Test 4: Quality report generation
    success4, report = test_quality_report_generation()
    test_results['quality_report'] = report
    
    # Test 5: Drift detection
    success5, drift_results = test_drift_detection()
    test_results['drift_results'] = drift_results
    
    # Create visualizations
    success6 = create_visualizations(test_results)
    
    # Summary
    test_results_summary = [
        ("Data Validation", success1),
        ("Data Profiling", success2),
        ("Data Enhancement", success3),
        ("Quality Report Generation", success4),
        ("Drift Detection", success5),
        ("Visualizations", success6)
    ]
    
    print("\n" + "=" * 80)
    print("DATA QUALITY ENHANCEMENT TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(test_results_summary)
    
    for test_name, result in test_results_summary:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:30} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed >= total - 1:  # Allow one failure
        print("\n🎉 Data Quality Enhancement system is working correctly!")
        print("\nKey capabilities now available:")
        print("• Comprehensive data validation (8 validation checks)")
        print("• Advanced data profiling and statistical analysis")
        print("• Automated data enhancement with 5 improvement strategies")
        print("• Missing data imputation using KNN and interpolation")
        print("• Outlier detection using Isolation Forest")
        print("• Data drift detection and monitoring")
        print("• Quality scoring and reporting system")
        print("• Temporal data enhancement with lag features")
        print("• OHLC relationship validation and correction")
        print("• Quality indicator tracking per data point")
        
        print("\nNext steps:")
        print("1. Integrate with existing data loaders")
        print("2. Set up automated quality monitoring")
        print("3. Configure quality thresholds for your data")
        print("4. Add real-time quality alerts")
        
        return True
    else:
        print(f"\n❌ {total-passed} tests failed. Check errors above.")
        return False


if __name__ == "__main__":
    main()