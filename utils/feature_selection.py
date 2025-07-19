import pandas as pd
import numpy as np
# import torch  # Optional dependency
from typing import List, Tuple, Dict, Optional
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import warnings

warnings.filterwarnings('ignore')


class CorrelationBasedFeatureSelector:
    """
    Fast correlation-based feature selection optimized for reducing 68+ financial indicators
    to essential features for faster training.
    """
    
    def __init__(self, target_features: int = 20, correlation_threshold: float = 0.95):
        """
        Args:
            target_features: Target number of features to select (default: 20, ~70% reduction)
            correlation_threshold: Remove features with correlation > threshold (default: 0.95)
        """
        self.target_features = target_features
        self.correlation_threshold = correlation_threshold
        self.selected_features = []
        self.feature_importance_scores = {}
        self.correlation_matrix = None
        
    def analyze_feature_correlations(self, data: pd.DataFrame, target_col: str = 'close') -> pd.DataFrame:
        """
        Analyze correlations between features and target variable
        """
        print("=" * 60)
        print("CORRELATION-BASED FEATURE ANALYSIS")
        print("=" * 60)
        
        # Exclude non-numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['timestamp', 'datetime']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols and col != target_col]
        
        print(f"Total features available: {len(feature_cols)}")
        print(f"Target variable: {target_col}")
        
        # Calculate correlation matrix
        feature_data = data[feature_cols + [target_col]]
        self.correlation_matrix = feature_data.corr()
        
        # Calculate target correlations
        target_correlations = self.correlation_matrix[target_col].abs().sort_values(ascending=False)
        target_correlations = target_correlations.drop(target_col)  # Remove self-correlation
        
        print(f"\nTop 10 features by correlation with {target_col}:")
        for i, (feature, corr) in enumerate(target_correlations.head(10).items()):
            print(f"  {i+1:2d}. {feature:<30} | Correlation: {corr:.4f}")
        
        return target_correlations
    
    def remove_highly_correlated_features(self, feature_correlations: pd.Series, 
                                        feature_data: pd.DataFrame) -> List[str]:
        """
        Remove redundant features that are highly correlated with each other
        """
        print(f"\nRemoving highly correlated features (threshold: {self.correlation_threshold})")
        
        # Start with features ranked by target correlation
        ranked_features = feature_correlations.index.tolist()
        selected = []
        removed_count = 0
        
        for feature in ranked_features:
            if feature not in feature_data.columns:
                continue
                
            # Check correlation with already selected features
            should_add = True
            for selected_feature in selected:
                if selected_feature in feature_data.columns:
                    corr = abs(feature_data[feature].corr(feature_data[selected_feature]))
                    if corr > self.correlation_threshold:
                        should_add = False
                        removed_count += 1
                        break
            
            if should_add:
                selected.append(feature)
        
        print(f"  Removed {removed_count} highly correlated features")
        print(f"  Remaining features: {len(selected)}")
        
        return selected
    
    def select_top_features_by_importance(self, data: pd.DataFrame, 
                                        candidate_features: List[str],
                                        target_col: str = 'close') -> List[str]:
        """
        Select top features using multiple importance metrics
        """
        print(f"\nSelecting top {self.target_features} features by importance...")
        
        feature_data = data[candidate_features]
        target_data = data[target_col]
        
        # Remove any NaN values
        mask = ~(feature_data.isnull().any(axis=1) | target_data.isnull())
        feature_data_clean = feature_data[mask]
        target_data_clean = target_data[mask]
        
        importance_scores = {}
        
        # 1. Pearson correlation with target
        print("  Computing Pearson correlations...")
        for feature in candidate_features:
            if feature in feature_data_clean.columns:
                corr, _ = pearsonr(feature_data_clean[feature], target_data_clean)
                importance_scores[f"{feature}_pearson"] = abs(corr)
        
        # 2. Mutual information
        print("  Computing mutual information scores...")
        try:
            # Scale features for mutual information
            scaler = StandardScaler()
            feature_data_scaled = scaler.fit_transform(feature_data_clean)
            
            mi_scores = mutual_info_regression(feature_data_scaled, target_data_clean, 
                                             random_state=42)
            for i, feature in enumerate(candidate_features):
                if i < len(mi_scores):
                    importance_scores[f"{feature}_mi"] = mi_scores[i]
        except Exception as e:
            print(f"    Mutual information failed: {e}")
        
        # 3. F-statistic (univariate linear regression)
        print("  Computing F-statistics...")
        try:
            f_scores, _ = f_regression(feature_data_clean, target_data_clean)
            for i, feature in enumerate(candidate_features):
                if i < len(f_scores):
                    importance_scores[f"{feature}_f_stat"] = f_scores[i]
        except Exception as e:
            print(f"    F-statistic computation failed: {e}")
        
        # Combine scores for each feature
        feature_combined_scores = {}
        for feature in candidate_features:
            scores = []
            if f"{feature}_pearson" in importance_scores:
                scores.append(importance_scores[f"{feature}_pearson"])
            if f"{feature}_mi" in importance_scores:
                scores.append(importance_scores[f"{feature}_mi"])
            if f"{feature}_f_stat" in importance_scores:
                # Normalize F-statistic
                max_f = max([v for k, v in importance_scores.items() if '_f_stat' in k])
                scores.append(importance_scores[f"{feature}_f_stat"] / max_f)
            
            if scores:
                feature_combined_scores[feature] = np.mean(scores)
        
        # Select top features
        sorted_features = sorted(feature_combined_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        
        selected_features = [feature for feature, score in sorted_features[:self.target_features]]
        
        print(f"\nSelected {len(selected_features)} features:")
        for i, (feature, score) in enumerate(sorted_features[:self.target_features]):
            print(f"  {i+1:2d}. {feature:<30} | Score: {score:.4f}")
        
        self.feature_importance_scores = dict(sorted_features)
        return selected_features
    
    def fit_select(self, data: pd.DataFrame, target_col: str = 'close') -> List[str]:
        """
        Complete feature selection pipeline
        """
        print("Starting correlation-based feature selection...")
        print(f"Input data shape: {data.shape}")
        
        # Step 1: Analyze correlations with target
        target_correlations = self.analyze_feature_correlations(data, target_col)
        
        # Step 2: Remove highly correlated features
        feature_cols = [col for col in data.select_dtypes(include=[np.number]).columns 
                       if col not in ['timestamp', 'datetime', target_col]]
        feature_data = data[feature_cols]
        
        deduplicated_features = self.remove_highly_correlated_features(
            target_correlations, feature_data)
        
        # Step 3: Select top features by importance
        if len(deduplicated_features) > self.target_features:
            self.selected_features = self.select_top_features_by_importance(
                data, deduplicated_features, target_col)
        else:
            self.selected_features = deduplicated_features
            print(f"Using all {len(deduplicated_features)} deduplicated features")
        
        print("\n" + "=" * 60)
        print("FEATURE SELECTION COMPLETE")
        print("=" * 60)
        print(f"Original features: {len(feature_cols)}")
        print(f"Selected features: {len(self.selected_features)}")
        reduction_pct = (1 - len(self.selected_features) / len(feature_cols)) * 100
        print(f"Feature reduction: {reduction_pct:.1f}%")
        print(f"Expected training speedup: {len(feature_cols) / len(self.selected_features):.1f}x")
        
        return self.selected_features
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature selection to new data
        """
        if not self.selected_features:
            raise ValueError("Must call fit_select() first")
        
        # Keep essential columns plus selected features
        essential_cols = ['timestamp']
        if 'datetime' in data.columns:
            essential_cols.append('datetime')
        
        # Add target column if it exists
        target_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in target_cols:
            if col in data.columns:
                essential_cols.append(col)
        
        selected_cols = essential_cols + self.selected_features
        
        # Only keep columns that exist in the data
        available_cols = [col for col in selected_cols if col in data.columns]
        
        return data[available_cols]
    
    def save_selection(self, filepath: str):
        """
        Save feature selection results
        """
        import json
        
        selection_data = {
            'selected_features': self.selected_features,
            'target_features': self.target_features,
            'correlation_threshold': self.correlation_threshold,
            'feature_importance_scores': self.feature_importance_scores
        }
        
        with open(filepath, 'w') as f:
            json.dump(selection_data, f, indent=2)
        
        print(f"Feature selection saved to: {filepath}")
    
    def load_selection(self, filepath: str):
        """
        Load feature selection results
        """
        import json
        
        with open(filepath, 'r') as f:
            selection_data = json.load(f)
        
        self.selected_features = selection_data['selected_features']
        self.target_features = selection_data['target_features']
        self.correlation_threshold = selection_data['correlation_threshold']
        self.feature_importance_scores = selection_data['feature_importance_scores']
        
        print(f"Feature selection loaded from: {filepath}")
        print(f"Loaded {len(self.selected_features)} selected features")


def quick_feature_selection(data_path: str, target_features: int = 20) -> List[str]:
    """
    Quick utility function for feature selection
    """
    print("=" * 60)
    print("QUICK FEATURE SELECTION")
    print("=" * 60)
    
    # Load data
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path)
    
    # Initialize selector
    selector = CorrelationBasedFeatureSelector(target_features=target_features)
    
    # Perform selection
    selected_features = selector.fit_select(data)
    
    # Save results
    output_path = data_path.replace('.csv', '_feature_selection.json')
    selector.save_selection(output_path)
    
    return selected_features


def create_enhanced_data_loader_optimized(original_file: str, output_file: str, 
                                        selected_features: List[str]):
    """
    Create optimized enhanced data loader with selected features only
    """
    print("=" * 60)
    print("CREATING OPTIMIZED DATA LOADER")
    print("=" * 60)
    
    # Read original enhanced data loader
    with open(original_file, 'r') as f:
        content = f.read()
    
    # Create feature filter code
    feature_filter_code = f"""
    # OPTIMIZED FEATURE SELECTION - Auto-generated
    # Selected {len(selected_features)} features from correlation-based selection
    SELECTED_FEATURES = {selected_features}
    
    def filter_features(df_raw):
        \"\"\"Filter to selected features only\"\"\"
        # Keep essential columns
        essential_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if 'datetime' in df_raw.columns:
            essential_cols.append('datetime')
        
        # Add selected technical features
        available_features = [col for col in SELECTED_FEATURES if col in df_raw.columns]
        keep_cols = essential_cols + available_features
        
        print(f"Filtering features: {{len(df_raw.columns)}} -> {{len(keep_cols)}} columns")
        return df_raw[keep_cols]
"""
    
    # Insert feature filtering in the data loading process
    # Find where technical indicators are computed and add filtering after
    modified_content = content.replace(
        "# Apply technical indicators",
        feature_filter_code + "\n        # Apply technical indicators"
    )
    
    # Add filtering call after technical indicators are computed
    modified_content = modified_content.replace(
        "df_raw = self.apply_technical_indicators(df_raw)",
        "df_raw = self.apply_technical_indicators(df_raw)\n        df_raw = filter_features(df_raw)"
    )
    
    # Write optimized version
    with open(output_file, 'w') as f:
        f.write(modified_content)
    
    print(f"Optimized data loader created: {output_file}")
    print(f"Features reduced to {len(selected_features)} most important indicators")


if __name__ == "__main__":
    # Example usage
    data_path = "./dataset/candlesticks-D.csv"  # Daily data for faster testing
    
    print("Starting feature selection for Time-LLM-Cryptex optimization...")
    
    # Quick feature selection
    selected_features = quick_feature_selection(data_path, target_features=20)
    
    print("\nFeature selection completed!")
    print(f"Selected features: {selected_features}")
    
    # Create optimized data loader
    create_enhanced_data_loader_optimized(
        "./data_provider/enhanced_data_loader.py",
        "./data_provider/enhanced_data_loader_optimized.py", 
        selected_features
    )