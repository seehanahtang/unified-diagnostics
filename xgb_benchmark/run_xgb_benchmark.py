#!/usr/bin/env python3
"""
XGBoost Benchmark for Diagnosis Prediction

Systematically evaluates XGBoost across all combinations of:
- Diagnosis types (11 types)
- Time points: 0yr (current), 1yr, 2yr, 5yr, 10yr
- Feature sets: demo+protein, demo+blood, demo+protein+blood, and top-50 variants

Outputs a comprehensive AUC table for all combinations.

Usage:
    python run_xgb_benchmark.py [--output-dir OUTPUT_DIR]
"""

import argparse
import os
import random
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

# Random seed for reproducibility
RANDOM_STATE = 42

# Data path
DATA_PATH = "/orcd/pool/003/dbertsim_shared/ukb/"

# Time points to evaluate (in years)
TIME_POINTS = [0, 1, 2, 5, 10]

# Feature set configurations
FEATURE_SETS = [
    'demo_protein',           # Demographics + Protein features
    'demo_blood',             # Demographics + Blood features  
    'demo_protein_blood',     # Demographics + Protein + Blood (all features)
    'demo_protein_top50',     # Demographics + Top 50 protein features
    'demo_blood_top50',       # Demographics + Top 50 blood features (or all if <50)
    'demo_protein_blood_top50'  # Demographics + Top 50 from protein+blood combined
]

# Demographic columns
DEMO_COLS = [
    'Age at recruitment',
    'Sex_male',
    'Body mass index (BMI)',
    'Systolic blood pressure, automated reading',
    'Diastolic blood pressure, automated reading',
    'Townsend deprivation index at recruitment',
]

# XGBoost parameters (from notebook)
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'enable_categorical': True,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.01,
    'verbosity': 0
}


# =============================================================================
# Utility Functions
# =============================================================================

def set_seeds(seed=RANDOM_STATE):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_data(data_path=DATA_PATH):
    """Load train and test datasets."""
    print("Loading data...")
    train_df = pd.read_csv(f"{data_path}ukb_diag_train.csv", low_memory=False)
    test_df = pd.read_csv(f"{data_path}ukb_diag_test.csv", low_memory=False)
    print(f"  Train: {len(train_df)} rows, {len(train_df.columns)} columns")
    print(f"  Test: {len(test_df)} rows, {len(test_df.columns)} columns")
    return train_df, test_df


def get_feature_columns(df):
    """Extract feature column lists from dataframe."""
    olink_cols = [c for c in df.columns if c.startswith('olink_')]
    blood_cols = [c for c in df.columns if c.startswith('blood_')]
    demo_cols = [c for c in DEMO_COLS if c in df.columns]
    return olink_cols, blood_cols, demo_cols


def get_diagnosis_types(df):
    """Extract diagnosis types from column names."""
    time_cols = [c for c in df.columns if c.endswith('_time_to_diagnosis')]
    diag_types = [c.replace('_time_to_diagnosis', '') for c in time_cols]
    return diag_types


def get_labels_current(df, diag_type):
    """
    Get binary labels for current diagnosis (time_point=0).
    Positive: diagnosed at or before baseline (time_to_diagnosis <= 0).
    """
    time_col = f"{diag_type}_time_to_diagnosis"
    if time_col not in df.columns:
        return None
    # Already diagnosed at baseline
    y = (df[time_col] <= 0).astype(int)
    return y


def get_labels_future(df, diag_type, horizon_years):
    """
    Get binary labels for future diagnosis within horizon.
    Positive: NOT diagnosed at baseline but diagnosed within horizon years.
    Excludes patients diagnosed within first 30 days (likely pre-existing).
    """
    time_col = f"{diag_type}_time_to_diagnosis"
    if time_col not in df.columns:
        return None
    
    # Positive: diagnosed between 30 days and horizon years
    # (time > 0 means not diagnosed at baseline, time <= horizon means within window)
    y = ((df[time_col] > 30/365.25) & (df[time_col] <= horizon_years)).astype(int)
    return y


def filter_for_future_prediction(df, diag_type):
    """
    Filter out patients already diagnosed at baseline for future prediction.
    Keep only those with time_to_diagnosis > 0 or NaN (never diagnosed).
    """
    time_col = f"{diag_type}_time_to_diagnosis"
    if time_col not in df.columns:
        return df
    mask = (df[time_col] > 0) | (df[time_col].isna())
    return df[mask].copy()


def get_feature_set(feature_set_name, olink_cols, blood_cols, demo_cols, 
                    top_features=None):
    """
    Get feature columns for a given feature set configuration.
    
    Args:
        feature_set_name: Name of feature set configuration
        olink_cols: List of protein feature columns
        blood_cols: List of blood biomarker columns
        demo_cols: List of demographic columns
        top_features: Dict with 'protein', 'blood', 'combined' top feature lists
    
    Returns:
        List of feature column names
    """
    if feature_set_name == 'demo_protein':
        return demo_cols + olink_cols
    
    elif feature_set_name == 'demo_blood':
        return demo_cols + blood_cols
    
    elif feature_set_name == 'demo_protein_blood':
        return demo_cols + olink_cols + blood_cols
    
    elif feature_set_name == 'demo_protein_top50':
        if top_features and 'protein' in top_features:
            return demo_cols + top_features['protein'][:50]
        return demo_cols + olink_cols[:50]  # fallback
    
    elif feature_set_name == 'demo_blood_top50':
        if top_features and 'blood' in top_features:
            # Blood might have fewer than 50 features
            return demo_cols + top_features['blood'][:50]
        return demo_cols + blood_cols[:50]  # fallback
    
    elif feature_set_name == 'demo_protein_blood_top50':
        if top_features and 'combined' in top_features:
            return demo_cols + top_features['combined'][:50]
        return demo_cols + (olink_cols + blood_cols)[:50]  # fallback
    
    else:
        raise ValueError(f"Unknown feature set: {feature_set_name}")


def select_top_features(model, feature_cols, olink_cols, blood_cols, n_top=50):
    """
    Select top N features based on XGBoost feature importance.
    
    Returns dict with:
        'protein': top N protein features
        'blood': top N blood features  
        'combined': top N combined features
    """
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Top protein features
    protein_importance = importance_df[
        importance_df['feature'].isin(olink_cols)
    ]
    top_protein = protein_importance.head(n_top)['feature'].tolist()
    
    # Top blood features
    blood_importance = importance_df[
        importance_df['feature'].isin(blood_cols)
    ]
    top_blood = blood_importance.head(n_top)['feature'].tolist()
    
    # Top combined (excluding demographics)
    combined_importance = importance_df[
        importance_df['feature'].isin(olink_cols + blood_cols)
    ]
    top_combined = combined_importance.head(n_top)['feature'].tolist()
    
    return {
        'protein': top_protein,
        'blood': top_blood,
        'combined': top_combined
    }


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Train XGBoost model and return test AUC.
    
    Returns:
        auc: Test AUC score
        model: Trained XGBoost model
    """
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    # Train model
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train_imp, y_train)
    
    # Evaluate
    y_pred = model.predict_proba(X_test_imp)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    
    return auc, model


# =============================================================================
# Main Benchmark Function
# =============================================================================

def run_benchmark(train_df, test_df, output_dir='.'):
    """
    Run complete benchmark across all diagnosis types, time points, and feature sets.
    
    Returns:
        DataFrame with columns: diag_type, time_point, feature_set, auc, n_train, 
                               n_test, n_pos_train, n_pos_test, status
    """
    set_seeds(RANDOM_STATE)
    
    # Get feature columns
    olink_cols, blood_cols, demo_cols = get_feature_columns(train_df)
    print(f"\nFeature counts: {len(olink_cols)} protein, {len(blood_cols)} blood, "
          f"{len(demo_cols)} demo")
    
    # Get diagnosis types
    diag_types = get_diagnosis_types(train_df)
    print(f"Diagnosis types ({len(diag_types)}): {diag_types}")
    
    # Results storage
    results = []
    top_features_cache = {}  # Cache top features per (diag_type, time_point)
    
    # Total combinations
    total_combos = len(diag_types) * len(TIME_POINTS) * len(FEATURE_SETS)
    combo_count = 0
    
    print(f"\nRunning {total_combos} combinations...")
    print("=" * 80)
    
    for diag_type in diag_types:
        print(f"\n{'='*60}")
        print(f"Diagnosis: {diag_type}")
        print(f"{'='*60}")
        
        for time_point in TIME_POINTS:
            print(f"\n  Time point: {time_point}yr")
            print(f"  {'-'*50}")
            
            # Get appropriate labels based on time point
            if time_point == 0:
                # Current diagnosis - use all data
                train_filtered = train_df.copy()
                test_filtered = test_df.copy()
                y_train = get_labels_current(train_filtered, diag_type)
                y_test = get_labels_current(test_filtered, diag_type)
            else:
                # Future diagnosis - filter out already diagnosed
                train_filtered = filter_for_future_prediction(train_df, diag_type)
                test_filtered = filter_for_future_prediction(test_df, diag_type)
                y_train = get_labels_future(train_filtered, diag_type, time_point)
                y_test = get_labels_future(test_filtered, diag_type, time_point)
            
            # Skip if insufficient samples
            if y_train is None or y_test is None:
                print(f"    Skipping: No time-to-diagnosis column")
                for fs in FEATURE_SETS:
                    results.append({
                        'diag_type': diag_type,
                        'time_point': time_point,
                        'feature_set': fs,
                        'auc': np.nan,
                        'n_train': 0,
                        'n_test': 0,
                        'n_pos_train': 0,
                        'n_pos_test': 0,
                        'status': 'no_column'
                    })
                continue
            
            n_pos_train = int(y_train.sum())
            n_pos_test = int(y_test.sum())
            
            if n_pos_train < 3 or n_pos_test < 3:
                print(f"    Skipping: Insufficient positives "
                      f"(train={n_pos_train}, test={n_pos_test})")
                for fs in FEATURE_SETS:
                    results.append({
                        'diag_type': diag_type,
                        'time_point': time_point,
                        'feature_set': fs,
                        'auc': np.nan,
                        'n_train': len(y_train),
                        'n_test': len(y_test),
                        'n_pos_train': n_pos_train,
                        'n_pos_test': n_pos_test,
                        'status': 'insufficient_positives'
                    })
                continue
            
            print(f"    Train: {len(y_train)} (pos={n_pos_train}, "
                  f"{n_pos_train/len(y_train)*100:.2f}%)")
            print(f"    Test: {len(y_test)} (pos={n_pos_test}, "
                  f"{n_pos_test/len(y_test)*100:.2f}%)")
            
            # First, train on full feature set to get top features
            cache_key = (diag_type, time_point)
            if cache_key not in top_features_cache:
                # Train with all features to determine top features
                all_features = demo_cols + olink_cols + blood_cols
                X_train_all = train_filtered[all_features].copy()
                X_test_all = test_filtered[all_features].copy()
                
                _, full_model = train_and_evaluate(
                    X_train_all, X_test_all, y_train, y_test
                )
                
                # Extract top features
                top_features_cache[cache_key] = select_top_features(
                    full_model, all_features, olink_cols, blood_cols, n_top=50
                )
            
            top_features = top_features_cache[cache_key]
            
            # Now evaluate each feature set
            for feature_set in FEATURE_SETS:
                combo_count += 1
                
                try:
                    # Get feature columns for this configuration
                    feature_cols = get_feature_set(
                        feature_set, olink_cols, blood_cols, demo_cols, top_features
                    )
                    
                    # Prepare data
                    X_train = train_filtered[feature_cols].copy()
                    X_test = test_filtered[feature_cols].copy()
                    
                    # Train and evaluate
                    auc, _ = train_and_evaluate(X_train, X_test, y_train, y_test)
                    
                    print(f"    {feature_set}: AUC = {auc:.4f} "
                          f"({len(feature_cols)} features)")
                    
                    results.append({
                        'diag_type': diag_type,
                        'time_point': time_point,
                        'feature_set': feature_set,
                        'auc': auc,
                        'n_train': len(y_train),
                        'n_test': len(y_test),
                        'n_pos_train': n_pos_train,
                        'n_pos_test': n_pos_test,
                        'n_features': len(feature_cols),
                        'status': 'completed'
                    })
                    
                except Exception as e:
                    print(f"    {feature_set}: ERROR - {str(e)}")
                    results.append({
                        'diag_type': diag_type,
                        'time_point': time_point,
                        'feature_set': feature_set,
                        'auc': np.nan,
                        'n_train': len(y_train),
                        'n_test': len(y_test),
                        'n_pos_train': n_pos_train,
                        'n_pos_test': n_pos_test,
                        'status': f'error: {str(e)}'
                    })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df, top_features_cache


def create_summary_tables(results_df, output_dir='.'):
    """
    Create and save summary tables in various formats.
    """
    # Filter to completed results
    completed = results_df[results_df['status'] == 'completed'].copy()
    
    if len(completed) == 0:
        print("No completed results to summarize.")
        return
    
    # 1. Pivot table: rows = (diag_type, time_point), columns = feature_set
    pivot_by_features = completed.pivot_table(
        index=['diag_type', 'time_point'],
        columns='feature_set',
        values='auc',
        aggfunc='first'
    )
    
    # 2. Pivot table: rows = (diag_type, feature_set), columns = time_point
    pivot_by_time = completed.pivot_table(
        index=['diag_type', 'feature_set'],
        columns='time_point',
        values='auc',
        aggfunc='first'
    )
    
    # 3. Summary statistics by feature set
    summary_by_features = completed.groupby('feature_set')['auc'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).round(4)
    
    # 4. Summary statistics by time point
    summary_by_time = completed.groupby('time_point')['auc'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).round(4)
    
    # 5. Summary statistics by diagnosis
    summary_by_diag = completed.groupby('diag_type')['auc'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).round(4)
    
    # Save all tables
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Full results
    results_df.to_csv(f'{output_dir}/benchmark_results_full_{timestamp}.csv', index=False)
    
    # Pivot tables
    pivot_by_features.to_csv(f'{output_dir}/benchmark_pivot_by_features_{timestamp}.csv')
    pivot_by_time.to_csv(f'{output_dir}/benchmark_pivot_by_time_{timestamp}.csv')
    
    # Summary tables
    summary_by_features.to_csv(f'{output_dir}/benchmark_summary_by_features_{timestamp}.csv')
    summary_by_time.to_csv(f'{output_dir}/benchmark_summary_by_time_{timestamp}.csv')
    summary_by_diag.to_csv(f'{output_dir}/benchmark_summary_by_diagnosis_{timestamp}.csv')
    
    # Also save a "latest" version for easy access
    results_df.to_csv(f'{output_dir}/benchmark_results_latest.csv', index=False)
    pivot_by_features.to_csv(f'{output_dir}/benchmark_pivot_by_features_latest.csv')
    pivot_by_time.to_csv(f'{output_dir}/benchmark_pivot_by_time_latest.csv')
    
    print(f"\n{'='*60}")
    print("SUMMARY TABLES SAVED")
    print(f"{'='*60}")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - benchmark_results_full_{timestamp}.csv")
    print(f"  - benchmark_pivot_by_features_{timestamp}.csv")
    print(f"  - benchmark_pivot_by_time_{timestamp}.csv")
    print(f"  - benchmark_summary_by_features_{timestamp}.csv")
    print(f"  - benchmark_summary_by_time_{timestamp}.csv")
    print(f"  - benchmark_summary_by_diagnosis_{timestamp}.csv")
    print(f"  - benchmark_results_latest.csv (for easy access)")
    
    # Print summary tables
    print(f"\n{'='*60}")
    print("SUMMARY BY FEATURE SET")
    print(f"{'='*60}")
    print(summary_by_features.to_string())
    
    print(f"\n{'='*60}")
    print("SUMMARY BY TIME POINT")
    print(f"{'='*60}")
    print(summary_by_time.to_string())
    
    print(f"\n{'='*60}")
    print("SUMMARY BY DIAGNOSIS")
    print(f"{'='*60}")
    print(summary_by_diag.to_string())
    
    return {
        'pivot_by_features': pivot_by_features,
        'pivot_by_time': pivot_by_time,
        'summary_by_features': summary_by_features,
        'summary_by_time': summary_by_time,
        'summary_by_diag': summary_by_diag
    }


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='XGBoost Benchmark for Diagnosis Prediction'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='.',
        help='Output directory for results (default: current directory)'
    )
    parser.add_argument(
        '--data-path', '-d',
        type=str,
        default=DATA_PATH,
        help=f'Path to data directory (default: {DATA_PATH})'
    )
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("=" * 60)
    print("XGBoost Benchmark for Diagnosis Prediction")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Random seed: {RANDOM_STATE}")
    print(f"  Data path: {args.data_path}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Time points: {TIME_POINTS}")
    print(f"  Feature sets: {FEATURE_SETS}")
    print(f"\nXGBoost parameters:")
    for k, v in XGB_PARAMS.items():
        print(f"  {k}: {v}")
    
    # Load data
    train_df, test_df = load_data(args.data_path)
    
    # Run benchmark
    results_df, top_features = run_benchmark(
        train_df, test_df, output_dir=args.output_dir
    )
    
    # Create summary tables
    summaries = create_summary_tables(results_df, output_dir=args.output_dir)
    
    # Save top features
    top_features_list = []
    for (diag, time), features in top_features.items():
        for feat_type, feat_list in features.items():
            for rank, feat in enumerate(feat_list, 1):
                top_features_list.append({
                    'diag_type': diag,
                    'time_point': time,
                    'feature_type': feat_type,
                    'rank': rank,
                    'feature': feat
                })
    
    if top_features_list:
        top_features_df = pd.DataFrame(top_features_list)
        top_features_df.to_csv(
            f'{args.output_dir}/top_features_per_condition.csv', 
            index=False
        )
        print(f"\nSaved top features to: {args.output_dir}/top_features_per_condition.csv")
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    
    return results_df, summaries


if __name__ == '__main__':
    main()
