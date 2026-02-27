#!/usr/bin/env python3
"""
Multi-Model Benchmark for Diagnosis Prediction

Systematically evaluates multiple ML models across all combinations of:
- Models: XGBoost, Logistic Regression, MLP
- Diagnosis types (11 types)
- Time points: 0yr (current), 1yr, 2yr, 5yr, 10yr
- Feature sets: demo+protein, demo+blood, demo+protein+blood, and top-50 variants

Outputs a comprehensive AUC table for all combinations.

Usage:
    python run_model_benchmark.py --model xgboost    # XGBoost only
    python run_model_benchmark.py --model logreg     # Logistic Regression only
    python run_model_benchmark.py --model mlp        # MLP only
    python run_model_benchmark.py --model all        # All models (default)
"""

import argparse
import os
import random
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

# Import XGBoost (optional - only needed if running XGBoost)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: XGBoost not available")


# =============================================================================
# Configuration
# =============================================================================

# Random seed for reproducibility
RANDOM_STATE = 42

# Data path
DATA_PATH = "/orcd/pool/003/dbertsim_shared/ukb/"

# Time points to evaluate (in years)
TIME_POINTS = [0, 1, 2, 5, 10]

# Feature set configurations - base models first, then top50 variants
FEATURE_SETS = [
    'demo_protein',           # Demographics + Protein features (base)
    'demo_blood',             # Demographics + Blood features (base)
    'demo_protein_blood',     # Demographics + Protein + Blood (all features)
    'demo_protein_top50',     # Demographics + Top 50 protein from demo_protein
    # 'demo_blood_top50',       # Demographics + Top 50 blood from demo_blood
    'demo_protein_blood_top50'  # Demographics + Top 50 protein + Top 50 blood
]

# Base feature sets that provide importances for top50 variants
BASE_FEATURE_SETS = ['demo_protein', 'demo_blood']
TOP50_FEATURE_SETS = ['demo_protein_top50', 'demo_protein_blood_top50']

# Demographic columns
DEMO_COLS = [
    'Age at recruitment',
    'Sex_male',
    'Body mass index (BMI)',
    'Systolic blood pressure, automated reading',
    'Diastolic blood pressure, automated reading',
    'Townsend deprivation index at recruitment',
]

# Available models
MODEL_TYPES = ['xgboost', 'logreg', 'mlp']

# XGBoost parameters
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

# Logistic Regression parameters
LOGREG_PARAMS = {
    'penalty': 'l2',
    'C': 1.0,
    'solver': 'lbfgs',
    'max_iter': 1000,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
}

# MLP parameters
MLP_PARAMS = {
    'hidden_layer_sizes': (128, 64, 32),  # 3-layer MLP
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.0001,  # L2 regularization
    'batch_size': 256,
    'learning_rate': 'adaptive',
    'learning_rate_init': 0.001,
    'max_iter': 500,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 20,
    'random_state': RANDOM_STATE,
}


# =============================================================================
# Utility Functions
# =============================================================================

def set_seeds(seed=RANDOM_STATE):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_data():
    """Load train and test datasets."""
    print("Loading data...")
    train_df = pd.read_csv(f"{DATA_PATH}ukb_diag_train.csv", low_memory=False)
    test_df = pd.read_csv(f"{DATA_PATH}ukb_diag_test.csv", low_memory=False)
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
    Positive: diagnosed at or before baseline (time_to_diagnosis <= 30 days).
    """
    # Already diagnosed at baseline
    y = df[diag_type]
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
    y = ((df[diag_type] == 0) & (df[time_col] <= horizon_years)).astype(int)
    return y


def filter_for_future_prediction(df, diag_type):
    """
    Filter out patients already diagnosed at baseline for future prediction.
    Keep only those with time_to_diagnosis > 30 days (never diagnosed).
    """
    mask = (df[diag_type] == 0)
    return df[mask].copy()


def get_feature_set(feature_set_name, olink_cols, blood_cols, demo_cols, 
                    top_features=None):
    """
    Get feature columns for a given feature set configuration.
    
    top_features dict structure:
        'demo_protein': list of top protein features from demo_protein model
        'demo_blood': list of top blood features from demo_blood model
    """
    if feature_set_name == 'demo_protein':
        return demo_cols + olink_cols
    
    elif feature_set_name == 'demo_blood':
        return demo_cols + blood_cols
    
    elif feature_set_name == 'demo_protein_blood':
        return demo_cols + olink_cols + blood_cols
    
    elif feature_set_name == 'demo_protein_top50':
        # Top 50 protein features from demo_protein model + demo
        if top_features and 'demo_protein' in top_features:
            return demo_cols + top_features['demo_protein'][:50]
        return demo_cols + olink_cols[:50]
    
    elif feature_set_name == 'demo_blood_top50':
        # Top 50 blood features from demo_blood model + demo
        if top_features and 'demo_blood' in top_features:
            return demo_cols + top_features['demo_blood'][:50]
        return demo_cols + blood_cols[:50]
    
    elif feature_set_name == 'demo_protein_blood_top50':
        # Top 50 protein + top 50 blood features (from respective models) + demo
        top_protein = top_features.get('demo_protein', olink_cols)[:50] if top_features else olink_cols[:50]
        top_blood = top_features.get('demo_blood', blood_cols)[:50] if top_features else blood_cols[:50]
        return demo_cols + top_protein + top_blood
    
    else:
        raise ValueError(f"Unknown feature set: {feature_set_name}")


def extract_top_features_from_model(model, feature_cols, target_cols, model_type, n_top=50):
    """
    Extract top N features from a trained model based on feature importance.
    Only returns features that are in target_cols (e.g., olink_cols or blood_cols).
    """
    # Get feature importances based on model type
    if model_type == 'xgboost':
        importance = model.feature_importances_
    elif model_type == 'logreg':
        importance = np.abs(model.coef_[0])
    elif model_type == 'mlp':
        # Use first layer weights magnitude as proxy
        importance = np.abs(model.coefs_[0]).sum(axis=1)
    else:
        importance = np.ones(len(feature_cols))
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Filter to only target feature type (e.g., protein or blood)
    filtered = importance_df[importance_df['feature'].isin(target_cols)]
    return filtered.head(n_top)['feature'].tolist()


def get_model(model_type):
    """
    Create and return a model instance based on model type.
    """
    if model_type == 'xgboost':
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is not installed")
        return xgb.XGBClassifier(**XGB_PARAMS)
    
    elif model_type == 'logreg':
        return LogisticRegression(**LOGREG_PARAMS)
    
    elif model_type == 'mlp':
        return MLPClassifier(**MLP_PARAMS)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_and_evaluate(X_train, X_test, y_train, y_test, model_type='xgboost'):
    """
    Train model and return test AUC.
    
    Returns:
        auc: Test AUC score
        model: Trained model
    """
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    # For LogReg and MLP, we need to scale the features
    if model_type in ['logreg', 'mlp']:
        scaler = StandardScaler()
        X_train_imp = scaler.fit_transform(X_train_imp)
        X_test_imp = scaler.transform(X_test_imp)
    
    # Train model
    model = get_model(model_type)
    model.fit(X_train_imp, y_train)
    
    # Evaluate
    y_pred = model.predict_proba(X_test_imp)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    
    return auc, model


# =============================================================================
# Main Benchmark Function
# =============================================================================

def run_benchmark(train_df, test_df, model_types):
    """
    Run complete benchmark across all diagnosis types, time points, feature sets,
    and model types.
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
    
    # Total combinations
    total_combos = len(diag_types) * len(TIME_POINTS) * len(FEATURE_SETS) * len(model_types)
    combo_count = 0
    
    print(f"\nRunning {total_combos} combinations...")
    print(f"Models: {model_types}")
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
                train_filtered = train_df.copy()
                test_filtered = test_df.copy()
                y_train = get_labels_current(train_filtered, diag_type)
                y_test = get_labels_current(test_filtered, diag_type)
            else:
                train_filtered = filter_for_future_prediction(train_df, diag_type)
                test_filtered = filter_for_future_prediction(test_df, diag_type)
                y_train = get_labels_future(train_filtered, diag_type, time_point)
                y_test = get_labels_future(test_filtered, diag_type, time_point)
            
            # Skip if insufficient samples
            if y_train is None or y_test is None:
                print(f"    Skipping: No time-to-diagnosis column")
                for fs in FEATURE_SETS:
                    for model_type in model_types:
                        results.append({
                            'model': model_type,
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
                    for model_type in model_types:
                        results.append({
                            'model': model_type,
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
            
            # Evaluate each model and feature set combination
            for model_type in model_types:
                print(f"\n    Model: {model_type.upper()}")
                
                # Store base models to extract feature importances
                base_models = {}
                top_features = {}
                
                for feature_set in FEATURE_SETS:
                    combo_count += 1
                    
                    try:
                        # Get feature columns
                        feature_cols = get_feature_set(
                            feature_set, olink_cols, blood_cols, demo_cols, top_features
                        )
                        
                        # Prepare data
                        X_train = train_filtered[feature_cols].copy()
                        X_test = test_filtered[feature_cols].copy()
                        
                        # Train and evaluate
                        auc, model = train_and_evaluate(
                            X_train, X_test, y_train, y_test, model_type
                        )
                        
                        # model.save_model(f"{models_dir}{model_type}_model_{diag_type}_{time_point}_{feature_set}.json")
                        # print(f"Saved model to {models_dir}{model_type}_model_{diag_type}_{time_point}_{feature_set}.json")

                        # Cache base models and extract top features for top50 variants
                        if feature_set in BASE_FEATURE_SETS:
                            base_models[feature_set] = model
                            if feature_set == 'demo_protein':
                                top_features['demo_protein'] = extract_top_features_from_model(
                                    model, feature_cols, olink_cols, model_type, n_top=50
                                )
                            elif feature_set == 'demo_blood':
                                top_features['demo_blood'] = extract_top_features_from_model(
                                    model, feature_cols, blood_cols, model_type, n_top=50
                                )
                        
                        print(f"      {feature_set}: AUC = {auc:.4f} "
                              f"({len(feature_cols)} features)")
                        
                        results.append({
                            'model': model_type,
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
                        print(f"      {feature_set}: ERROR - {str(e)}")
                        results.append({
                            'model': model_type,
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
    
    return results_df


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Model Benchmark for Diagnosis Prediction'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='all',
        choices=['xgboost', 'logreg', 'mlp', 'all'],
        help='Model type to benchmark (default: all)'
    )

    args = parser.parse_args()
    
    # Determine which models to run
    if args.model == 'all':
        model_types = MODEL_TYPES.copy()
        if not XGB_AVAILABLE:
            model_types.remove('xgboost')
            print("Warning: XGBoost not available, skipping")
    else:
        if args.model == 'xgboost' and not XGB_AVAILABLE:
            raise ImportError("XGBoost is not installed")
        model_types = [args.model]
    
    # Print configuration
    print("=" * 60)
    print("Multi-Model Benchmark for Diagnosis Prediction")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Random seed: {RANDOM_STATE}")
    print(f"  Models: {model_types}")
    print(f"  Time points: {TIME_POINTS}")
    print(f"  Feature sets: {FEATURE_SETS}")
    
    print(f"\nModel parameters:")
    if 'xgboost' in model_types:
        print(f"\n  XGBoost:")
        for k, v in XGB_PARAMS.items():
            print(f"    {k}: {v}")
    if 'logreg' in model_types:
        print(f"\n  Logistic Regression:")
        for k, v in LOGREG_PARAMS.items():
            print(f"    {k}: {v}")
    if 'mlp' in model_types:
        print(f"\n  MLP:")
        for k, v in MLP_PARAMS.items():
            print(f"    {k}: {v}")
    
    # Load data
    train_df, test_df = load_data()
    
    # Run benchmark
    results_df = run_benchmark(train_df, test_df, model_types)
    results_df.to_csv('results/model_benchmark_results.csv', index=False)
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    
    return results_df


if __name__ == '__main__':
    main()
