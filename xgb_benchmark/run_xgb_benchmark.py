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
models_dir = "models/"
os.makedirs(models_dir, exist_ok=True)

# Time points to evaluate (in years)
TIME_POINTS = [0, 1, 2, 5, 10]

# Feature set configurations
FEATURE_SETS = [
    'demo_protein',           # Demographics + Protein features
    'demo_blood',             # Demographics + Blood features  
    'demo_protein_blood',     # Demographics + Protein + Blood (all features)
    # 'demo_protein_top50',     # Demographics + Top 50 protein features
    # 'demo_blood_top50',       # Demographics + Top 50 blood features (or all if <50)
    'demo_protein_blood_top50'  # Demographics + Top 50 from protein+blood combined
]

BASE_FEATURE_SETS = ['demo_protein', 'demo_blood']
# TOP50_FEATURE_SETS = ['demo_protein_top50', 'demo_protein_blood_top50']
TOP50_FEATURE_SETS = ['demo_protein_blood_top50']

FEATURE_SETS = [
    'demo_protein_blood',     # Demographics + Protein + Blood (all features)
]

# Demographic columns
DEMO_COLS = [
    'Age at recruitment',
    'Sex_male',
    'Body mass index (BMI)',
    'Systolic blood pressure, automated reading',
    'Diastolic blood pressure, automated reading',
    'Townsend deprivation index at recruitment',
    'Smoking status', 
    'Alcohol intake frequency.'
]

# Demographic columns that should be treated as categorical
CATEGORICAL_DEMO_COLS = [
    'Smoking status',
    'Alcohol intake frequency.'
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
    """Load train, test, and external validation datasets."""
    print("Loading data...")
    train_df = pd.read_csv(f"{data_path}ukb_disease_train.csv", low_memory=False)
    valid_df = pd.read_csv(f"{data_path}ukb_disease_valid.csv", low_memory=False)
    test_df = pd.read_csv(f"{data_path}ukb_disease_test.csv", low_memory=False)
    external_df = pd.read_csv(f"{data_path}ukb_disease_scotland_wales.csv", low_memory=False)
    print(f"  Train: {len(train_df)} rows, {len(train_df.columns)} columns")
    print(f"  Valid: {len(valid_df)} rows, {len(valid_df.columns)} columns")
    print(f"  Test: {len(test_df)} rows, {len(test_df.columns)} columns")
    print(f"  External (Scotland/Wales): {len(external_df)} rows, {len(external_df.columns)} columns")
    return train_df, valid_df, test_df, external_df


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
        return demo_cols + top_protein + top_blood
    
    else:
        raise ValueError(f"Unknown feature set: {feature_set_name}")
        
def extract_top_features_from_model(model, feature_cols, target_cols, n_top=50):
    """
    Extract top N features from a trained model based on feature importance.
    Only returns features that are in target_cols (e.g., olink_cols or blood_cols).
    """
    # Get feature importances 
    importance = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Filter to only target feature type (e.g., protein or blood)
    filtered = importance_df[importance_df['feature'].isin(target_cols)]
    return filtered.head(n_top)['feature'].tolist()


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


def train_and_evaluate(X_train, X_test, y_train, y_test,
                       X_external=None, y_external=None):
    """
    Train XGBoost model and return test AUC and optional external validation AUC.
    
    Assumes any desired imputation has already been applied to X inputs.
    
    Returns:
        auc: Test AUC score
        external_auc: External validation AUC score (np.nan if not provided)
        model: Trained XGBoost model
    """
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)
    
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    
    external_auc = np.nan
    if X_external is not None and y_external is not None and len(y_external) > 0:
        y_ext_pred = model.predict_proba(X_external)[:, 1]
        if y_external.sum() >= 1 and y_external.sum() < len(y_external):
            external_auc = roc_auc_score(y_external, y_ext_pred)
    
    return auc, external_auc, model


# =============================================================================
# Main Benchmark Function
# =============================================================================

def run_benchmark(train_df, valid_df, test_df, external_df, output_dir='.', data_path=DATA_PATH):
    """
    Run complete benchmark across all diagnosis types, time points, and feature sets.
    
    Returns:
        DataFrame with columns: diag_type, time_point, feature_set, auc, external_auc,
                               n_train, n_test, n_external, n_pos_train, n_pos_test,
                               n_pos_external, status
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
        
        # For each diagnosis, collect risk predictions for each dataset and time point
        # Structure: risk_store[split][eid] -> {time_point: risk}
        risk_store = {
            'train': {},
            'valid': {},
            'test': {},
            'external': {},
        }
        
        for time_point in TIME_POINTS:
            print(f"\n  Time point: {time_point}yr")
            print(f"  {'-'*50}")
            
            # Get appropriate labels based on time point
            if time_point == 0:
                train_filtered = train_df.copy()
                test_filtered = test_df.copy()
                valid_filtered = valid_df.copy()
                external_filtered = external_df.copy()
                y_train = get_labels_current(train_filtered, diag_type)
                y_test = get_labels_current(test_filtered, diag_type)
                y_valid = get_labels_current(valid_filtered, diag_type)
                y_external = get_labels_current(external_filtered, diag_type)
            else:
                train_filtered = filter_for_future_prediction(train_df, diag_type)
                test_filtered = filter_for_future_prediction(test_df, diag_type)
                valid_filtered = filter_for_future_prediction(valid_df, diag_type)
                external_filtered = filter_for_future_prediction(external_df, diag_type)
                y_train = get_labels_future(train_filtered, diag_type, time_point)
                y_test = get_labels_future(test_filtered, diag_type, time_point)
                y_valid = get_labels_future(valid_filtered, diag_type, time_point)
                y_external = get_labels_future(external_filtered, diag_type, time_point)
            
            # Skip if insufficient samples
            if y_train is None or y_test is None or y_valid is None:
                print(f"    Skipping: No time-to-diagnosis column")
                for fs in FEATURE_SETS:
                    results.append({
                        'diag_type': diag_type,
                        'time_point': time_point,
                        'feature_set': fs,
                        'auc': np.nan,
                        'external_auc': np.nan,
                        'n_train': 0,
                        'n_test': 0,
                        'n_external': 0,
                        'n_pos_train': 0,
                        'n_pos_test': 0,
                        'n_pos_external': 0,
                        'status': 'no_column'
                    })
                continue
            
            n_pos_train = int(y_train.sum())
            n_pos_test = int(y_test.sum())
            n_pos_external = int(y_external.sum()) if y_external is not None else 0
            
            if n_pos_train < 3 or n_pos_test < 3:
                print(f"    Skipping: Insufficient positives "
                      f"(train={n_pos_train}, test={n_pos_test})")
                for fs in FEATURE_SETS:
                    results.append({
                        'diag_type': diag_type,
                        'time_point': time_point,
                        'feature_set': fs,
                        'auc': np.nan,
                        'external_auc': np.nan,
                        'n_train': len(y_train),
                        'n_test': len(y_test),
                        'n_external': len(y_external) if y_external is not None else 0,
                        'n_pos_train': n_pos_train,
                        'n_pos_test': n_pos_test,
                        'n_pos_external': n_pos_external,
                        'status': 'insufficient_positives'
                    })
                continue
            
            print(f"    Train: {len(y_train)} (pos={n_pos_train}, "
                  f"{n_pos_train/len(y_train)*100:.2f}%)")
            print(f"    Test: {len(y_test)} (pos={n_pos_test}, "
                  f"{n_pos_test/len(y_test)*100:.2f}%)")
            if y_external is not None and len(y_external) > 0:
                print(f"    External: {len(y_external)} (pos={n_pos_external}, "
                      f"{n_pos_external/len(y_external)*100:.2f}%)")
            
            top_features = {}
            
            # Now evaluate each feature set
            for feature_set in FEATURE_SETS:
                combo_count += 1
                
                try:
                    feature_cols = get_feature_set(
                        feature_set, olink_cols, blood_cols, demo_cols, top_features
                    )
                    
                    X_train = train_filtered[feature_cols].copy()
                    X_valid = valid_filtered[feature_cols].copy()
                    X_test = test_filtered[feature_cols].copy()
                    
                    X_external = None
                    if y_external is not None and len(external_filtered) > 0:
                        X_external = external_filtered[feature_cols].copy()

                    # Ensure selected demographic categorical columns use pandas category dtype
                    for c in CATEGORICAL_DEMO_COLS:
                        X_train[c] = X_train[c].astype('category')
                        X_valid[c] = X_valid[c].astype('category')
                        X_test[c] = X_test[c].astype('category')
                        if X_external is not None:
                            X_external[c] = X_external[c].astype('category')

                    # Only impute numeric biomarker columns (olink_*, blood_*)
                    numeric_cols = [
                        c for c in feature_cols
                        if (c in olink_cols) or (c in blood_cols)
                    ]
                    if numeric_cols:
                        imputer = SimpleImputer(strategy='median')
                        X_train[numeric_cols] = imputer.fit_transform(
                            X_train[numeric_cols]
                        )
                        X_valid[numeric_cols] = imputer.transform(
                            X_valid[numeric_cols]
                        )
                        X_test[numeric_cols] = imputer.transform(
                            X_test[numeric_cols]
                        )
                        if X_external is not None:
                            X_external[numeric_cols] = imputer.transform(
                                X_external[numeric_cols]
                            )

                    auc, external_auc, model = train_and_evaluate(
                        X_train, X_test, y_train, y_test,
                        X_external=X_external, y_external=y_external
                    )

                    model.save_model(f"{models_dir}xgb_model_{diag_type}_{time_point}_{feature_set}.json")
                    print(f"Saved model to {models_dir}xgb_model_{diag_type}_{time_point}_{feature_set}.json")

                    # Collect risk predictions for the main feature set
                    # (currently 'demo_protein_blood' is the only feature set used)
                    if feature_set == 'demo_protein_blood':
                        # Train risks
                        train_proba = model.predict_proba(X_train)[:, 1]
                        for eid, p in zip(train_filtered['eid'], train_proba):
                            risk_store['train'].setdefault(eid, {})[time_point] = float(p)

                        # Validation risks
                        valid_proba = model.predict_proba(X_valid)[:, 1]
                        for eid, p in zip(valid_filtered['eid'], valid_proba):
                            risk_store['valid'].setdefault(eid, {})[time_point] = float(p)

                        # Test risks
                        test_proba = model.predict_proba(X_test)[:, 1]
                        for eid, p in zip(test_filtered['eid'], test_proba):
                            risk_store['test'].setdefault(eid, {})[time_point] = float(p)

                        # External risks (if available)
                        if X_external is not None and y_external is not None and len(y_external) > 0:
                            ext_proba = model.predict_proba(X_external)[:, 1]
                            for eid, p in zip(external_filtered['eid'], ext_proba):
                                risk_store['external'].setdefault(eid, {})[time_point] = float(p)
                    
                    if feature_set in BASE_FEATURE_SETS:
                        if feature_set == 'demo_protein':
                            top_features['demo_protein'] = extract_top_features_from_model(
                                model, feature_cols, olink_cols, n_top=50
                            )
                        elif feature_set == 'demo_blood':
                            top_features['demo_blood'] = extract_top_features_from_model(
                                model, feature_cols, blood_cols, n_top=50
                            )
                    
                    ext_str = f", External AUC = {external_auc:.4f}" if not np.isnan(external_auc) else ""
                    print(f"    {feature_set}: AUC = {auc:.4f}{ext_str} "
                          f"({len(feature_cols)} features)")
                    
                    results.append({
                        'diag_type': diag_type,
                        'time_point': time_point,
                        'feature_set': feature_set,
                        'auc': auc,
                        'external_auc': external_auc,
                        'n_train': len(y_train),
                        'n_test': len(y_test),
                        'n_external': len(y_external) if y_external is not None else 0,
                        'n_pos_train': n_pos_train,
                        'n_pos_test': n_pos_test,
                        'n_pos_external': n_pos_external,
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
                        'external_auc': np.nan,
                        'n_train': len(y_train),
                        'n_test': len(y_test),
                        'n_external': len(y_external) if y_external is not None else 0,
                        'n_pos_train': n_pos_train,
                        'n_pos_test': n_pos_test,
                        'n_pos_external': n_pos_external,
                        'status': f'error: {str(e)}'
                    })
    
        # After all time points and feature sets for this diagnosis, write risk files
        risk_base_dir = os.path.join(data_path, "disease_risk", "xgb_risk")
        os.makedirs(risk_base_dir, exist_ok=True)

        time_col_map = {t: f"risk_{t}years" for t in TIME_POINTS}

        for split_name, store in risk_store.items():
            if not store:
                continue

            rows = []
            for eid, risks in store.items():
                row = {"eid": eid}
                for t, col in time_col_map.items():
                    row[col] = risks.get(t, np.nan)
                rows.append(row)

            risk_df = pd.DataFrame(rows)
            risk_fname = f"{diag_type}_xgb_risk_{split_name}.csv"
            risk_path = os.path.join(risk_base_dir, risk_fname)
            risk_df.to_csv(risk_path, index=False)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df, top_features_cache


def save_results(results_df, output_dir='.'):
    """
    Save results to a CSV file.
    """
    
    # Save results to a CSV file
    results_df.to_csv(f'{output_dir}xgb_benchmark_results.csv', index=False)
   
    return results_df

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
    train_df, valid_df, test_df, external_df = load_data(args.data_path)
    
    # Run benchmark
    results_df, top_features = run_benchmark(
        train_df, valid_df, test_df, external_df,
        output_dir=args.output_dir,
        data_path=args.data_path,
    )
    
    # Create results file
    results_df = save_results(results_df, output_dir=args.output_dir)
    
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
            f'{args.output_dir}top_features_per_condition.csv', 
            index=False
        )
        print(f"\nSaved top features to: {args.output_dir}top_features_per_condition.csv")
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    
    return results_df


if __name__ == '__main__':
    main()