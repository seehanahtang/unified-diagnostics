#!/usr/bin/env python3
"""
XGBoost experiment with specific features excluded.

For kidney: excludes blood_Cystatin C, blood_Urea, blood_Creatinine
For t2d: excludes blood_Glycated haemoglobin (HbA1c), blood_Cholesterol, blood_Glucose, blood_Triglycerides

Time point: 0yr only
Feature sets: demo_blood, demo_protein_blood
"""

import os
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

RANDOM_STATE = 42
DATA_PATH = "/orcd/pool/003/dbertsim_shared/ukb/"

# Features to exclude per diagnosis
EXCLUDED_FEATURES = {
    'kidney': [
        'blood_Cystatin C',
        'blood_Urea', 
        'blood_Creatinine'
    ],
    't2d': [
        'blood_Glycated haemoglobin (HbA1c)',
        'blood_Cholesterol',
        'blood_Glucose',
        'blood_Triglycerides'
    ]     
}

# Demographic columns
DEMO_COLS = [
    'Age at recruitment',
    'Sex_male',
    'Body mass index (BMI)',
    'Systolic blood pressure, automated reading',
    'Diastolic blood pressure, automated reading',
    'Townsend deprivation index at recruitment',
]

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


def set_seeds(seed=RANDOM_STATE):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_labels_current(df, diag_type):
    """Get binary labels for current diagnosis (time_point=0)."""
    y = (df[diag_type] == 1).astype(int)
    return y


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    """Train XGBoost model and return test AUC, model, and top features."""
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train_imp, y_train)
    
    y_pred = model.predict_proba(X_test_imp)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    
    # Get feature importances
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return auc, model, importance_df


def main():
    set_seeds(RANDOM_STATE)
    
    print("=" * 70)
    print("XGBoost Experiment: Excluded Features (0yr)")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(f"{DATA_PATH}ukb_diag_train.csv", low_memory=False)
    test_df = pd.read_csv(f"{DATA_PATH}ukb_diag_test.csv", low_memory=False)
    print(f"  Train: {len(train_df)} rows")
    print(f"  Test: {len(test_df)} rows")
    
    # Get feature columns
    olink_cols = [c for c in train_df.columns if c.startswith('olink_')]
    blood_cols = [c for c in train_df.columns if c.startswith('blood_')]
    demo_cols = [c for c in DEMO_COLS if c in train_df.columns]
    
    print(f"\nBase features: {len(olink_cols)} protein, {len(blood_cols)} blood, {len(demo_cols)} demo")
    
    # Results storage
    results = []
    top_features_all = []
    
    for diag_type in ['kidney', 't2d']:
        print(f"\n{'='*60}")
        print(f"Diagnosis: {diag_type}")
        print(f"{'='*60}")
        
        # Get excluded features for this diagnosis
        excluded = EXCLUDED_FEATURES[diag_type]
        print(f"\nExcluded features: {excluded}")
        
        # Filter blood columns
        blood_cols_filtered = [c for c in blood_cols if c not in excluded]
        print(f"Blood features: {len(blood_cols)} -> {len(blood_cols_filtered)} (removed {len(blood_cols) - len(blood_cols_filtered)})")
        
        # Get labels
        y_train = get_labels_current(train_df, diag_type)
        y_test = get_labels_current(test_df, diag_type)
        
        n_pos_train = int(y_train.sum())
        n_pos_test = int(y_test.sum())
        print(f"\nTrain: {len(y_train)} samples, {n_pos_train} positive ({n_pos_train/len(y_train)*100:.2f}%)")
        print(f"Test: {len(y_test)} samples, {n_pos_test} positive ({n_pos_test/len(y_test)*100:.2f}%)")
        
        # Feature sets to test
        feature_sets = {
            'demo_blood': demo_cols + blood_cols_filtered,
            'demo_protein_blood': demo_cols + olink_cols + blood_cols_filtered,
            # Also run with original features for comparison
            'demo_blood_original': demo_cols + blood_cols,
            'demo_protein_blood_original': demo_cols + olink_cols + blood_cols,
        }
        
        print(f"\n{'Feature Set':<35} {'# Features':>12} {'AUC':>10}")
        print("-" * 60)
        
        for fs_name, feature_cols in feature_sets.items():
            X_train = train_df[feature_cols].copy()
            X_test = test_df[feature_cols].copy()
            
            auc, model, importance_df = train_and_evaluate(
                X_train, X_test, y_train, y_test, feature_cols
            )
            
            is_excluded = 'excluded' if 'original' not in fs_name else 'original'
            print(f"{fs_name:<35} {len(feature_cols):>12} {auc:>10.4f}")
            
            # Store top 20 features
            top_20 = importance_df.head(20).copy()
            top_20['rank'] = range(1, 21)
            top_20['diag_type'] = diag_type
            top_20['feature_set'] = fs_name
            top_20['auc'] = auc
            top_features_all.append(top_20)
            
            results.append({
                'diag_type': diag_type,
                'feature_set': fs_name,
                'n_features': len(feature_cols),
                'auc': auc,
                'n_train': len(y_train),
                'n_test': len(y_test),
                'n_pos_train': n_pos_train,
                'n_pos_test': n_pos_test,
                'excluded_features': ', '.join(excluded) if 'original' not in fs_name else 'none'
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Combine all top features
    top_features_df = pd.concat(top_features_all, ignore_index=True)
    top_features_df = top_features_df[['diag_type', 'feature_set', 'auc', 'rank', 'feature', 'importance']]
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    output_path = 'results/excluded_features_experiment.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n\nResults saved to: {output_path}")
    
    top_features_path = 'results/excluded_features_top20.csv'
    top_features_df.to_csv(top_features_path, index=False)
    print(f"Top 20 features saved to: {top_features_path}")
    
    # Print comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    for diag_type in ['kidney', 't2d']:
        print(f"\n{diag_type.upper()}:")
        diag_results = results_df[results_df['diag_type'] == diag_type]
        
        for base_set in ['demo_blood', 'demo_protein_blood']:
            orig = diag_results[diag_results['feature_set'] == f'{base_set}_original']['auc'].values[0]
            excl = diag_results[diag_results['feature_set'] == base_set]['auc'].values[0]
            diff = excl - orig
            print(f"  {base_set}: {orig:.4f} (original) -> {excl:.4f} (excluded) | Δ = {diff:+.4f}")
    
    # Print top 20 features for each experiment
    print("\n" + "=" * 70)
    print("TOP 20 FEATURES PER EXPERIMENT")
    print("=" * 70)
    
    for diag_type in ['kidney', 't2d']:
        for fs_name in ['demo_blood', 'demo_protein_blood', 'demo_blood_original', 'demo_protein_blood_original']:
            mask = (top_features_df['diag_type'] == diag_type) & (top_features_df['feature_set'] == fs_name)
            subset = top_features_df[mask]
            
            if len(subset) > 0:
                auc = subset['auc'].iloc[0]
                print(f"\n{diag_type} | {fs_name} (AUC: {auc:.4f})")
                print("-" * 50)
                for _, row in subset.iterrows():
                    print(f"  {row['rank']:2d}. {row['feature']:<45} {row['importance']:.4f}")
    
    return results_df, top_features_df


if __name__ == '__main__':
    main()
