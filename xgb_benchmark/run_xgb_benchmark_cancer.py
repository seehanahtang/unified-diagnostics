#!/usr/bin/env python3
"""
XGBoost Benchmark for Cancer Prediction

Same as run_xgb_benchmark.py but uses the cancer dataset with sex-specific cohorts
(aligned with run_rsf.py):
- Breast, ovarian, uterine cancers: ukb_cancer_train/valid/test_female
- Prostate cancer: ukb_cancer_train/valid/test_male
- Other cancers: ukb_cancer_train/valid/test (default)

No external validation set for cancer. Writes risk files to {data_path}/cancer_risk/xgb_risk.

Usage:
    python run_xgb_benchmark_cancer.py [--output-dir OUTPUT_DIR] [--data-path DATA_PATH]
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

RANDOM_STATE = 42
DATA_PATH = "/orcd/pool/003/dbertsim_shared/ukb/"
models_dir = "models_cancer/"
os.makedirs(models_dir, exist_ok=True)

TIME_POINTS = [0, 1, 2, 5, 10]

FEATURE_SETS = [
    'demo_protein_blood',
]
BASE_FEATURE_SETS = ['demo_protein', 'demo_blood']

# Cancer outcomes that use sex-specific cohorts (same as run_rsf.py)
FEMALE_SPECIFIC_CANCERS = {"breast_cancer", "ovarian_cancer", "uterine_cancer"}
MALE_SPECIFIC_CANCERS = {"prostate_cancer"}

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

CATEGORICAL_DEMO_COLS = [
    'Smoking status',
    'Alcohol intake frequency.'
]

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
# Data loading (same dataset logic as run_rsf.py for cancer)
# =============================================================================

def get_cancer_data_suffix(diag_type: str) -> str:
    """Return '' for default, '_female' for female-specific, '_male' for male-specific."""
    key = diag_type.lower().replace(" ", "_").strip()
    if key in FEMALE_SPECIFIC_CANCERS:
        return "_female"
    if key in MALE_SPECIFIC_CANCERS:
        return "_male"
    return ""


def load_data_cancer(data_path=DATA_PATH):
    """Load cancer train/valid/test with default + female + male cohorts (same as run_rsf.py)."""
    data_path = data_path.rstrip("/") + "/"
    print("Loading cancer data...")
    train_df = pd.read_csv(f"{data_path}ukb_cancer_train.csv", low_memory=False)
    valid_df = pd.read_csv(f"{data_path}ukb_cancer_valid.csv", low_memory=False)
    test_df = pd.read_csv(f"{data_path}ukb_cancer_test.csv", low_memory=False)
    train_f = pd.read_csv(f"{data_path}ukb_cancer_train_female.csv", low_memory=False)
    valid_f = pd.read_csv(f"{data_path}ukb_cancer_valid_female.csv", low_memory=False)
    test_f = pd.read_csv(f"{data_path}ukb_cancer_test_female.csv", low_memory=False)
    train_m = pd.read_csv(f"{data_path}ukb_cancer_train_male.csv", low_memory=False)
    valid_m = pd.read_csv(f"{data_path}ukb_cancer_valid_male.csv", low_memory=False)
    test_m = pd.read_csv(f"{data_path}ukb_cancer_test_male.csv", low_memory=False)
    cancer_sets = {
        "": (train_df, valid_df, test_df),
        "_female": (train_f, valid_f, test_f),
        "_male": (train_m, valid_m, test_m),
    }
    print(f"  Default: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
    print(f"  Female:  train={len(train_f)}, valid={len(valid_f)}, test={len(test_f)}")
    print(f"  Male:    train={len(train_m)}, valid={len(valid_m)}, test={len(test_m)}")
    return cancer_sets


def get_cancer_types(df):
    """Extract cancer types from column names (columns ending with 'cancer')."""
    return [c for c in df.columns if c.endswith('cancer')]


# =============================================================================
# Utility Functions (same as run_xgb_benchmark.py)
# =============================================================================

def set_seeds(seed=RANDOM_STATE):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_feature_columns(df):
    olink_cols = [c for c in df.columns if c.startswith('olink_')]
    blood_cols = [c for c in df.columns if c.startswith('blood_')]
    demo_cols = [c for c in DEMO_COLS if c in df.columns]
    return olink_cols, blood_cols, demo_cols


def get_labels_current(df, diag_type):
    y = df[diag_type]
    return y


def get_labels_future(df, diag_type, horizon_years):
    time_col = f"{diag_type}_time_to_diagnosis"
    if time_col not in df.columns:
        return None
    y = ((df[diag_type] == 0) & (df[time_col] <= horizon_years)).astype(int)
    return y


def filter_for_future_prediction(df, diag_type):
    mask = (df[diag_type] == 0)
    return df[mask].copy()


def get_feature_set(feature_set_name, olink_cols, blood_cols, demo_cols, top_features=None):
    if feature_set_name == 'demo_protein':
        return demo_cols + olink_cols
    elif feature_set_name == 'demo_blood':
        return demo_cols + blood_cols
    elif feature_set_name == 'demo_protein_blood':
        return demo_cols + olink_cols + blood_cols
    elif feature_set_name == 'demo_protein_top50':
        if top_features and 'demo_protein' in top_features:
            return demo_cols + top_features['demo_protein'][:50]
        return demo_cols + olink_cols[:50]
    elif feature_set_name == 'demo_blood_top50':
        if top_features and 'demo_blood' in top_features:
            return demo_cols + top_features['demo_blood'][:50]
        return demo_cols + blood_cols[:50]
    elif feature_set_name == 'demo_protein_blood_top50':
        top_protein = top_features.get('demo_protein', olink_cols)[:50] if top_features else olink_cols[:50]
        top_blood = top_features.get('demo_blood', blood_cols)[:50] if top_features else blood_cols[:50]
        return demo_cols + top_protein + top_blood
    else:
        raise ValueError(f"Unknown feature set: {feature_set_name}")


def extract_top_features_from_model(model, feature_cols, target_cols, n_top=50):
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    filtered = importance_df[importance_df['feature'].isin(target_cols)]
    return filtered.head(n_top)['feature'].tolist()


def train_and_evaluate(X_train, X_test, y_train, y_test, X_external=None, y_external=None):
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
# Main Benchmark (cancer: sex-specific data per cancer type, no external)
# =============================================================================

def run_benchmark_cancer(cancer_sets, data_path=DATA_PATH, output_dir='.'):
    set_seeds(RANDOM_STATE)
    data_path = data_path.rstrip("/") + "/"

    # Use default cohort to get feature columns and list of all cancer types
    train_default, valid_default, test_default = cancer_sets[""]
    olink_cols, blood_cols, demo_cols = get_feature_columns(train_default)
    diag_types = get_cancer_types(train_default)
    print(f"\nFeature counts: {len(olink_cols)} protein, {len(blood_cols)} blood, {len(demo_cols)} demo")
    print(f"Cancer types ({len(diag_types)}): {diag_types}")

    results = []
    total_combos = len(diag_types) * len(TIME_POINTS) * len(FEATURE_SETS)
    print(f"\nRunning {total_combos} combinations...")
    print("=" * 80)

    for diag_type in diag_types:
        suffix = get_cancer_data_suffix(diag_type)
        train_df, valid_df, test_df = cancer_sets[suffix]
        if diag_type not in train_df.columns or f"{diag_type}_time_to_diagnosis" not in train_df.columns:
            print(f"\nSkipping {diag_type} (not in {suffix or 'default'} cohort)")
            for tp in TIME_POINTS:
                for fs in FEATURE_SETS:
                    results.append({
                        'diag_type': diag_type,
                        'time_point': tp,
                        'feature_set': fs,
                        'auc': np.nan,
                        'external_auc': np.nan,
                        'n_train': 0, 'n_test': 0, 'n_external': 0,
                        'n_pos_train': 0, 'n_pos_test': 0, 'n_pos_external': 0,
                        'status': 'cohort_skip'
                    })
            continue

        if suffix:
            print(f"\n{'='*60}\nCancer: {diag_type} (using {suffix} cohort)\n{'='*60}")
        else:
            print(f"\n{'='*60}\nCancer: {diag_type}\n{'='*60}")

        risk_store = {'train': {}, 'valid': {}, 'test': {}}

        for time_point in TIME_POINTS:
            print(f"\n  Time point: {time_point}yr")
            print(f"  {'-'*50}")

            if time_point == 0:
                train_filtered = train_df.copy()
                test_filtered = test_df.copy()
                valid_filtered = valid_df.copy()
                y_train = get_labels_current(train_filtered, diag_type)
                y_test = get_labels_current(test_filtered, diag_type)
                y_valid = get_labels_current(valid_filtered, diag_type)
            else:
                train_filtered = filter_for_future_prediction(train_df, diag_type)
                test_filtered = filter_for_future_prediction(test_df, diag_type)
                valid_filtered = filter_for_future_prediction(valid_df, diag_type)
                y_train = get_labels_future(train_filtered, diag_type, time_point)
                y_test = get_labels_future(test_filtered, diag_type, time_point)
                y_valid = get_labels_future(valid_filtered, diag_type, time_point)

            if y_train is None or y_test is None or y_valid is None:
                print(f"    Skipping: No time-to-diagnosis column")
                for fs in FEATURE_SETS:
                    results.append({
                        'diag_type': diag_type, 'time_point': time_point, 'feature_set': fs,
                        'auc': np.nan, 'external_auc': np.nan,
                        'n_train': 0, 'n_test': 0, 'n_external': 0,
                        'n_pos_train': 0, 'n_pos_test': 0, 'n_pos_external': 0,
                        'status': 'no_column'
                    })
                continue

            n_pos_train = int(y_train.sum())
            n_pos_test = int(y_test.sum())
            if n_pos_train < 3 or n_pos_test < 3:
                print(f"    Skipping: Insufficient positives (train={n_pos_train}, test={n_pos_test})")
                for fs in FEATURE_SETS:
                    results.append({
                        'diag_type': diag_type, 'time_point': time_point, 'feature_set': fs,
                        'auc': np.nan, 'external_auc': np.nan,
                        'n_train': len(y_train), 'n_test': len(y_test), 'n_external': 0,
                        'n_pos_train': n_pos_train, 'n_pos_test': n_pos_test, 'n_pos_external': 0,
                        'status': 'insufficient_positives'
                    })
                continue

            print(f"    Train: {len(y_train)} (pos={n_pos_train}, {n_pos_train/len(y_train)*100:.2f}%)")
            print(f"    Test: {len(y_test)} (pos={n_pos_test}, {n_pos_test/len(y_test)*100:.2f}%)")

            top_features = {}
            for feature_set in FEATURE_SETS:
                try:
                    feature_cols = get_feature_set(
                        feature_set, olink_cols, blood_cols, demo_cols, top_features
                    )
                    X_train = train_filtered[feature_cols].copy()
                    X_valid = valid_filtered[feature_cols].copy()
                    X_test = test_filtered[feature_cols].copy()

                    for c in CATEGORICAL_DEMO_COLS:
                        if c in X_train.columns:
                            X_train[c] = X_train[c].astype('category')
                            X_valid[c] = X_valid[c].astype('category')
                            X_test[c] = X_test[c].astype('category')

                    numeric_cols = [
                        c for c in feature_cols
                        if (c in olink_cols) or (c in blood_cols)
                    ]
                    if numeric_cols:
                        imputer = SimpleImputer(strategy='median')
                        X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
                        X_valid[numeric_cols] = imputer.transform(X_valid[numeric_cols])
                        X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])

                    auc, external_auc, model = train_and_evaluate(
                        X_train, X_test, y_train, y_test,
                        X_external=None, y_external=None
                    )

                    # model.save_model(f"{models_dir}xgb_model_{diag_type}_{time_point}_{feature_set}.json")

                    if feature_set == 'demo_protein_blood':
                        for eid, p in zip(train_filtered['eid'], model.predict_proba(X_train)[:, 1]):
                            risk_store['train'].setdefault(eid, {})[time_point] = float(p)
                        for eid, p in zip(valid_filtered['eid'], model.predict_proba(X_valid)[:, 1]):
                            risk_store['valid'].setdefault(eid, {})[time_point] = float(p)
                        for eid, p in zip(test_filtered['eid'], model.predict_proba(X_test)[:, 1]):
                            risk_store['test'].setdefault(eid, {})[time_point] = float(p)

                    if feature_set in BASE_FEATURE_SETS:
                        if feature_set == 'demo_protein':
                            top_features['demo_protein'] = extract_top_features_from_model(
                                model, feature_cols, olink_cols, n_top=50
                            )
                        elif feature_set == 'demo_blood':
                            top_features['demo_blood'] = extract_top_features_from_model(
                                model, feature_cols, blood_cols, n_top=50
                            )

                    print(f"    {feature_set}: AUC = {auc:.4f} ({len(feature_cols)} features)")
                    results.append({
                        'diag_type': diag_type, 'time_point': time_point, 'feature_set': feature_set,
                        'auc': auc, 'external_auc': np.nan,
                        'n_train': len(y_train), 'n_test': len(y_test), 'n_external': 0,
                        'n_pos_train': n_pos_train, 'n_pos_test': n_pos_test, 'n_pos_external': 0,
                        'n_features': len(feature_cols), 'status': 'completed'
                    })
                except Exception as e:
                    print(f"    {feature_set}: ERROR - {str(e)}")
                    results.append({
                        'diag_type': diag_type, 'time_point': time_point, 'feature_set': feature_set,
                        'auc': np.nan, 'external_auc': np.nan,
                        'n_train': len(y_train), 'n_test': len(y_test), 'n_external': 0,
                        'n_pos_train': n_pos_train, 'n_pos_test': n_pos_test, 'n_pos_external': 0,
                        'status': f'error: {str(e)}'
                    })

        risk_base_dir = os.path.join(data_path, "cancer_risk", "xgb_risk")
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

    return pd.DataFrame(results)


def save_results(results_df, output_dir='.'):
    results_df.to_csv(os.path.join(output_dir, 'xgb_benchmark_cancer_results.csv'), index=False)
    return results_df


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='XGBoost Benchmark for Cancer Prediction')
    parser.add_argument('--output-dir', '-o', type=str, default='.', help='Output directory for results')
    parser.add_argument('--data-path', '-d', type=str, default=DATA_PATH, help='Path to data directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 60)
    print("XGBoost Benchmark for Cancer Prediction")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Random seed: {RANDOM_STATE}")
    print(f"  Data path: {args.data_path}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Time points: {TIME_POINTS}")
    print(f"  Feature sets: {FEATURE_SETS}")
    print(f"  Sex-specific: breast/ovarian/uterine -> female; prostate -> male")

    cancer_sets = load_data_cancer(args.data_path)
    results_df = run_benchmark_cancer(
        cancer_sets,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )
    results_df = save_results(results_df, output_dir=args.output_dir)
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    return results_df


if __name__ == '__main__':
    main()
