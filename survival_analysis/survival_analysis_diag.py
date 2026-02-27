import os
import sys
import time
import logging
from typing import List, Optional

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from joblib import dump, load

from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

# -----------------------
# Logging setup
# -----------------------
def setup_logger(log_dir: str, name: str = "ukb_rsf_diag") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # avoid duplicate logs in notebooks

    # Clear old handlers if re-running in notebook
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # File
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# -----------------------
# Config
# -----------------------
data_path = "/orcd/pool/003/dbertsim_shared/ukb/"
output_dir = "results/"
logger = setup_logger("logs")

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

CATEGORICAL_DEMO = ["Smoking status", "Alcohol intake frequency."]

# Feature set configurations
FEATURE_SETS = [
    'demo_protein',           # Demographics + Protein features
    'demo_blood',             # Demographics + Blood features  
    'demo_protein_blood',     # Demographics + Protein + Blood (all features)
]

# -----------------------
# Helpers
# -----------------------
def get_diagnosis_types(df):
    """Extract diagnosis types from column names."""
    time_cols = [c for c in df.columns if c.endswith('_time_to_diagnosis')]
    diag_types = [c.replace('_time_to_diagnosis', '') for c in time_cols]
    return diag_types

def encode_demo(X_train, X_test):
    X_train = pd.get_dummies(X_train, columns=CATEGORICAL_DEMO, dummy_na=True)
    X_test  = pd.get_dummies(X_test,  columns=CATEGORICAL_DEMO, dummy_na=True)

    X_train = X_train.reindex(sorted(X_train.columns), axis=1)
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

    return X_train, X_test

def clean_future(df, outcome): 
    # Restrict to future
    df = df.loc[df[outcome] == 0].copy()
    df.loc[:, f"{outcome}_future"] = 0
    df.loc[df[f"{outcome}_time_to_diagnosis"] > 0, f"{outcome}_future"] = 1
    
    mask = df[f"{outcome}_future"] == 0
    df.loc[mask, f'{outcome}_time_to_diagnosis'] = 15.0
    
    return df 

def sis(X, y, k=None, frac=None):
    is_df = isinstance(X, pd.DataFrame)
    colnames = X.columns if is_df else None
    
    # convert to array
    X = np.asarray(X)
    y = np.asarray(y).astype(float)

    n, p = X.shape

    # determine number of features to keep
    if k is None:
        if frac is None:
            k = p
        else:
            k = max(1, int(np.floor(frac * p)))

    # center
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()

    # correlation numerator
    cov = np.sum(Xc * yc[:, None], axis=0)

    # std dev
    X_std = np.sqrt(np.sum(Xc**2, axis=0))
    y_std = np.sqrt(np.sum(yc**2))

    # SIS scores
    scores = np.abs(cov / (X_std * y_std + 1e-12))

    # sort features
    score_dict = {colnames[i]: scores[i] for i in range(len(colnames))}


    return score_dict

def get_feature_set(df, outcome, feature_set_name, olink_cols, blood_cols, demo_cols):
    if feature_set_name == 'demo_blood':
        return demo_cols + blood_cols
    
    sis_cols = get_selected_features(df_train, outcome, olink_cols)
    if feature_set_name == 'demo_protein':
        return demo_cols + sis_cols
    
    elif feature_set_name == 'demo_protein_blood':
        return demo_cols + sis_cols + blood_cols
    
    else:
        raise ValueError(f"Unknown feature set: {feature_set_name}")

def get_feature_columns(df):
    """Extract feature column lists from dataframe."""
    olink_cols = [c for c in df.columns if c.startswith('olink_')]
    blood_cols = [c for c in df.columns if c.startswith('blood_')]
    demo_cols = [c for c in DEMO_COLS if c in df.columns]
    return olink_cols, blood_cols, demo_cols

def get_selected_features(df, outcome, features):
    X = df[features]
    y = df[f"{outcome}_future"]
    
    scores = sis(X, y)
    scores_sorted = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    sis_cols = [item[0] for item in scores_sorted[:500]]
    logger.info(f"Selected 50 columns with largest sis scores")
    
    return sis_cols

def train_and_evaluate(X_train, X_test, y_train, y_test):
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    logger.info(f"[{outcome}] Imputed median on all features.")

    # survial random forest 
    logger.info("Training random forest...")
    rsf = RandomSurvivalForest(
        n_estimators=100,
        min_samples_split=10,
        min_samples_leaf=100,
        max_depth = 10,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )
    
    t0 = time.time()
    rsf.fit(X_train, y_train)
    logger.info(f"[{outcome}] Fit RSF done in {(time.time()-t0)/60:.1f}min")
    
    t1 = time.time()
    y_pred = rsf.predict(X_test)
    logger.info(f"[{outcome}] Predicted in {(time.time()-t1):.1f}s")
    
    c_index = concordance_index_censored(y_test[f"{outcome}_future"], y_test[f"{outcome}_time_to_diagnosis"], y_pred)
    logger.info(f"[{outcome}] C-index = {c_index[0]:.4f}")

    return rsf,c_index[0]

# -----------------------
# Load data
# -----------------------
logger.info("Loading train/test...")
df_train_base = pd.read_csv(f"{data_path}ukb_disease_train.csv", low_memory=False)
df_test_base  = pd.read_csv(f"{data_path}ukb_disease_test.csv", low_memory=False)

outcomes = get_diagnosis_types(df_train_base)
olink_cols, blood_cols, demo_cols = get_feature_columns(df_train_base)

results = []
for outcome in outcomes:
    for fs in FEATURE_SETS:
        logger.info("=" * 80)
        logger.info(f"Starting outcome: {outcome}")
        
        df_train = df_train_base.copy()
        df_test  = df_test_base.copy()
        
        df_train = clean_future(df_train, outcome)
        df_test = clean_future(df_test, outcome)
        
        feature_cols = get_feature_set(
            df_train, outcome, fs, olink_cols, blood_cols, demo_cols
        )
        X_train = df_train[feature_cols]
        X_test = df_test[feature_cols]
        X_train, X_test = encode_demo(X_train, X_test)
        
        logger.info(f"Selected {len(feature_cols)} {fs} features")
        
        y_train = Surv.from_dataframe(f"{outcome}_future", f"{outcome}_time_to_diagnosis", df_train)
        y_test  = Surv.from_dataframe(f"{outcome}_future", f"{outcome}_time_to_diagnosis", df_test)
        
        model, c_index = train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # Overall event rate + mean time among events
        overall = pd.concat([df_train, df_test], ignore_index=True)
        overall_rate = overall[f"{outcome}_future"].mean()
        mean_event_time = overall.loc[overall[f"{outcome}_future"] == 1, f"{outcome}_time_to_diagnosis"].mean()
        logger.info(f"[{outcome}] Overall event_rate={overall_rate:.4f} | mean_event_time={mean_event_time:.3f}")
        
        results.append({
            'disease': outcome,
            'feature_set': fs,
            'c_index': c_index,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'event_rate': round(overall_rate,4),
            'n_features': len(feature_cols),
        })
        
results_df = pd.DataFrame(results)
results_df.to_csv(f'{output_dir}rsf_disease_results.csv', index=False)
logger.info("Done")