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
def setup_logger(log_dir: str, name: str = "ukb_rsf_cancer_base") -> logging.Logger:
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
data_path = "/orcd/pool/003/dbertsim_shared/ukb"
logger = setup_logger("logs")

DEMO_FEATURES = [
    'Age at recruitment', 
    'Sex_male',
    # 'Ethnic background',
    'Body mass index (BMI)', 
    'Systolic blood pressure, automated reading',
    'Diastolic blood pressure, automated reading',
    'Townsend deprivation index at recruitment',
    'Smoking status', 
    'Alcohol intake frequency.',
    # 'Medication for cholesterol, blood pressure or diabetes'
]
CATEGORICAL_DEMO = ["Ethnic background", "Smoking status", "Alcohol intake frequency."]

# -----------------------
# Helpers
# -----------------------
def get_cancer_types(df):
    """Extract cancer types from column names."""
    cancers = [c for c in df.columns if c.endswith('cancer')]
    return cancers

def encode_demo(X_train, X_test):
    X_train = pd.get_dummies(X_train, columns=CATEGORICAL_DEMO, dummy_na=True)
    X_test = pd.get_dummies(X_test, columns=CATEGORICAL_DEMO, dummy_na=True)
    
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    return X_train, X_test

def merge_tabtext(df):
    demo = pd.read_csv(f"{data_path}/demo_embeddings.csv")
    cn_columns = [c for c in demo.columns if (c.startswith('cn_'))]
    df = pd.merge(df, demo[['eid']+ cn_columns], how = 'left', on = 'eid')
    return df

def clean_future(df, outcome): 
    # Restrict to future
    df = df.loc[df[f"{outcome}"] == 0].copy()
    df.loc[:, f"{outcome}_future"] = 0
    df.loc[df[f"{outcome}_time_to_diagnosis"] > 0, f"{outcome}_future"] = 1
    
    mask = df[f"{outcome}_future"] == 0
    df.loc[mask, f'{outcome}_time_to_diagnosis'] = df.loc[mask, 'time_to_follow_up']
    
    imputer = SimpleImputer(strategy="median")
    df[protein_cols] = imputer.fit_transform(df[protein_cols])
    logger.info(f"[{outcome}] Imputed Olink median on {len(protein_cols)} proteins")
    
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
    ranked_idx = np.argsort(-scores)[:k]
    score_dict = {colnames[i]: scores[i] for i in range(len(colnames))}

    return score_dict

def get_numeric_feature_columns(
    df: pd.DataFrame,
    use_olink: bool = True,
    use_blood: bool = True,
    use_demo: bool = True,
    use_tabtext: bool = False,
    cancer_type: Optional[str] = None
) -> List[str]:
    features = []
    
    if use_tabtext:
        embedding_cols = [col for col in df.columns if col.startswith("cn_")]
        features.extend(embedding_cols)
    
    if use_olink:
        olink_cols = [col for col in df.columns if col.startswith("olink_")]
        features.extend(olink_cols)
    
    if use_blood:
        blood_cols = [col for col in df.columns if col.startswith("blood_")]
        features.extend(blood_cols)
    
    return features

def get_feature_columns(df):
    """Extract feature column lists from dataframe."""
    olink_cols = [c for c in df.columns if c.startswith('olink_')]
    blood_cols = [c for c in df.columns if c.startswith('blood_')]
    demo_cols = [c for c in DEMO_COLS if c in df.columns]
    return olink_cols, blood_cols, demo_cols

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
    
# -----------------------
# Load data
# -----------------------
logger.info("Loading train/test...")
df_train_base = pd.read_csv(f"{data_path}ukb_cancer_train.csv", low_memory=False)
df_test_base  = pd.read_csv(f"{data_path}ukb_cancer_test.csv", low_memory=False)

outcomes = get_cancer_types(df_train_base)
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
results_df.to_csv(f'{output_dir}rsf_cancer_results.csv', index=False)
logger.info("Done")