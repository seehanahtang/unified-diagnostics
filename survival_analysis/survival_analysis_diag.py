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
data_path = "/orcd/pool/003/dbertsim_shared/ukb"
logger = setup_logger("logs")

DEMO_FEATURES = [
    'Age at recruitment', 
    'Sex_male',
    'Body mass index (BMI)', 
    'Systolic blood pressure, automated reading',
    'Diastolic blood pressure, automated reading',
    'Townsend deprivation index at recruitment',
    'Smoking status', 
    'Alcohol intake frequency.',
    # 'Medication for cholesterol, blood pressure or diabetes'
]
outcomes = ["ischemia", "stroke", "alzheimers", "copd", "lower_resp", "kidney", "t2d", "hhd", "lung", "colorectal", "stomach"]
CATEGORICAL_DEMO = ["Smoking status", "Alcohol intake frequency."]

# -----------------------
# Helpers
# -----------------------
def encode_demo(X_train, X_test):
    X_train = pd.get_dummies(X_train, columns=CATEGORICAL_DEMO, dummy_na=True)
    X_test  = pd.get_dummies(X_test,  columns=CATEGORICAL_DEMO, dummy_na=True)

    X_train = X_train.reindex(sorted(X_train.columns), axis=1)
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

    return X_train, X_test

def merge_tabtext(df):
    demo = pd.read_csv(f"{data_path}/demo_embeddings.csv")
    cn_columns = [c for c in demo.columns if (c.startswith('cn_'))]
    df = pd.merge(df, demo[['eid']+ cn_columns], how = 'left', on = 'eid')
    return df

def clean_future(df, outcome): 
    # Restrict to future
    df = df.loc[df[outcome] == 0].copy()
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
    score_dict = {colnames[i]: scores[i] for i in range(len(colnames))}


    return score_dict

def get_numeric_feature_columns(
    df: pd.DataFrame,
    use_olink: bool = True,
    use_blood: bool = True,
    use_demo: bool = True,
    use_tabtext: bool = True,
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

def get_selected_features(df, outcome):
    X = df[get_numeric_feature_columns(df)]
    y = df[f"{outcome}_future"]
    
    scores = sis(X, y)
    scores_sorted = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    sis_cols = [item[0] for item in scores_sorted if item[1] > 0.03]
    logger.info(f"Selected {len(sis_cols)} with sis value > 0.03")
    
    return sis_cols

    
# -----------------------
# Load data
# -----------------------
logger.info("Loading train/test (combining train and valid)...")
df_train_base = pd.read_csv(f"{data_path}/ukb_diag_train.csv")
df_test_base  = pd.read_csv(f"{data_path}/ukb_diag_test.csv")

    
for outcome in outcomes:
    logger.info("=" * 80)
    logger.info(f"Starting outcome: {outcome}")
    
    df_train = df_train_base.copy()
    df_test  = df_test_base.copy()
    
    # Clean + merge embeddings
    df_train = clean_future(df_train, outcome)
    df_train = merge_tabtext(df_train)
    
    df_test = clean_future(df_test, outcome)
    df_test = merge_tabtext(df_test)
    
    sis_cols = get_selected_features(df_train, outcome)
    feature_cols = DEMO_FEATURES + sis_cols
    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]
    X_train, X_test = encode_demo(X_train, X_test)
    
    logger.info(f"Selected {len(feature_cols)} demo and sis features")
    
    y_train = Surv.from_dataframe(f"{outcome}_future", f"{outcome}_time_to_diagnosis", df_train)
    y_test  = Surv.from_dataframe(f"{outcome}_future", f"{outcome}_time_to_diagnosis", df_test)
    
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

    # Overall event rate + mean time among events
    overall = pd.concat([df_train, df_test], ignore_index=True)
    overall_rate = overall[f"{outcome}_future"].mean()
    mean_event_time = overall.loc[overall[f"{outcome}_future"] == 1, f"{outcome}_time_to_diagnosis"].mean()
    logger.info(f"[{outcome}] Overall event_rate={overall_rate:.4f} | mean_event_time={mean_event_time:.3f}")

    
    risk_cols = [c for c in df_train.columns if c.startswith(f"{outcome}_risk_") or c.startswith(f"{outcome}_delta_")]
    df_train_base = df_train_base.merge(df_train[["eid"] + risk_cols], on="eid", how="left")
    df_test_base  = df_test_base.merge(df_test[["eid"] + risk_cols],  on="eid", how="left")

    