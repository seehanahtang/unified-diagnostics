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
OUTCOME = "cancer"

outcomes = ["skin", "breast", "prostate", "lung", "colorectal", "bladder"]
DEMO_FEATURES = [
    'Age at recruitment', 
    'Sex_male',
    'Ethnic background',
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

def clean_future(df, outcome, exclude_other = True): 
    # Restrict to future
    df = df.loc[df[f"{outcome}"] == 0].copy()
    df.loc[:, f"{outcome}_future"] = 0
    df.loc[df[f"{outcome}_time_to_diagnosis"] > 0, f"{outcome}_future"] = 1
    
    # Filter out patients who had cancer that is not in outcomes
    # Check if patient has any cancer not in our studied outcomes
    cancer_cols = [f"{outc}_time_to_diagnosis" for outc in outcomes]
    if exclude_other:
        has_studied_cancer = (df[cancer_cols] > 0).any(axis=1)
        no_cancer = (df[f"{outcome}_time_to_diagnosis"].isna())
        has_other_cancer = ~(has_studied_cancer | no_cancer)
        before = len(df)
        df = df.loc[~has_other_cancer].copy()
        after = len(df)
        logger.info(f"[{outcome}] Filtered out patients with cancer not in outcomes: {before-after} (kept {after})")
    
    for outc in outcomes:
        df.loc[:, f"{outc}_future"] = 0
        df.loc[df[f"{outc}_time_to_diagnosis"] > 0, f"{outc}_future"] = 1
    
    mask = df[f"{outcome}_future"] == 0
    df.loc[mask, f'{outcome}_time_to_diagnosis'] = df.loc[mask, 'time_to_follow_up']
    
    protein_cols = [c for c in df.columns if (c.startswith('olink_'))]
    
    # drop those with more than 25% missing 
    miss_rate = df[protein_cols].isna().mean(axis=1)
    before = len(df)
    df = df.loc[miss_rate <= 0.25].copy().reset_index(drop=True)
    after = len(df)
    logger.info(f"[{outcome}] Dropped for >25% missing Olink: {before-after} (kept {after})")
    
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
    use_tabtext: bool = True,
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
# logger.info("Loading train/test (combining train + valid into train)...")
# df_train_base = pd.read_csv(f"{data_path}/ukb_cancer_train_with_skin.csv")
# df_valid_base = pd.read_csv(f"{data_path}/ukb_cancer_valid_with_skin.csv")
# df_train_base = pd.concat([df_train_base, df_valid_base], axis=0)
# df_test_base  = pd.read_csv(f"{data_path}/ukb_cancer_test_with_skin.csv")
train_df = pd.read_csv(f'{config.data_path}ukb_cancer_train_new.csv')
test_df = pd.read_csv(f'{config.data_path}ukb_cancer_test_new.csv')
    
logger.info("=" * 80)
logger.info(f"Starting outcome: {OUTCOME}")

df_train = df_train_base.copy()
df_test  = df_test_base.copy()

# Clean + merge embeddings
df_train = clean_future(df_train, OUTCOME)
df_train = merge_tabtext(df_train)

df_test = clean_future(df_test, OUTCOME)
df_test = merge_tabtext(df_test)

sis_cols = get_selected_features(df_train, OUTCOME)
feature_cols = DEMO_FEATURES + sis_cols
X_train = df_train[feature_cols]
X_test = df_test[feature_cols]
X_train, X_test = encode_demo(X_train, X_test)

y_train = Surv.from_dataframe(f"{OUTCOME}_future", f"{OUTCOME}_time_to_diagnosis", df_train)
y_test  = Surv.from_dataframe(f"{OUTCOME}_future", f"{OUTCOME}_time_to_diagnosis", df_test)



# -----------------------
# Optuna Hyperparameter Tuning for RSF
# -----------------------
import optuna

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7, None]),
        "n_jobs": -1,
        "random_state": 42
    }
    model = RandomSurvivalForest(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    c_index = concordance_index_censored(y_test[f"{OUTCOME}_future"], y_test[f"{OUTCOME}_time_to_diagnosis"], y_pred)[0]
    return c_index

logger.info("Starting Optuna hyperparameter tuning for RSF...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
logger.info(f"Optuna best params: {study.best_params}")

# Train final RSF model with best params
best_params = study.best_params
best_params["n_jobs"] = -1
best_params["random_state"] = 42
rsf = RandomSurvivalForest(**best_params)
t0 = time.time()
rsf.fit(X_train, y_train)
logger.info(f"[{OUTCOME}] Fit RSF (Optuna best) done in {(time.time()-t0)/60:.1f}min")

dump(rsf, f"models/rsf_{OUTCOME}_sis_only_optuna.joblib")  
logger.info(f"[{OUTCOME}] Saved model as models/rsf_{OUTCOME}_sis_only_optuna.joblib")

t1 = time.time()
y_pred = rsf.predict(X_test)
logger.info(f"[{OUTCOME}] Predicted in {(time.time()-t1):.1f}s")

c_index = concordance_index_censored(y_test[f"{OUTCOME}_future"], y_test[f"{OUTCOME}_time_to_diagnosis"], y_pred)
logger.info(f"[{OUTCOME}] C-index = {c_index[0]:.4f}")

# Overall event rate + mean time among events
overall = pd.concat([df_train, df_test], ignore_index=True)
overall_rate = overall[f"{OUTCOME}_future"].mean()
mean_event_time = overall.loc[overall[f"{OUTCOME}_future"] == 1, f"{OUTCOME}_time_to_diagnosis"].mean()
logger.info(f"[{OUTCOME}] Overall event_rate={overall_rate:.4f} | mean_event_time={mean_event_time:.3f}")
