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
def setup_logger(log_dir: str, name: str = "ukb_rsf_cancer_risk_only") -> logging.Logger:
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
    'Ethnic background',
    'Body mass index (BMI)', 
    'Systolic blood pressure, automated reading',
    'Diastolic blood pressure, automated reading',
    'Townsend deprivation index at recruitment',
    'Smoking status', 
    'Alcohol intake frequency.',
    # 'Medication for cholesterol, blood pressure or diabetes'
]
outcomes = ["skin", "breast", "prostate", "lung", "colorectal", "bladder"]
CATEGORICAL_DEMO = ["Ethnic background", "Smoking status", "Alcohol intake frequency."]

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
    if outcome == "cancer":
        df = df.loc[df["cancer"] == 0].copy()
    else:
        df = df.loc[df[f"{outcome}_cancer"] == 0].copy()
    df.loc[:, f"{outcome}_future"] = 0
    df.loc[df[f"{outcome}_time_to_diagnosis"] > 0, f"{outcome}_future"] = 1
    
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

def get_cancer_model_features(df, X, outcome, rsf):
        
    # Get survival functions for all samples
    surv_funcs = rsf.predict_survival_function(X)

    model_times = surv_funcs[0].x  # Time points from the model
    max_time = model_times.max()

    # Create time points at 0.5-year intervals
    time_points = np.arange(0.5, max_time + 0.5, 0.5)
    time_points = time_points[time_points <= max_time]

    logger.info(f"[{outcome}] Generating features for {len(time_points)} time points (0.5 to {time_points[-1]:.1f} years)")

    # Initialize arrays for risk values
    n_samples = len(surv_funcs)
    risk_matrix = np.zeros((n_samples, len(time_points)))

    # Extract survival probabilities at each time point
    for i, sf in enumerate(surv_funcs):
        for j, t in enumerate(time_points):
            # Interpolate survival probability at time t
            surv_prob = sf(t)
            risk_matrix[i, j] = 1 - surv_prob  # Risk = 1 - survival

    # Add risk columns
    for j, t in enumerate(time_points):
        col_name = f"{outcome}_risk_{t:.1f}y"
        df[col_name] = risk_matrix[:, j]

    # Add delta columns (change in risk from previous time point)
    prev_risk = np.zeros(n_samples)
    for j, t in enumerate(time_points):
        delta = risk_matrix[:, j] - prev_risk
        col_name = f"{outcome}_delta_{t:.1f}y"
        df[col_name] = delta
        prev_risk = risk_matrix[:, j].copy()
            
    logger.info(f"[{outcome}] Added {len(time_points) * 2} columns (risk + delta)")
    
    return df

    
# -----------------------
# Load data
# -----------------------
# logger.info("Loading train/test (combining train and valid)...")
# df_train_base = pd.read_csv(f"{data_path}/ukb_cancer_train_with_skin.csv")
# df_valid_base = pd.read_csv(f"{data_path}/ukb_cancer_valid_with_skin.csv")
# df_train_base = pd.concat([df_train_base, df_valid_base], axis=0)
# df_test_base  = pd.read_csv(f"{data_path}/ukb_cancer_test_with_skin.csv")

    
# for outcome in outcomes:
#     logger.info("=" * 80)
#     logger.info(f"Starting outcome: {outcome}")
    
#     df_train = df_train_base.copy()
#     df_test  = df_test_base.copy()
    
#     # Clean + merge embeddings
#     df_train = clean_future(df_train, outcome)
#     df_train = merge_tabtext(df_train)
    
#     df_test = clean_future(df_test, outcome)
#     df_test = merge_tabtext(df_test)
    
#     sis_cols = get_selected_features(df_train, outcome)
#     feature_cols = DEMO_FEATURES + sis_cols
#     X_train = df_train[feature_cols]
#     X_test = df_test[feature_cols]
#     X_train, X_test = encode_demo(X_train, X_test)
    
#     logger.info(f"Selected {len(feature_cols)} demo and sis features")
    
#     y_train = Surv.from_dataframe(f"{outcome}_future", f"{outcome}_time_to_diagnosis", df_train)
#     y_test  = Surv.from_dataframe(f"{outcome}_future", f"{outcome}_time_to_diagnosis", df_test)

    
#     # survial random forest 
#     logger.info("Training random forest...")
#     rsf = RandomSurvivalForest(
#         n_estimators=100,
#         min_samples_split=10,
#         min_samples_leaf=100,
#         max_depth = 10,
#         max_features="sqrt",
#         n_jobs=-1,
#         random_state=42
#     )
    
#     t0 = time.time()
#     rsf.fit(X_train, y_train)
#     logger.info(f"[{outcome}] Fit RSF done in {(time.time()-t0)/60:.1f}min")

#     # dump(rsf, f"models/rsf_{outcome}_sis.joblib")  
#     # logger.info(f"[{outcome}] Saved model as models/rsf_{outcome}_sis.joblib")
    
#     t1 = time.time()
#     y_pred = rsf.predict(X_test)
#     logger.info(f"[{outcome}] Predicted in {(time.time()-t1):.1f}s")
    
#     c_index = concordance_index_censored(y_test[f"{outcome}_future"], y_test[f"{outcome}_time_to_diagnosis"], y_pred)
#     logger.info(f"[{outcome}] C-index = {c_index[0]:.4f}")

#     # Overall event rate + mean time among events
#     overall = pd.concat([df_train, df_test], ignore_index=True)
#     overall_rate = overall[f"{outcome}_future"].mean()
#     mean_event_time = overall.loc[overall[f"{outcome}_future"] == 1, f"{outcome}_time_to_diagnosis"].mean()
#     logger.info(f"[{outcome}] Overall event_rate={overall_rate:.4f} | mean_event_time={mean_event_time:.3f}")
    
#     # logger.info(f"[{outcome}] Writing train and test datasets to data/data/cancer_train_with_sis_selected_features_{outcome}.csv")
#     # df_train.to_csv(f"data/cancer_train_with_sis_selected_features_{outcome}.csv", index=False)
#     # df_test.to_csv(f"data/cancer_test_with_sis_selected_features_{outcome}.csv", index=False)
    
#     # Add risk columns
#     df_train = get_cancer_model_features(df_train, X_train, outcome, rsf)
#     df_test  = get_cancer_model_features(df_test,  X_test,  outcome, rsf)
    
#     risk_cols = [c for c in df_train.columns if c.startswith(f"{outcome}_risk_") or c.startswith(f"{outcome}_delta_")]
#     df_train_base = df_train_base.merge(df_train[["eid"] + risk_cols], on="eid", how="left")
#     df_test_base  = df_test_base.merge(df_test[["eid"] + risk_cols],  on="eid", how="left")

    
# logger.info("=" * 80)
# outcome = "cancer"
# logger.info(f"Starting outcome: {outcome}")

# df_train = df_train_base.copy()
# df_test  = df_test_base.copy()

# # Clean + merge embeddings
# df_train = clean_future(df_train, outcome)
# df_train = merge_tabtext(df_train)

# df_test = clean_future(df_test, outcome)
# df_test = merge_tabtext(df_test)

# sis_cols = get_selected_features(df_train, outcome)
# risk_cols = [
#     col
#     for outcome in outcomes
#     for col in df_train.columns
#     if col.startswith(f"{outcome}_risk_") or col.startswith(f"{outcome}_delta_")
# ]
# feature_cols = DEMO_FEATURES + sis_cols + risk_cols
# X_train = df_train[feature_cols]
# X_test = df_test[feature_cols]
# X_train, X_test = encode_demo(X_train, X_test)

# y_train = Surv.from_dataframe(f"{outcome}_future", f"{outcome}_time_to_diagnosis", df_train)
# y_test  = Surv.from_dataframe(f"{outcome}_future", f"{outcome}_time_to_diagnosis", df_test)


# # survial random forest 
# logger.info("Training random forest...")
# rsf = RandomSurvivalForest(
#     n_estimators=100,
#     min_samples_split=10,
#     min_samples_leaf=100,
#     max_depth = 10,
#     max_features="sqrt",
#     n_jobs=-1,
#     random_state=42
# )

# t0 = time.time()
# rsf.fit(X_train, y_train)
# logger.info(f"[{outcome}] Fit RSF done in {(time.time()-t0)/60:.1f}min")

# # dump(rsf, f"models/rsf_{outcome}_sis.joblib")  
# # logger.info(f"[{outcome}] Saved model as models/rsf_{outcome}_sis.joblib")

# t1 = time.time()
# y_pred = rsf.predict(X_test)
# logger.info(f"[{outcome}] Predicted in {(time.time()-t1):.1f}s")

# c_index = concordance_index_censored(y_test[f"{outcome}_future"], y_test[f"{outcome}_time_to_diagnosis"], y_pred)
# logger.info(f"[{outcome}] C-index = {c_index[0]:.4f}")


# logger.info("Writing cancer train and test datasets to data/data/cancer_train_with_sis_selected_features.csv")
# df_train.to_csv(f"data/cancer_train_with_sis_selected_features.csv", index=False)
# df_test.to_csv(f"data/cancer_test_with_sis_selected_features.csv", index=False)

outcome = "cancer"

df_train = pd.read_csv("data/cancer_train_with_sis_selected_features.csv")
df_test = pd.read_csv("data/cancer_test_with_sis_selected_features.csv")

logger.info("Training with only the risk features...")
risk_cols = [
    col
    for outcome in outcomes
    for col in df_train.columns
    if col.startswith(f"{outcome}_risk_") or col.startswith(f"{outcome}_delta_")
]
X_train = df_train[risk_cols]
X_test = df_test[risk_cols]

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

dump(rsf, f"models/rsf_{outcome}_sis.joblib")  
logger.info(f"[{outcome}] Saved model as models/rsf_{outcome}_sis.joblib")

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
