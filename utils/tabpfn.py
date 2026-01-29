import os
import sys
import gc
import time
import logging
import argparse

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

from tabpfn import TabPFNClassifier, save_fitted_tabpfn_model, load_fitted_tabpfn_model


# -----------------------------
# Argument Parser
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("OUTCOME", type=str, help="Cancer outcome prefix (e.g., breast, lung)")
parser.add_argument("--impute", action="store_true", help="Apply median imputation + scaling")
parser.add_argument("--protein-only", action="store_true", help="Only use protein features")
args = parser.parse_args()

OUTCOME = args.OUTCOME
DO_IMPUTE = args.impute
PROTEIN_ONLY = args.protein_only

models_dir = "output"
os.makedirs(models_dir, exist_ok=True)

# -----------------------------
# Logging setup
# -----------------------------
log_path = os.path.join(models_dir, f"tabpfn_{OUTCOME}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

categorical_cols = ['Ethnic background', 'Smoking status', 'Medication for cholesterol, blood pressure or diabetes']
demographic_cols = ['Age at recruitment', 'Sex_male', 'Ethnic background', \
       'Body mass index (BMI)', 'Systolic blood pressure, automated reading', \
       'Diastolic blood pressure, automated reading', \
       'Townsend deprivation index at recruitment', 'Smoking status', 'Alcohol intake frequency.', \
        'Medication for cholesterol, blood pressure or diabetes'] 

# -----------------------------
# Data loading & preprocessing
# -----------------------------
def preprocess_categorical(df):
    for col in categorical_cols:
        df[col] = df[col].astype("category")
    return df

def get_data(outcome: str):
    logger.info("Loading train/valid/test CSV files...")

    train_df = preprocess_categorical(pd.read_csv('/orcd/pool/003/dbertsim_shared/ukb/ukb_cancer_train.csv'))
    val_df   = preprocess_categorical(pd.read_csv('/orcd/pool/003/dbertsim_shared/ukb/ukb_cancer_valid.csv'))
    test_df  = preprocess_categorical(pd.read_csv('/orcd/pool/003/dbertsim_shared/ukb/ukb_cancer_test.csv'))

    outcome_col = f"{outcome}_cancer"

    df_all = pd.concat([train_df, val_df, test_df])
    olink_cols = [c for c in df_all.columns if "olink" in c]
    missing_pct = df_all[olink_cols].isnull().mean()
    
    if PROTEIN_ONLY:
        selected_cols = missing_pct.sort_values().index[:2000]
    else:
        selected_cols = missing_pct.sort_values().index[:1990]
        selected_cols = demographic_cols + selected_cols
        

    logger.info(f"Selected {len(selected_cols)} features with lowest missingness.")

    def split_xy(df_):
        return df_[selected_cols], df_[outcome_col].values

    X_train, y_train = split_xy(train_df)
    X_valid, y_valid = split_xy(val_df)
    X_test,  y_test  = split_xy(test_df)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, selected_cols


def impute_and_scale(X_train, X_valid, X_test):
    logger.info("Applying median imputation + scaling...")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_valid_imp = imputer.transform(X_valid)
    X_test_imp  = imputer.transform(X_test)

    X_train_s = scaler.fit_transform(X_train_imp)
    X_valid_s = scaler.transform(X_valid_imp)
    X_test_s  = scaler.transform(X_test_imp)

    return X_train_s, X_valid_s, X_test_s


# -----------------------------
# Main execution
# -----------------------------
def main():
    
    start_time = time.perf_counter()
    logger.info(f"Starting TabPFN for OUTCOME={OUTCOME}, impute={DO_IMPUTE}")

    X_train, y_train, X_valid, y_valid, X_test, y_test, feature_names = get_data(OUTCOME)

    # ------------------------
    # Optional Imputation Step
    # ------------------------
    if DO_IMPUTE:
        X_train_s, X_valid_s, X_test_s = impute_and_scale(X_train, X_valid, X_test)
    else:
        logger.info("Skipping imputation — passing NaNs directly to TabPFN.")
        X_train_s = X_train.values
        X_valid_s = X_valid.values
        X_test_s  = X_test.values

    # Combine train+valid
    X_big = np.vstack([X_train_s, X_valid_s])
    y_big = np.concatenate([y_train, y_valid])

    # ------------------------
    # TabPFN
    # ------------------------
    if not torch.cuda.is_available():
        logger.warning("device='cuda' requested but no CUDA device is available. Falling back to CPU.")
        DEVICE = "cpu"
    else: 
        DEVICE = "cuda"

    logger.info(f"Initializing TabPFNClassifier on device={DEVICE}...")
    clf = TabPFNClassifier(
        device=DEVICE,
    )

    logger.info("Fitting TabPFN...")
    clf.fit(X_big, y_big)

    # Predict Test AUC
    logger.info("Predicting on test set...")
    probs = clf.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, probs)
    logger.info(f"Test AUC = {auc:.6f}")
    
    # ------------------------
    # Save GPU model
    # ------------------------
    model_path = os.path.join(models_dir, f"tabpfn_{OUTCOME}.tabpfn_fit")
    logger.info(f"Saving fitted GPU TabPFN model → {model_path}")
    save_fitted_tabpfn_model(clf, model_path)

    # ------------------------
    # Free GPU memory
    # ------------------------
    logger.info("Clearing GPU memory before permutation importance...")
    # Drop big training objects if they exist in scope
    for var_name in ["clf", "X_big", "X_big_sub", "y_big", "y_big_sub"]:
        if var_name in globals():
            del globals()[var_name]

    torch.cuda.empty_cache()
    gc.collect()
    
    # ------------------------
    # Reload model on CPU
    # ------------------------
    logger.info("Reloading TabPFN model on GPU for permutation importance...")
    # Load fitted model directly on GPU
    clf = load_fitted_tabpfn_model(model_path, device="cpu")

    # (Optional) make sure test array is writable to avoid warnings
    X_test_perm = np.array(X_test_s, copy=True)
    y_test_perm = np.array(y_test, copy=True)

    # (Optional but recommended) subsample for stability / memory
    max_perm_n = 1000
    n_test = X_test_perm.shape[0]
    if n_test > max_perm_n:
        logger.info(f"Subsampling test set for permutation importance: {n_test} → {max_perm_n}")
        rng = np.random.RandomState(42)
        idx_perm = rng.choice(n_test, size=max_perm_n, replace=False)
        X_test_perm = X_test_perm[idx_perm]
        y_test_perm = y_test_perm[idx_perm]

    # --- FEATURE IMPORTANCE (permutation) on GPU-backed model ---
    logger.info("Computing permutation-based feature importance on GPU...")
    perm_start = time.perf_counter()
    perm = permutation_importance(
        clf,
        X_test_perm,
        y_test_perm,
        n_repeats=3,
        random_state=42,
        n_jobs=1,   
    )
    perm_end = time.perf_counter()
    logger.info(f"Permutation importance completed in {(perm_end - perm_start) / 60:.3f} minutes.")

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)

    fi_path = os.path.join(models_dir, f"tabpfn_{OUTCOME}_feature_importances.csv")
    logger.info(f"Saving feature importances → {fi_path}")
    fi_df.to_csv(fi_path, index=False)

    total_minutes = (time.perf_counter() - start_time) / 60
    logger.info(f"Done. Total runtime = {total_minutes:.2f} minutes.")


if __name__ == "__main__":
    """
    Wrap main in a try/except so that even if something fails
    before or during logging, we dump a fatal log.
    """
    # try:
    main()
    # except Exception as e:
    #     logger.info(f"Exception: {e}.")
    
