import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from joblib import Parallel, delayed

import xgboost as xgb
import shap

OUTPUT_DIR = "data/protein/"
OUTCOME = sys.argv[1]


def get_data(outcome: str):

    train_df = pd.read_csv('/orcd/pool/003/dbertsim_shared/ukb/ukb_cancer_train.csv')
    val_df   = pd.read_csv('/orcd/pool/003/dbertsim_shared/ukb/ukb_cancer_valid.csv')
    test_df  = pd.read_csv('/orcd/pool/003/dbertsim_shared/ukb/ukb_cancer_test.csv')

    outcome_col = f"{outcome}_cancer"

    df_all = pd.concat([train_df, val_df, test_df])
    olink_cols = [c for c in df_all.columns if "olink" in c]

    missing_pct = df_all[olink_cols].isnull().mean()
    selected_cols = missing_pct[missing_pct < 0.4].index # can tune this proportion

    print(f"Selected {len(selected_cols)} features with < 40% missingness.")

    def split_xy(df_):
        return df_[selected_cols], df_[outcome_col].values

    X_train, y_train = split_xy(train_df)
    X_valid, y_valid = split_xy(val_df)
    X_test,  y_test  = split_xy(test_df)
    
    X_trainvalid = pd.concat([X_train, X_valid], axis=0)
    y_trainvalid = np.concatenate((y_train, y_valid))
    

    return X_trainvalid, y_trainvalid, X_test, y_test

def stability_selection_elastic_net(
    X_df,
    y,
    n_bootstrap=100,
    sample_frac=0.75,
    C=0.1,
    l1_ratio=0.5,
    random_state=42,
    n_jobs=-1,
):
    """
    Stability selection using logistic regression with elastic net penalty.
    X_df: pandas DataFrame of features (columns = protein IDs/names)
    y:    1D array-like of labels
    """
    rng = np.random.RandomState(random_state)
    X = X_df.values  # use numpy internally
    feature_names = X_df.columns.to_list()

    n_samples, n_features = X.shape

    selection_counts = np.zeros(n_features, dtype=int)
    coeff_sum = np.zeros(n_features, dtype=float)
    
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X_df.values)
    X_scaled = scaler.fit_transform(X_imputed)

    # Pipeline: standardize + elastic net logistic regression
    base_logreg = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=l1_ratio,
        C=C,
        max_iter=1000,      # keep this modest; can increase if needed
        tol=1e-3,           # slightly looser tolerance for speed
        random_state=random_state,
        n_jobs=1            # IMPORTANT: avoid nested parallelism
    )
    
    # ---- Pre-generate bootstrap indices ----
    n_sub = int(sample_frac * n_samples)
    bootstrap_indices = [
        rng.choice(n_samples, size=n_sub, replace=True)
        for _ in range(n_bootstrap)
    ]
    
    # ---- Function to run one bootstrap fit ----
    def _fit_one_bootstrap(idx):
        X_b = X_scaled[idx]
        y_b = y[idx]

        model = clone(base_logreg)
        model.fit(X_b, y_b)
        coef = model.coef_.ravel()  # shape: (n_features,)
        return coef
    
    # ---- Run bootstraps in parallel ----
    coef_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_fit_one_bootstrap)(idx)
        for idx in bootstrap_indices
    )
    coefs = np.vstack(coef_list)
    
    # ---- Aggregate stability stats ----
    nonzero_mask = (coefs != 0)
    selection_counts = nonzero_mask.sum(axis=0) # times feature selected
    coeff_sum = np.abs(coefs).sum(axis=0) # sum of abs coefs

    selection_freq = selection_counts / n_bootstrap
    avg_coeff = np.divide(
        coeff_sum,
        np.maximum(selection_counts, 1),  # avoid div-by-zero
        where=selection_counts > 0
    )

    stab_df = pd.DataFrame({
        "feature": feature_names,
        "selection_freq": selection_freq,
        "avg_abs_coef": avg_coeff
    }).sort_values("selection_freq", ascending=False)

    return stab_df.reset_index(drop=True)

# ----- CONFIG -----
X_train, y_train, X_test, y_test = get_data(OUTCOME)
print("Data loaded")

stab_df = stability_selection_elastic_net(
    X_train, y_train,
    n_bootstrap=30,   # can increase to 200+ if time allows
    sample_frac=0.5,
    C=0.1,
    l1_ratio=0.5,
    random_state=42
)

print("Elastic net completed")

# Basic XGBoost classifier (tune hyperparameters as you like)
xgb_clf = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

xgb_clf.fit(X_train, y_train)

print("XGBoost model completed")

# SHAP values
explainer = shap.TreeExplainer(xgb_clf)

# For speed, you can subsample rows for SHAP
shap_sample = X_train
if X_train.shape[0] > 5000:
    idx = np.random.choice(X_train.shape[0], 5000, replace=False)
    shap_sample = X_train.iloc[idx]

shap_values = explainer.shap_values(shap_sample)  # shape: (n_samples, n_features)

# Mean absolute SHAP value per feature
mean_abs_shap = np.abs(shap_values).mean(axis=0)

feature_names = X_train.columns.to_list()
shap_df = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

# Merge stability and SHAP
merged = stab_df.merge(shap_df, on="feature", how="inner")

# Normalize SHAP for combination
shap_max = merged["mean_abs_shap"].max()
if shap_max > 0:
    merged["shap_norm"] = merged["mean_abs_shap"] / shap_max
else:
    merged["shap_norm"] = 0.0

# Combined score (you can adjust this formula)
# Example: average of selection_freq and normalized SHAP
merged["combined_score"] = 0.5 * merged["selection_freq"] + 0.5 * merged["shap_norm"]

# Rank by combined score
merged = merged.sort_values("combined_score", ascending=False).reset_index(drop=True)

# Apply filters:
selection_freq_threshold = 0.5   # require stable selection in ≥ 50% of bootstraps
top_k_shap = 500                 # require being in top 200 by SHAP (change as needed)

# Mark top-k by SHAP
merged["rank_by_shap"] = merged["mean_abs_shap"].rank(ascending=False, method="min")
filtered = merged[
    (merged["selection_freq"] >= selection_freq_threshold) &
    (merged["rank_by_shap"] <= top_k_shap)
].copy()

filtered = filtered.sort_values("combined_score", ascending=False).reset_index(drop=True)

print("Number of proteins passing filters:", filtered.shape[0])

# Save full ranked list
merged.to_csv(f"{OUTPUT_DIR}{OUTCOME}_protein_importance_full.csv", index=False)

# Save only the filtered/stable important proteins
filtered.to_csv(f"{OUTPUT_DIR}{OUTCOME}_protein_importance_filtered_for_bed.csv", index=False)

print(f"Done. Feature importance has been written to {OUTPUT_DIR}")
