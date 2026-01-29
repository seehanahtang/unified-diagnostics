import os
import glob
import time
import sys
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

def get_sample_eids():
    """
    Load train/valid CSVs and return their eids as lists.
    Assumes preprocess_categorical() is defined elsewhere.
    """
    train_df = pd.read_csv(f"{DATA_DIR}/ukb_cancer_train.csv")
    valid_df = pd.read_csv(f"{DATA_DIR}/ukb_cancer_valid.csv")
    train_eids = list(train_df["eid"])
    valid_eids = list(valid_df["eid"])
    return train_df, valid_df, train_eids, valid_eids

def get_df(path, samples, variants, chrom=18):
    # 1. Read wide-genotype parquet
    tbl = pq.read_table(path)
    wide = tbl.to_pandas()   # columns: sample_idx, ch8_12345, ch8_67890, ...

    # 2. Attach ID_1 (eid) via sample_idx
    merged = wide.merge(
        samples[["sample_idx", "ID_1"]],
        on="sample_idx",
        how="inner"
    )

    # 3. Drop sample_idx and rename ID_1 -> eid
    merged = merged.drop(columns=["sample_idx"])
    merged = merged.rename(columns={"ID_1": "eid"})

    # 4. Build mapping from variant_idx -> rsid
    #    (variants should be pre-filtered to your BED overlaps, etc.)
    idx_to_rsid = dict(zip(variants["variant_idx"], variants["rsid"]))

    # 5. Figure out which columns are genotype columns
    prefix = f"c{chrom}_"
    geno_cols = [c for c in merged.columns if c.startswith(prefix)]

    # 6. Build renaming dict: ch8_12345 -> rsid_for_12345
    col_rename = {}
    for col in geno_cols:
        try:
            var_idx = int(col.split("_")[1])  # from "ch8_12345" -> 12345
        except (IndexError, ValueError):
            continue

        if var_idx in idx_to_rsid:
            col_rename[col] = idx_to_rsid[var_idx]
        # else: this variant not in your variants list → drop it later

    # 7. Keep only eid + the genotype columns that we know how to map
    cols_keep = ["eid"] + list(col_rename.keys())
    print(path, "number of columns: ", len(cols_keep)-1)
    final = merged[cols_keep].rename(columns=col_rename)

    final = final.replace(-1, np.nan)

    return final


CHROM = 18
OUTCOME = sys.argv[1] #breast, lung, prostate
DATA_DIR = "/orcd/pool/003/dbertsim_shared/ukb"
LOGFILE = f"logs/xgb_chr{CHROM}_{OUTCOME}.log"
BGEN_PATH = f"{DATA_DIR}/bgen/ch{CHROM}"
file = f"c{CHROM}_b0_v1"

train_pheno, test_pheno, train_eids, test_eids = get_sample_eids()

samples = pd.read_csv(f"{BGEN_PATH}/c{CHROM}_b0_v1_samples.csv")

train_samples = samples.loc[samples["ID_1"].isin(train_eids)].copy()
test_samples = samples.loc[samples["ID_1"].isin(test_eids)].copy()

# variants = pd.read_csv(f"{DATA_DIR}/bed/variants_overlaps_{outcome}_cancer.csv")
variants = pd.read_csv(f"variants_overlaps_{OUTCOME}_cancer.csv")
variant_idxs = variants['variant_idx']

parquet_files = sorted(glob.glob(os.path.join(BGEN_PATH, f"wide_format/c{CHROM}_*.parquet")))
train_chunks = []
test_chunks = []

for path in parquet_files:
    start, end = re.findall(r"_(\d+)", path)[-2:]
    start = int(start)
    end = int(end)
    if any(start <= x <= end for x in variant_idxs):
        train_chunk = get_df(path, train_samples, variants, CHROM)
        test_chunk = get_df(path, test_samples, variants, CHROM)

        # Use eid as index to guarantee consistent ordering when concatenating
        train_chunks.append(train_chunk.set_index("eid"))
        test_chunks.append(test_chunk.set_index("eid"))
    
X_train_geno = pd.concat(train_chunks, axis=1)
X_test_geno = pd.concat(test_chunks, axis=1)

X_train_geno = X_train_geno.reset_index()
X_test_geno = X_test_geno.reset_index()

# Merge with phenotype to align labels and features
outcome_col = f"{OUTCOME}_cancer"

train_merge = (
    train_pheno[["eid", outcome_col]]
    .merge(X_train_geno, on="eid", how="inner")
)
valid_merge = (
    test_pheno[["eid", outcome_col]]
    .merge(X_test_geno, on="eid", how="inner")
)
feature_cols = [c for c in train_merge.columns if c not in ["eid", outcome_col]]

X_train = train_merge[feature_cols]
y_train = train_merge[outcome_col].astype(int)

X_valid = valid_merge[feature_cols]
y_valid = valid_merge[outcome_col].astype(int)

train_merge.to_csv(f"ukb_train_{outcome_col}_selected_variants.csv",index=False)
valid_merge.to_csv(f"ukb_valid_{outcome_col}_selected_variants.csv",index=False)

# XGBoost with your fixed hyperparameters
model = XGBClassifier(
    objective="binary:logistic",
    tree_method="hist",
    device="cuda",
    n_estimators=300,
    max_depth=5,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
)
start_time = time.time()
sys.stdout = open(LOGFILE, "a")

print(f"Training XGBoost on chr{CHROM}, outcome={OUTCOME}")
print(f"n_train={len(X_train)}, n_valid={len(X_valid)}")
print("Fixed hyperparameters:")
print({
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.02,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
})
print("========================================")

model.fit(X_train, y_train)

print(f"Time taken: {(time.time() - start_time)/60:.2f} minutes")
print("========================================")

# ------------------ EVALUATE ------------------ #
y_valid_pred_prob = model.predict_proba(X_valid)[:, 1]
auc_valid = roc_auc_score(y_valid, y_valid_pred_prob)
print(f"Validation AUC: {auc_valid:.4f}")
print("========================================")

sys.stdout.close()