#!/usr/bin/env python

import argparse
import os

import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

DATA_DIR = "/orcd/pool/003/dbertsim_shared/ukb"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine protein (Olink) + genomic features and run XGBoost."
    )
    parser.add_argument(
        "--outcome-col",
        type=str,
        required=True,
        help="Name of the outcome column (target variable).",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="eid",
        help="ID column common to protein and variant files (default: eid).",
    )
    parser.add_argument(
        "--olink-prefix",
        type=str,
        default="olink_",
        help="Prefix for Olink protein features (default: 'olink_').",
    )
    parser.add_argument(
        "--chrom",
        type=int,
        default=18,
        help="Chromosome number.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of threads for XGBoost (default: -1 = all).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="ukb_cancer_prot_geno_xgb",
        help="Prefix for output prediction files.",
    )
    return parser.parse_args()


def load_data(args):
    # --- Protein data ---
    train_prot_path = os.path.join(DATA_DIR, "ukb_cancer_train.csv")
    valid_prot_path = os.path.join(DATA_DIR, "ukb_cancer_valid.csv")

    print(f"Loading protein train from {train_prot_path}")
    train_prot = pd.read_csv(train_prot_path)
    print(f"Loading protein valid from {valid_prot_path}")
    valid_prot = pd.read_csv(valid_prot_path)

    # --- Selected variant data ---
    train_var_path = f"ukb_train_{args.outcome_col}_selected_variants.csv"
    valid_var_path = f"ukb_valid_{args.outcome_col}_selected_variants.csv"

    print(f"Loading variant train from {train_var_path}")
    train_var = pd.read_csv(train_var_path)
    print(f"Loading variant valid from {valid_var_path}")
    valid_var = pd.read_csv(valid_var_path)

    # --- Merge on ID column ---
    if args.id_col not in train_prot.columns or args.id_col not in train_var.columns:
        raise ValueError(
            f"ID column '{args.id_col}' must be present in both protein and variant train data."
        )
    if args.id_col not in valid_prot.columns or args.id_col not in valid_var.columns:
        raise ValueError(
            f"ID column '{args.id_col}' must be present in both protein and variant valid data."
        )

    print(f"Merging train on {args.id_col}...")
    train_merged = train_prot.merge(train_var, on=[args.id_col,args.outcome_col], how="inner")

    print(f"Merging valid on {args.id_col}...")
    valid_merged = valid_prot.merge(valid_var, on=[args.id_col,args.outcome_col], how="inner")

    return train_merged, valid_merged


def select_features(train_df, valid_df, args):
    # Olink protein features
    olink_cols = [c for c in train_df.columns if c.startswith(args.olink_prefix)]

    # SNP features (columns that begin with snp-prefix, e.g. 'c18...', 'c1...')
    snp_cols = [c for c in train_df.columns if c.startswith(args.snp_prefix)]

    if len(olink_cols) == 0:
        print("Warning: no protein columns found with prefix:", args.olink_prefix)
    if len(snp_cols) == 0:
        print("Warning: no SNP columns found with prefix:", args.snp_prefix)

    feature_cols = olink_cols + snp_cols

    # Sanity check: make sure outcome is present
    if args.outcome_col not in train_df.columns:
        raise ValueError(f"Outcome column '{args.outcome_col}' not found in train data.")

    X_train = train_df[feature_cols]
    y_train = train_df[args.outcome_col]

    X_valid = valid_df[feature_cols]
    y_valid = valid_df[args.outcome_col]

    print(f"Using {len(feature_cols)} features total:")
    print(f"  {len(olink_cols)} protein (Olink) features")
    print(f"  {len(snp_cols)} SNP features")

    return X_train, y_train, X_valid, y_valid, feature_cols


def train_xgb(X_train, y_train, X_valid, y_valid, args):
    # >>> Put your *existing* hyperparameters here to keep them identical <<<
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

    print("Fitting XGBoost model on combined protein + genomic features...")
    model.fit(
        X_train,
        y_train,
        # eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=True,
    )

    # Predict probabilities on valid set
    valid_pred_proba = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, valid_pred_proba)
    print(f"Validation AUC: {auc:.4f}")

    return model, valid_pred_proba, auc


def main():
    args = parse_args()
    args.snp_prefix = f"{args.chrom}"
    

    train_merged, valid_merged = load_data(args)
    X_train, y_train, X_valid, y_valid, feature_cols = select_features(
        train_merged, valid_merged, args
    )

    model, valid_pred_proba, auc = train_xgb(X_train, y_train, X_valid, y_valid, args)

    # Save predictions & feature list
    # pred_out_path = f"{args.output_prefix}_valid_preds.csv"
    # feat_out_path = f"{args.output_prefix}_feature_list.txt"
    model_out_path = f"{args.output_prefix}_model.json"

    # print(f"Saving validation predictions to {pred_out_path}")
    # pd.DataFrame(
    #     {
    #         args.id_col: valid_merged[args.id_col].values,
    #         args.outcome_col: y_valid.values,
    #         "pred_proba": valid_pred_proba,
    #     }
    # ).to_csv(pred_out_path, index=False)

    # print(f"Saving feature list to {feat_out_path}")
    # with open(feat_out_path, "w") as f:
    #     for c in feature_cols:
    #         f.write(c + "\n")

    print(f"Saving XGBoost model to {model_out_path}")
    model.save_model(model_out_path)

    print("Done.")


if __name__ == "__main__":
    main()
