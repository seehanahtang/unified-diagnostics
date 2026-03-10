#!/usr/bin/env python3
"""
Run ensemble methods over cancer risk predictions from multiple base models.

Loads probabilities from cox, rsf, xgb, ordinal for each cancer and
time horizon; merges on eid; uses validation set as meta-model train and test
set as evaluation. There is no external validation set for cancer.

Sex-specific cohorts follow run_xgb_benchmark_cancer.py:
- Breast, ovarian, uterine cancers: *_female cohorts
- Prostate cancer: *_male cohorts
- Other cancers: default cohorts.

Runs simple_ensemble, weighted_ensemble, weighted_softmax_ensemble, pnn_ensemble
and writes a summary CSV with valid_AUC, test_AUC per
(cancer, time_frame, ensemble_method).

Usage:
    python run_ensemble_cancer_risk.py [--data-path PATH] [--labels-path PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import os
import warnings
from itertools import product

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from models import (
    pnn_ensemble,
    weighted_softmax_ensemble,
)

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

DATA_PATH = "/orcd/pool/003/dbertsim_shared/ukb/cancer_risk"

MODELS_COX_XGB = ["cox", "xgb", "rsf"]  # filename: {disease}_{model}_risk_{dataset}.csv
MODELS_PROB_NAMING = ["ordinal"]  # filename: {disease}_{dataset}.csv

MODELS = MODELS_COX_XGB 

# cox/xgb/rsf column names
RISK_COLS = ["risk_1years", "risk_2years", "risk_5years", "risk_10years"]
# ordinal column names
PROB_COLS = ["prob_1yr", "prob_2yr", "prob_5yr", "prob_10yr"]

# TIME_FRAMES = [1, 2, 5, 10]
TIME_FRAMES = [5]
TIME_TO_RISK_COL = {1: "risk_1years", 2: "risk_2years", 5: "risk_5years", 10: "risk_10years"}
TIME_TO_PROB_COL = {1: "prob_1yr", 2: "prob_2yr", 5: "prob_5yr", 10: "prob_10yr"}

CANCER_TYPES = [
    "breast_cancer",
    "colorectal_cancer",
    "kidney_cancer",
    "lung_cancer",
    "lymphoma_cancer",
    "melanoma_cancer",
    "ovarian_cancer",
    "prostate_cancer",
    "skin_cancer",
    "uterine_cancer",
]

# Backwards-compatible alias for CLI default
DISEASES = CANCER_TYPES

# Sex-specific cohorts (aligned with run_xgb_benchmark_cancer.py)
FEMALE_SPECIFIC_CANCERS = {"breast_cancer", "ovarian_cancer", "uterine_cancer"}
MALE_SPECIFIC_CANCERS = {"prostate_cancer"}


def get_cancer_data_suffix(diag_type: str) -> str:
    """Return '' for default, '_female' for female-specific, '_male' for male-specific."""
    key = diag_type.lower().replace(" ", "_").strip()
    if key in FEMALE_SPECIFIC_CANCERS:
        return "_female"
    if key in MALE_SPECIFIC_CANCERS:
        return "_male"
    return ""


def load_data_cancer(labels_path: str):
    """
    Load cancer train/valid/test with default + female + male cohorts.

    Mirrors the data-loading logic in run_xgb_benchmark_cancer.py.
    """
    base = labels_path.rstrip("/") + "/"
    print("Loading cancer label data...")
    train_df = pd.read_csv(os.path.join(base, "ukb_cancer_train.csv"), low_memory=False)
    valid_df = pd.read_csv(os.path.join(base, "ukb_cancer_valid.csv"), low_memory=False)
    test_df = pd.read_csv(os.path.join(base, "ukb_cancer_test.csv"), low_memory=False)
    train_f = pd.read_csv(os.path.join(base, "ukb_cancer_train_female.csv"), low_memory=False)
    valid_f = pd.read_csv(os.path.join(base, "ukb_cancer_valid_female.csv"), low_memory=False)
    test_f = pd.read_csv(os.path.join(base, "ukb_cancer_test_female.csv"), low_memory=False)
    train_m = pd.read_csv(os.path.join(base, "ukb_cancer_train_male.csv"), low_memory=False)
    valid_m = pd.read_csv(os.path.join(base, "ukb_cancer_valid_male.csv"), low_memory=False)
    test_m = pd.read_csv(os.path.join(base, "ukb_cancer_test_male.csv"), low_memory=False)

    cancer_sets = {
        "": (train_df, valid_df, test_df),
        "_female": (train_f, valid_f, test_f),
        "_male": (train_m, valid_m, test_m),
    }
    print(f"  Default: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")
    print(f"  Female:  train={len(train_f)}, valid={len(valid_f)}, test={len(test_f)}")
    print(f"  Male:    train={len(train_m)}, valid={len(valid_m)}, test={len(test_m)}")
    return cancer_sets


def get_cancer_types(df: pd.DataFrame) -> list[str]:
    """Extract cancer types from column names (columns ending with 'cancer')."""
    return [c for c in df.columns if c.endswith("cancer")]


def get_labels_future(df: pd.DataFrame, diag_type: str, horizon_years: int) -> pd.Series | None:
    """Binary labels for future diagnosis within horizon. Positive = diagnosed within horizon."""
    time_col = f"{diag_type}_time_to_diagnosis"
    if time_col not in df.columns:
        return None
    y = ((df[diag_type] == 0) & (df[time_col] <= horizon_years)).astype(int)
    return y


def filter_for_future_prediction(df: pd.DataFrame, diag_type: str) -> pd.DataFrame:
    """Keep only patients not diagnosed at baseline."""
    if diag_type not in df.columns:
        return df
    return df.loc[df[diag_type] == 0].copy()


def load_risk_csv(
    data_path: str,
    model: str,
    disease: str,
    dataset: str,
    is_cox_xgb: bool,
) -> pd.DataFrame | None:
    """Load one risk CSV. Returns None if file missing."""
    if is_cox_xgb:
        # dataset in train/valid/test
        fname = f"{disease}_{model}_risk_{dataset}.csv"
    else:
        fname = f"{disease}_{dataset}.csv"
    path = os.path.join(data_path, model, fname)
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path, low_memory=False).dropna(subset=["eid"])


def load_probs_for_split(
    data_path: str,
    disease: str,
    time_frame: int,
    dataset: str,
) -> pd.DataFrame | None:
    """
    Load probabilities for one (disease, time_frame, dataset) from all models.
    Returns DataFrame with columns [eid, model_0, model_1, ...] or None if any file missing.
    """
    risk_col = TIME_TO_RISK_COL[time_frame]
    prob_col = TIME_TO_PROB_COL[time_frame]

    dfs = []
    for model in MODELS:
        is_cox_xgb = model in MODELS_COX_XGB
        df = load_risk_csv(data_path, model, disease, dataset, is_cox_xgb)
        if df is None:
            return None
        col = risk_col if is_cox_xgb else prob_col
        if col not in df.columns:
            return None
        dfs.append(df[["eid", col]].rename(columns={col: model}))

    out = dfs[0]
    for d in dfs[1:]:
        out = out.merge(d, on="eid", how="inner")
    return out


def run_one(
    data_path: str,
    labels_path: str,
    disease: str,
    time_frame: int,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> list[dict]:
    """Run all ensemble methods for one (disease, time_frame). Returns list of row dicts."""
    # Filter to future prediction cohort and get labels
    valid_f = filter_for_future_prediction(valid_df, disease)
    test_f = filter_for_future_prediction(test_df, disease)

    y_valid = get_labels_future(valid_f, disease, time_frame)
    y_test = get_labels_future(test_f, disease, time_frame)

    if y_valid is None or y_test is None:
        return []

    # Align: only rows present in risk files and with valid labels
    X_valid_probs = load_probs_for_split(data_path, disease, time_frame, "valid")
    X_test_probs = load_probs_for_split(data_path, disease, time_frame, "test")
    if X_valid_probs is None or X_test_probs is None:
        return []

    # Merge labels with risk eids so we have same order
    valid_merge = X_valid_probs[["eid"]].merge(
        valid_f[["eid"]].assign(y=y_valid.values),
        on="eid",
        how="inner",
    )
    test_merge = X_test_probs[["eid"]].merge(
        test_f[["eid"]].assign(y=y_test.values),
        on="eid",
        how="inner",
    )

    print("Merged eids: ", len(valid_merge), len(test_merge))

    y_train = valid_merge["y"].values
    y_test_arr = test_merge["y"].values

    # Align probability matrices to merged eids
    train_eids = valid_merge["eid"].values
    test_eids = test_merge["eid"].values

    X_train_probs = X_valid_probs.set_index("eid").loc[train_eids][MODELS].values
    X_test_probs_arr = X_test_probs.set_index("eid").loc[test_eids][MODELS].values

    # Skip if all probabilities are NaN for this (disease, time_frame)
    if np.isnan(X_train_probs).all() or np.isnan(X_test_probs_arr).all():
        print(f"Skipping {disease} @ {time_frame}yr: all probabilities are NaN")
        return []

    # Restrict to complete-case rows (no NaN in any model) so sklearn and ensemble get finite inputs
    train_ok = ~np.isnan(X_train_probs).any(axis=1)
    test_ok = ~np.isnan(X_test_probs_arr).any(axis=1)
    if train_ok.sum() == 0 or test_ok.sum() == 0:
        print(f"Skipping {disease} @ {time_frame}yr: no complete-case rows (all have at least one NaN)")
        return []

    # Validation AUCs per base model (for weighted methods)
    validation_aucs = np.array(
        [roc_auc_score(y_train, X_train_probs[:, i]) for i in range(len(MODELS))]
    )

    rows: list[dict] = []

    # ------------------------------------------------------------------
    # Base-model rows: report valid / test AUC for all models
    # ------------------------------------------------------------------
    for j, model_name in enumerate(MODELS):
        # Validation AUC (already computed above)
        valid_auc_model = validation_aucs[j]

        # Test AUC for this base model
        try:
            test_auc_model = roc_auc_score(y_test_arr, X_test_probs_arr[:, j])
        except ValueError:
            test_auc_model = np.nan

        rows.append({
            "disease": disease,
            "time_frame": time_frame,
            "ensemble_method": model_name,
            "valid_AUC": valid_auc_model,
            "test_AUC": test_auc_model,
            "external_AUC": np.nan,
            "n_valid": len(y_train),
            "n_test": len(y_test_arr),
            "n_ext": 0,
        })

    def add_row(method: str, result, valid_auc_override=None):
        valid_auc = valid_auc_override if valid_auc_override is not None else result.metrics.get("AUC_binary", np.nan)
        test_auc = roc_auc_score(result.y_test, result.proba_test[:, 1])
        rows.append({
            "disease": disease,
            "time_frame": time_frame,
            "ensemble_method": method,
            "valid_AUC": valid_auc,
            "test_AUC": test_auc,
            "external_AUC": np.nan,
            "n_valid": len(y_train),
            "n_test": len(y_test_arr),
            "n_ext": 0,
        })

    # 1. Simple ensemble
    # res_simple = simple_ensemble(
    #     X_test_probs_arr, y_test_arr,
    # )
    # # Valid AUC: mean probs on validation set
    # p_valid = X_train_probs.mean(axis=1)
    # valid_auc_simple = roc_auc_score(y_train, p_valid)
    # rows.append({
    #     "disease": disease,
    #     "time_frame": time_frame,
    #     "ensemble_method": "simple_ensemble",
    #     "valid_AUC": valid_auc_simple,
    #     "test_AUC": res_simple.metrics["AUC_binary"],
    #     "external_AUC": np.nan,
    #     "n_valid": len(y_train),
    #     "n_test": len(y_test_arr),
    #     "n_ext": 0,
    # })

    # 2. Weighted ensemble
    # res_w = weighted_ensemble(
    #     X_test_probs_arr, y_test_arr,
    #     validation_aucs=validation_aucs,
    # )
    # p_valid_w = X_train_probs @ (validation_aucs / validation_aucs.sum())
    # valid_auc_w = roc_auc_score(y_train, p_valid_w)
    # rows.append({
    #     "disease": disease,
    #     "time_frame": time_frame,
    #     "ensemble_method": "weighted_ensemble",
    #     "valid_AUC": valid_auc_w,
    #     "test_AUC": res_w.metrics["AUC_binary"],
    #     "external_AUC": np.nan,
    #     "n_valid": len(y_train),
    #     "n_test": len(y_test_arr),
    #     "n_ext": 0,
    # })

    # 3. Weighted softmax ensemble
    from models import _softmax_weights

    res_sw = weighted_softmax_ensemble(
        X_test_probs_arr, y_test_arr,
        validation_aucs=validation_aucs,
    )
    w = _softmax_weights(validation_aucs, T=0.1)
    p_valid_sw = X_train_probs @ w
    valid_auc_sw = roc_auc_score(y_train, p_valid_sw)
    rows.append({
        "disease": disease,
        "time_frame": time_frame,
        "ensemble_method": "weighted_softmax_ensemble",
        "valid_AUC": valid_auc_sw,
        "test_AUC": res_sw.metrics["AUC_binary"],
        "external_AUC": np.nan,
        "n_valid": len(y_train),
        "n_test": len(y_test_arr),
        "n_ext": 0,
    })

    # 4. PNN ensemble
    res_pnn = None
    try:
        res_pnn = pnn_ensemble(
            X_train_probs, y_train,
            X_test_probs_arr, y_test_arr,
        )
        # Valid AUC: forward pass on validation set
        device = next(res_pnn.model.parameters()).device
        with torch.no_grad():
            logits = res_pnn.model(torch.from_numpy(X_train_probs.astype(np.float32)).to(device))
            p_valid_pnn = (1.0 / (1.0 + np.exp(-logits.cpu().numpy().ravel())))
        valid_auc_pnn = roc_auc_score(y_train, p_valid_pnn)
        add_row("pnn_ensemble", res_pnn, valid_auc_override=valid_auc_pnn)
    except Exception as e:
        rows.append({
            "disease": disease,
            "time_frame": time_frame,
            "ensemble_method": "pnn_ensemble",
            "valid_AUC": np.nan,
            "test_AUC": np.nan,
            "external_AUC": np.nan,
            "n_valid": len(y_train),
            "n_test": len(y_test_arr),
            "n_ext": 0,
            "error": str(e),
        })

    # ------------------------------------------------------------------
    # Write predicted test probabilities at 5-year horizon
    # ------------------------------------------------------------------
    if time_frame == 5:
        risk_col = TIME_TO_RISK_COL[time_frame]
        ensemble_preds = {
            "weighted_softmax_ensemble": res_sw.proba_test[:, 1],
        }
        if res_pnn is not None:
            ensemble_preds["pnn_ensemble"] = res_pnn.proba_test[:, 1]

        for method_name, preds in ensemble_preds.items():
            out_dir = os.path.join(data_path, method_name)
            os.makedirs(out_dir, exist_ok=True)
            risk_df = pd.DataFrame({
                "eid": test_eids,
                risk_col: preds,
            })
            out_path = os.path.join(out_dir, f"{disease}_{method_name}_risk_test.csv")
            risk_df.to_csv(out_path, index=False)
            print(f"  Wrote {len(risk_df)} rows to {out_path}")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Run ensemble methods on cancer risk predictions")
    parser.add_argument(
        "--data-path",
        type=str,
        default=DATA_PATH,
        help="Base path to cancer_risk (contains cox/, rsf/, xgb/, ordinal/)",
    )
    parser.add_argument(
        "--labels-path",
        type=str,
        default=None,
        help="Path to folder with ukb_cancer_*.csv. Default: parent of data-path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ensemble_cancer_summary.csv",
        help="Output summary CSV path",
    )
    parser.add_argument(
        "--diseases",
        type=str,
        nargs="*",
        default=None,
        help="Subset of cancers (default: all in data)",
    )
    parser.add_argument(
        "--time-frames",
        type=int,
        nargs="*",
        default=None,
        help="Subset of time frames in years (default: 1 2 5 10)",
    )
    args = parser.parse_args()

    data_path = os.path.abspath(args.data_path)
    labels_path = args.labels_path or os.path.dirname(data_path)
    labels_path = os.path.abspath(labels_path)

    # Load cancer cohorts (default + female + male) using the same logic
    # as run_xgb_benchmark_cancer.py.
    cancer_sets = load_data_cancer(labels_path)
    train_default, valid_default, test_default = cancer_sets[""]

    # Diseases default to all cancer types present in the data unless a subset is specified
    diseases = args.diseases or get_cancer_types(train_default)
    time_frames = args.time_frames or TIME_FRAMES

    all_rows: list[dict] = []
    n_tasks = len(diseases) * len(time_frames)
    for idx, (disease, tf) in enumerate(product(diseases, time_frames)):
        suffix = get_cancer_data_suffix(disease)
        _, valid_df, test_df = cancer_sets.get(suffix, cancer_sets[""])

        # Ensure the required columns exist for this cancer in the chosen cohort
        if disease not in valid_df.columns or f"{disease}_time_to_diagnosis" not in valid_df.columns:
            print(f"Skipping {disease} (not present in {suffix or 'default'} cohort)")
            continue

        print(f"[{idx + 1}/{n_tasks}] {disease} @ {tf}yr (cohort: {suffix or 'default'})")
        rows = run_one(
            data_path, labels_path, disease, tf,
            valid_df, test_df,
        )
        all_rows.extend(rows)

    summary = pd.DataFrame(all_rows)
    summary.to_csv(args.output, index=False)
    print(f"Wrote {len(summary)} rows to {args.output}")


if __name__ == "__main__":
    main()
