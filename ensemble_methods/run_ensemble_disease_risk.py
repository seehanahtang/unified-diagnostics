#!/usr/bin/env python3
"""
Run ensemble methods over disease risk predictions from multiple base models.

Loads probabilities from cox, ordinal, m3h, ordinal_m3h, xgb for each disease and
time horizon; merges on eid; uses validation set as meta-model train, test as test,
external as external. Runs simple_ensemble, weighted_ensemble,
weighted_softmax_ensemble, pnn_ensemble and writes a summary CSV with valid_AUC,
test_AUC, external_AUC per (disease, time_frame, ensemble_method).

Usage:
    python run_ensemble_disease_risk.py [--data-path PATH] [--labels-path PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import os
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch

from models import (
    pnn_ensemble,
    simple_ensemble,
    weighted_ensemble,
    weighted_softmax_ensemble,
)

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

DATA_PATH = "/orcd/pool/003/dbertsim_shared/ukb/disease_risk"

MODELS_COX_XGB = ["cox", "xgb"]  # filename: {disease}_{model}_risk_{dataset}.csv
MODELS_PROB_NAMING = ["ordinal"]  # filename: {disease}_{dataset}.csv

MODEL_DIR_OVERRIDE = {"ordinal": "ordinal_updated"}

MODELS = MODELS_COX_XGB + MODELS_PROB_NAMING

# cox/xgb column names
RISK_COLS = ["risk_1years", "risk_2years", "risk_5years", "risk_10years"]
# m3h/ordinal/ordinal_m3h column names
PROB_COLS = ["prob_1yr", "prob_2yr", "prob_5yr", "prob_10yr"]

TIME_FRAMES = [1, 2, 5, 10]
TIME_TO_RISK_COL = {1: "risk_1years", 2: "risk_2years", 5: "risk_5years", 10: "risk_10years"}
TIME_TO_PROB_COL = {1: "prob_1yr", 2: "prob_2yr", 5: "prob_5yr", 10: "prob_10yr"}

DISEASES = [
    "acute_kidney_injury",
    "alzheimers_disease",
    "atrial_fibrillation",
    "chronic_kidney_disease",
    "copd",
    "end_stage_renal_disease",
    "heart_failure",
    "hypertensive_heart_kidney_diseases",
    "ischemic_heart_disease",
    "liver_disease",
    "lower_respiratory_disease",
    "other_dementia",
    "parkinsons",
    "peripheral_vascular_disease",
    "stroke",
    "type_1_diabetes",
    "type_2_diabetes",
]

ENSEMBLE_METHODS = [
    "simple_ensemble",
    "weighted_ensemble",
    "weighted_softmax_ensemble",
    "pnn_ensemble",
]


def get_labels_future(df: pd.DataFrame, diag_type: str, horizon_years: int) -> pd.Series | None:
    """Binary labels for future diagnosis within horizon. Positive = diagnosed within horizon."""
    time_col = f"{diag_type}_time_to_diagnosis"
    if time_col not in df.columns:
        return None
    # Positive: not diagnosed at baseline but diagnosed within horizon
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
        # dataset in train/valid/external/test
        fname = f"{disease}_{model}_risk_{dataset}.csv"
    else:
        # train -> validation for ordinal/m3h/ordinal_m3h
        ds = "validation" if dataset == "valid" else dataset
        fname = f"{disease}_{ds}.csv"
    model_dir = MODEL_DIR_OVERRIDE.get(model, model)
    path = os.path.join(data_path, model_dir, fname)
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
    external_df: pd.DataFrame | None,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    """Run all ensemble methods for one (disease, time_frame).

    Returns (rows, ext_preds) where ext_preds maps ensemble method name to a
    DataFrame with columns [eid, risk_{tf}years] for the external set.
    """
    # Filter to future prediction cohort and get labels
    valid_f = filter_for_future_prediction(valid_df, disease)
    test_f = filter_for_future_prediction(test_df, disease)
    external_f = filter_for_future_prediction(external_df, disease) if external_df is not None else None

    y_valid = get_labels_future(valid_f, disease, time_frame)
    y_test = get_labels_future(test_f, disease, time_frame)
    y_ext = get_labels_future(external_f, disease, time_frame) if external_f is not None else None

    if y_valid is None or y_test is None:
        return [], {}
    # Align: only rows present in risk files and with valid labels
    X_valid_probs = load_probs_for_split(data_path, disease, time_frame, "valid")
    X_test_probs = load_probs_for_split(data_path, disease, time_frame, "test")
    X_ext_probs = load_probs_for_split(data_path, disease, time_frame, "external") if external_df is not None else None

    if X_valid_probs is None or X_test_probs is None:
        return [], {}

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
    ext_merge = None
    if X_ext_probs is not None and y_ext is not None and external_f is not None:
        ext_merge = X_ext_probs[["eid"]].merge(
            external_f[["eid"]].assign(y=y_ext.values),
            on="eid",
            how="inner",
        )

    y_train = valid_merge["y"].values
    y_test_arr = test_merge["y"].values
    y_ext_arr = ext_merge["y"].values if ext_merge is not None else None

    # Align probability matrices to merged eids
    train_eids = valid_merge["eid"].values
    test_eids = test_merge["eid"].values
    ext_eids = ext_merge["eid"].values if ext_merge is not None else None

    X_train_probs = X_valid_probs.set_index("eid").loc[train_eids][MODELS].values
    X_test_probs_arr = X_test_probs.set_index("eid").loc[test_eids][MODELS].values
    X_ext_probs_arr = None
    if ext_eids is not None and X_ext_probs is not None:
        X_ext_probs_arr = X_ext_probs.set_index("eid").loc[ext_eids][MODELS].values

    # Skip if all probabilities are NaN for this (disease, time_frame)
    if np.isnan(X_train_probs).all() or np.isnan(X_test_probs_arr).all():
        print(f"Skipping {disease} @ {time_frame}yr: all probabilities are NaN")
        return [], {}

    # Restrict to complete-case rows (no NaN in any model) so sklearn and ensemble get finite inputs
    train_ok = ~np.isnan(X_train_probs).any(axis=1)
    test_ok = ~np.isnan(X_test_probs_arr).any(axis=1)
    if train_ok.sum() == 0 or test_ok.sum() == 0:
        print(f"Skipping {disease} @ {time_frame}yr: no complete-case rows (all have at least one NaN)")
        return [], {}

    # Validation AUCs per base model (for weighted methods)
    validation_aucs = np.array(
        [roc_auc_score(y_train, X_train_probs[:, i]) for i in range(len(MODELS))]
    )

    rows = []
    ext_preds: dict[str, pd.DataFrame] = {}
    risk_col = TIME_TO_RISK_COL[time_frame]

    # ------------------------------------------------------------------
    # Base-model rows: report valid / test / external AUC for all models
    # ------------------------------------------------------------------
    for j, model_name in enumerate(MODELS):
        # Validation AUC (already computed above)
        valid_auc_model = validation_aucs[j]

        # Test AUC for this base model
        try:
            test_auc_model = roc_auc_score(y_test_arr, X_test_probs_arr[:, j])
        except ValueError:
            test_auc_model = np.nan

        # External AUC for this base model (if external data available)
        ext_auc_model = np.nan
        if X_ext_probs_arr is not None and y_ext_arr is not None:
            try:
                ext_auc_model = roc_auc_score(y_ext_arr, X_ext_probs_arr[:, j])
            except ValueError:
                ext_auc_model = np.nan

        rows.append({
            "disease": disease,
            "time_frame": time_frame,
            "ensemble_method": model_name,
            "valid_AUC": valid_auc_model,
            "test_AUC": test_auc_model,
            "external_AUC": ext_auc_model,
            "n_valid": len(y_train),
            "n_test": len(y_test_arr),
            "n_ext": len(y_ext_arr) if y_ext_arr is not None else 0,
        })

    def add_row(method: str, result, valid_auc_override=None):
        valid_auc = valid_auc_override if valid_auc_override is not None else result.metrics.get("AUC_binary", np.nan)
        test_auc = roc_auc_score(result.y_test, result.proba_test[:, 1])
        ext_auc = np.nan
        if result.proba_ext is not None and result.y_ext is not None:
            ext_auc = roc_auc_score(result.y_ext, result.proba_ext[:, 1])
        rows.append({
            "disease": disease,
            "time_frame": time_frame,
            "ensemble_method": method,
            "valid_AUC": valid_auc,
            "test_AUC": test_auc,
            "external_AUC": ext_auc,
            "n_valid": len(y_train),
            "n_test": len(y_test_arr),
            "n_ext": len(y_ext_arr) if y_ext_arr is not None else 0,
        })

    # 1. Simple ensemble
    res = simple_ensemble(
        X_test_probs_arr, y_test_arr,
        X_ext_probs=X_ext_probs_arr,
        y_ext=y_ext_arr,
    )
    # Valid AUC: mean probs on validation set
    p_valid = X_train_probs.mean(axis=1)
    valid_auc_simple = roc_auc_score(y_train, p_valid)
    rows.append({
        "disease": disease, "time_frame": time_frame, "ensemble_method": "simple_ensemble",
        "valid_AUC": valid_auc_simple, "test_AUC": res.metrics["AUC_binary"],
        "external_AUC": res.metrics.get("Ext_AUC_binary", np.nan),
        "n_valid": len(y_train), "n_test": len(y_test_arr),
        "n_ext": len(y_ext_arr) if y_ext_arr is not None else 0,
    })
    if ext_eids is not None and res.proba_ext is not None:
        ext_preds["simple_ensemble"] = pd.DataFrame({"eid": ext_eids, risk_col: res.proba_ext[:, 1]})

    # 2. Weighted ensemble
    res = weighted_ensemble(
        X_test_probs_arr, y_test_arr,
        validation_aucs=validation_aucs,
        X_ext_probs=X_ext_probs_arr,
        y_ext=y_ext_arr,
    )
    p_valid_w = X_train_probs @ (validation_aucs / validation_aucs.sum())
    valid_auc_w = roc_auc_score(y_train, p_valid_w)
    rows.append({
        "disease": disease, "time_frame": time_frame, "ensemble_method": "weighted_ensemble",
        "valid_AUC": valid_auc_w, "test_AUC": res.metrics["AUC_binary"],
        "external_AUC": res.metrics.get("Ext_AUC_binary", np.nan),
        "n_valid": len(y_train), "n_test": len(y_test_arr),
        "n_ext": len(y_ext_arr) if y_ext_arr is not None else 0,
    })
    if ext_eids is not None and res.proba_ext is not None:
        ext_preds["weighted_ensemble"] = pd.DataFrame({"eid": ext_eids, risk_col: res.proba_ext[:, 1]})

    # 3. Weighted softmax ensemble
    res = weighted_softmax_ensemble(
        X_test_probs_arr, y_test_arr,
        validation_aucs=validation_aucs,
        X_ext_probs=X_ext_probs_arr,
        y_ext=y_ext_arr,
    )
    from models import _softmax_weights
    w = _softmax_weights(validation_aucs, T=0.1)
    p_valid_sw = X_train_probs @ w
    valid_auc_sw = roc_auc_score(y_train, p_valid_sw)
    rows.append({
        "disease": disease, "time_frame": time_frame,
        "ensemble_method": "weighted_softmax_ensemble",
        "valid_AUC": valid_auc_sw, "test_AUC": res.metrics["AUC_binary"],
        "external_AUC": res.metrics.get("Ext_AUC_binary", np.nan),
        "n_valid": len(y_train), "n_test": len(y_test_arr),
        "n_ext": len(y_ext_arr) if y_ext_arr is not None else 0,
    })
    if ext_eids is not None and res.proba_ext is not None:
        ext_preds["weighted_softmax_ensemble"] = pd.DataFrame({"eid": ext_eids, risk_col: res.proba_ext[:, 1]})

    # 4. PNN ensemble
    try:
        res = pnn_ensemble(
            X_train_probs, y_train,
            X_test_probs_arr, y_test_arr,
            X_ext_probs=X_ext_probs_arr,
            y_ext=y_ext_arr,
        )
        # Valid AUC: forward pass on validation set
        import torch
        device = next(res.model.parameters()).device
        with torch.no_grad():
            logits = res.model(torch.from_numpy(X_train_probs.astype(np.float32)).to(device))
            p_valid_pnn = (1.0 / (1.0 + np.exp(-logits.cpu().numpy().ravel())))
        valid_auc_pnn = roc_auc_score(y_train, p_valid_pnn)
        add_row("pnn_ensemble", res, valid_auc_override=valid_auc_pnn)
        if ext_eids is not None and res.proba_ext is not None:
            ext_preds["pnn_ensemble"] = pd.DataFrame({"eid": ext_eids, risk_col: res.proba_ext[:, 1]})
    except Exception as e:
        rows.append({
            "disease": disease, "time_frame": time_frame, "ensemble_method": "pnn_ensemble",
            "valid_AUC": np.nan, "test_AUC": np.nan, "external_AUC": np.nan,
            "n_valid": len(y_train), "n_test": len(y_test_arr),
            "n_ext": len(y_ext_arr) if y_ext_arr is not None else 0,
            "error": str(e),
        })

    return rows, ext_preds


def main():
    parser = argparse.ArgumentParser(description="Run ensemble methods on disease risk predictions")
    parser.add_argument(
        "--data-path",
        type=str,
        default=DATA_PATH,
        help="Base path to disease_risk (contains cox/, ordinal/, m3h/, ordinal_m3h/, xgb/)",
    )
    parser.add_argument(
        "--labels-path",
        type=str,
        default=None,
        help="Path to folder with ukb_disease_*.csv. Default: parent of data-path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ensemble_summary.csv",
        help="Output summary CSV path",
    )
    parser.add_argument(
        "--diseases",
        type=str,
        nargs="*",
        default=None,
        help="Subset of diseases (default: all)",
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
    diseases = args.diseases or DISEASES
    time_frames = args.time_frames or TIME_FRAMES

    print("Loading label data...")
    valid_df = pd.read_csv(os.path.join(labels_path, "ukb_disease_valid.csv"), low_memory=False)
    test_df = pd.read_csv(os.path.join(labels_path, "ukb_disease_test.csv"), low_memory=False)
    external_path = os.path.join(labels_path, "ukb_disease_scotland_wales.csv")
    external_df = pd.read_csv(external_path, low_memory=False) if os.path.isfile(external_path) else None
    print(f"  Valid: {len(valid_df)}, Test: {len(test_df)}, External: {len(external_df) if external_df is not None else 0}")

    all_rows = []
    # (disease, method) -> list of DataFrames, one per time_frame, each with [eid, risk_{tf}years]
    all_ext_preds: dict[tuple[str, str], list[pd.DataFrame]] = {}

    n_tasks = len(diseases) * len(time_frames)
    for idx, (disease, tf) in enumerate(product(diseases, time_frames)):
        print(f"[{idx + 1}/{n_tasks}] {disease} @ {tf}yr")
        rows, ext_preds = run_one(
            data_path, labels_path, disease, tf,
            valid_df, test_df, external_df,
        )
        all_rows.extend(rows)
        for method, df in ext_preds.items():
            all_ext_preds.setdefault((disease, method), []).append(df)

    summary = pd.DataFrame(all_rows)
    summary.to_csv(args.output, index=False)
    print(f"Wrote {len(summary)} rows to {args.output}")

    # Write external prediction CSVs: disease_risk/{ensemble_method}/{disease}_external.csv
    for (disease, method), dfs in all_ext_preds.items():
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on="eid", how="outer")
        out_dir = os.path.join(data_path, method)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{disease}_external.csv")
        merged.to_csv(out_path, index=False)
        print(f"Wrote external predictions to {out_path}")


if __name__ == "__main__":
    main()
