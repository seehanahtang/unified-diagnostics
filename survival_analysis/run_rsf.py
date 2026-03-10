#!/usr/bin/env python3
"""
Combined RSF (Random Survival Forest) for disease or cancer outcomes.

Trains RSF per outcome and feature set, reports C-index and train/test AUC at 5 years,
and writes predicted probabilities at 5 years for validation, test, and (disease only) external.

Usage:
    python run_rsf.py --mode disease [--data-path PATH] [--output-dir DIR]
    python run_rsf.py --mode cancer [--data-path PATH] [--output-dir DIR]
"""

import argparse
import json
import os
import sys
import time
import logging
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

# -----------------------
# Logging setup
# -----------------------
def setup_logger(log_dir: str, name: str = "rsf") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# -----------------------
# Config
# -----------------------
DEFAULT_DATA_PATH = "/orcd/pool/003/dbertsim_shared/ukb/"

DEMO_COLS = [
    'Age at recruitment',
    'Sex_male',
    'Body mass index (BMI)',
    'Systolic blood pressure, automated reading',
    'Diastolic blood pressure, automated reading',
    'Townsend deprivation index at recruitment',
    'Smoking status',
    'Alcohol intake frequency.'
]

CATEGORICAL_DEMO = ["Smoking status", "Alcohol intake frequency."]

# FEATURE_SETS = [
#     'demo_protein',
#     'demo_blood',
#     'demo_protein_blood',
# ]
FEATURE_SETS = ['demo_protein_blood']

HORIZON_YEARS = 5

# Cancer outcomes that use sex-specific cohorts (file suffix _female or _male)
FEMALE_SPECIFIC_CANCERS = {"breast_cancer", "ovarian_cancer", "uterine_cancer"}
MALE_SPECIFIC_CANCERS = {"prostate_cancer"}


# -----------------------
# Helpers
# -----------------------
def get_outcomes(df: pd.DataFrame, mode: str) -> List[str]:
    """Extract outcome names from column names."""
    if mode == "disease":
        time_cols = [c for c in df.columns if c.endswith('_time_to_diagnosis')]
        return [c.replace('_time_to_diagnosis', '') for c in time_cols]
    else:  # cancer
        return [c for c in df.columns if c.endswith('cancer')]


def get_cancer_data_suffix(outcome: str) -> str:
    """Return '' for default, '_female' for female-specific, '_male' for male-specific cancer data."""
    key = outcome.lower().replace(" ", "_").strip()
    if key in FEMALE_SPECIFIC_CANCERS:
        return "_female"
    if key in MALE_SPECIFIC_CANCERS:
        return "_male"
    return ""


def encode_demo(X_train: pd.DataFrame, X_other: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train = pd.get_dummies(X_train, columns=CATEGORICAL_DEMO, dummy_na=True)
    X_train = X_train.reindex(sorted(X_train.columns), axis=1)
    X_other = pd.get_dummies(X_other, columns=CATEGORICAL_DEMO, dummy_na=True)
    X_other = X_other.reindex(columns=X_train.columns, fill_value=0)
    return X_train, X_other


def encode_and_align_to(X_other: pd.DataFrame, reference_columns: pd.Index) -> pd.DataFrame:
    """Encode categorical demo columns in X_other and align to reference_columns (e.g. from already-encoded train)."""
    X_enc = pd.get_dummies(X_other, columns=CATEGORICAL_DEMO, dummy_na=True)
    return X_enc.reindex(columns=reference_columns, fill_value=0)


def clean_future(df: pd.DataFrame, outcome: str, mode: str) -> pd.DataFrame:
    """Restrict to future cohort (not diagnosed at baseline), set _future and censor time."""
    df = df.loc[df[outcome] == 0].copy()
    df.loc[:, f"{outcome}_future"] = 0
    df.loc[df[f"{outcome}_time_to_diagnosis"] > 0, f"{outcome}_future"] = 1

    mask = df[f"{outcome}_future"] == 0
    if mode == "disease":
        df.loc[mask, f'{outcome}_time_to_diagnosis'] = 15.0
    else:  # cancer
        df.loc[mask, f'{outcome}_time_to_diagnosis'] = df.loc[mask, 'time_to_follow_up']
    return df


def sis(X, y, k=None, frac=None):
    is_df = isinstance(X, pd.DataFrame)
    colnames = X.columns if is_df else None
    X = np.asarray(X)
    y = np.asarray(y).astype(float)
    n, p = X.shape
    if k is None:
        k = p if frac is None else max(1, int(np.floor(frac * p)))
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    cov = np.sum(Xc * yc[:, None], axis=0)
    X_std = np.sqrt(np.sum(Xc**2, axis=0))
    y_std = np.sqrt(np.sum(yc**2))
    scores = np.abs(cov / (X_std * y_std + 1e-12))
    return {colnames[i]: scores[i] for i in range(len(colnames))}


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    olink_cols = [c for c in df.columns if c.startswith('olink_')]
    blood_cols = [c for c in df.columns if c.startswith('blood_')]
    demo_cols = [c for c in DEMO_COLS if c in df.columns]
    return olink_cols, blood_cols, demo_cols


def get_selected_features(
    df: pd.DataFrame, outcome: str, features: List[str], logger: logging.Logger, top_k: int = 500
) -> List[str]:
    X = df[features]
    y = df[f"{outcome}_future"]
    scores = sis(X, y)
    scores_sorted = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    sis_cols = [item[0] for item in scores_sorted[:top_k]]
    logger.info(f"Selected {len(sis_cols)} columns with largest SIS scores")
    return sis_cols


def get_feature_set(
    df_train: pd.DataFrame,
    outcome: str,
    feature_set_name: str,
    olink_cols: List[str],
    blood_cols: List[str],
    demo_cols: List[str],
    logger: logging.Logger,
) -> List[str]:
    if feature_set_name == 'demo_blood':
        return demo_cols + blood_cols
    sis_cols = get_selected_features(df_train, outcome, olink_cols, logger)
    if feature_set_name == 'demo_protein':
        return demo_cols + sis_cols
    if feature_set_name == 'demo_protein_blood':
        return demo_cols + sis_cols + blood_cols
    raise ValueError(f"Unknown feature set: {feature_set_name}")


def get_label_5y(df: pd.DataFrame, outcome: str) -> np.ndarray:
    """Binary label: event within 5 years (among those not diagnosed at baseline)."""
    future = df[f"{outcome}_future"].values
    time_col = df[f"{outcome}_time_to_diagnosis"].values
    return ((future == 1) & (time_col <= HORIZON_YEARS)).astype(np.float64)


def predict_risk_5y(rsf: RandomSurvivalForest, X: np.ndarray) -> np.ndarray:
    """Predict 5-year risk (1 - S(5)) from fitted RSF."""
    surv_funcs = rsf.predict_survival_function(X)
    risk_5y = np.array([1.0 - sf(HORIZON_YEARS) for sf in surv_funcs])
    return risk_5y


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_surv,
    y_test_surv,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    outcome: str,
    logger: logging.Logger,
) -> Tuple[RandomSurvivalForest, SimpleImputer, float, float, float]:
    """
    Impute, fit RSF, compute C-index and train/test AUC at 5 years.
    Returns: (model, imputer, c_index, train_auc_5y, test_auc_5y).
    """
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    logger.info(f"[{outcome}] Imputed median on all features.")

    logger.info("Training random survival forest...")
    rsf = RandomSurvivalForest(
        n_estimators=100,
        min_samples_split=10,
        min_samples_leaf=100,
        max_depth=10,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    t0 = time.time()
    rsf.fit(X_train_imp, y_train_surv)
    logger.info(f"[{outcome}] Fit RSF done in {(time.time() - t0) / 60:.1f}min")

    t1 = time.time()
    y_pred = rsf.predict(X_test_imp)
    logger.info(f"[{outcome}] Predicted in {(time.time() - t1):.1f}s")

    c_index = concordance_index_censored(
        y_test_surv[f"{outcome}_future"],
        y_test_surv[f"{outcome}_time_to_diagnosis"],
        y_pred,
    )[0]
    logger.info(f"[{outcome}] C-index = {c_index:.4f}")

    # 5-year AUC on train and test
    y_train_5y = get_label_5y(df_train, outcome)
    y_test_5y = get_label_5y(df_test, outcome)
    risk_train_5y = predict_risk_5y(rsf, X_train_imp)
    risk_test_5y = predict_risk_5y(rsf, X_test_imp)

    n_pos_train = int(y_train_5y.sum())
    n_pos_test = int(y_test_5y.sum())
    if n_pos_train >= 1 and n_pos_train < len(y_train_5y) and n_pos_test >= 1 and n_pos_test < len(y_test_5y):
        train_auc_5y = roc_auc_score(y_train_5y, risk_train_5y)
        test_auc_5y = roc_auc_score(y_test_5y, risk_test_5y)
        logger.info(f"[{outcome}] Train AUC (5y) = {train_auc_5y:.4f} | Test AUC (5y) = {test_auc_5y:.4f}")
    else:
        train_auc_5y = np.nan
        test_auc_5y = np.nan
        logger.info(f"[{outcome}] Skipping 5y AUC (insufficient positives: train={n_pos_train}, test={n_pos_test})")

    return rsf, imputer, c_index, train_auc_5y, test_auc_5y


def prepare_and_predict_5y(
    X_encoded: np.ndarray,
    eid: np.ndarray,
    imputer: SimpleImputer,
    rsf: RandomSurvivalForest,
) -> pd.DataFrame:
    """Predict 5y risk from encoded feature matrix. Returns DataFrame with eid and risk_5years."""
    X_imp = imputer.transform(X_encoded)
    risk_5y = predict_risk_5y(rsf, X_imp)
    return pd.DataFrame({"eid": eid, "risk_5years": risk_5y})


def write_risk_files(
    data_path: str,
    mode: str,
    outcome: str,
    feature_set: str,
    X_valid: Optional[np.ndarray],
    eid_valid: Optional[np.ndarray],
    X_test: np.ndarray,
    eid_test: np.ndarray,
    X_external: Optional[np.ndarray],
    eid_external: Optional[np.ndarray],
    imputer: SimpleImputer,
    rsf: RandomSurvivalForest,
    logger: logging.Logger,
) -> None:
    """Write validation, test, and (disease only) external risk at 5y to CSV."""
    if mode == "disease":
        base_dir = os.path.join(data_path, "disease_risk", "rsf")
    else:
        base_dir = os.path.join(data_path, "cancer_risk", "rsf")
    os.makedirs(base_dir, exist_ok=True)

    if X_valid is not None and eid_valid is not None and len(eid_valid) > 0:
        out = prepare_and_predict_5y(X_valid, eid_valid, imputer, rsf)
        fname = f"{outcome}_{feature_set}_rsf_risk_valid.csv"
        path = os.path.join(base_dir, fname)
        out.to_csv(path, index=False)
        logger.info(f"[{outcome}] Wrote {path}")

    if X_test is not None and eid_test is not None and len(eid_test) > 0:
        out = prepare_and_predict_5y(X_test, eid_test, imputer, rsf)
        fname = f"{outcome}_{feature_set}_rsf_risk_test.csv"
        path = os.path.join(base_dir, fname)
        out.to_csv(path, index=False)
        logger.info(f"[{outcome}] Wrote {path}")

    if mode == "disease" and X_external is not None and eid_external is not None and len(eid_external) > 0:
        out = prepare_and_predict_5y(X_external, eid_external, imputer, rsf)
        fname = f"{outcome}_{feature_set}_rsf_risk_external.csv"
        path = os.path.join(base_dir, fname)
        out.to_csv(path, index=False)
        logger.info(f"[{outcome}] Wrote external {path}")


# -----------------------
# Main
# -----------------------
def load_data(data_path: str, mode: str):
    """Load train, valid, test; external only for disease. For cancer, also return sex-specific sets."""
    if mode == "disease":
        df_train = pd.read_csv(f"{data_path}ukb_disease_train.csv", low_memory=False)
        df_valid = pd.read_csv(f"{data_path}ukb_disease_valid.csv", low_memory=False)
        df_test = pd.read_csv(f"{data_path}ukb_disease_test.csv", low_memory=False)
        df_external = pd.read_csv(f"{data_path}ukb_disease_scotland_wales.csv", low_memory=False)
        return df_train, df_valid, df_test, df_external, None

    # Cancer: load default plus female and male cohorts
    df_train = pd.read_csv(f"{data_path}ukb_cancer_train.csv", low_memory=False)
    df_valid = pd.read_csv(f"{data_path}ukb_cancer_valid.csv", low_memory=False)
    df_test = pd.read_csv(f"{data_path}ukb_cancer_test.csv", low_memory=False)
    df_train_f = pd.read_csv(f"{data_path}ukb_cancer_train_female.csv", low_memory=False)
    df_valid_f = pd.read_csv(f"{data_path}ukb_cancer_valid_female.csv", low_memory=False)
    df_test_f = pd.read_csv(f"{data_path}ukb_cancer_test_female.csv", low_memory=False)
    df_train_m = pd.read_csv(f"{data_path}ukb_cancer_train_male.csv", low_memory=False)
    df_valid_m = pd.read_csv(f"{data_path}ukb_cancer_valid_male.csv", low_memory=False)
    df_test_m = pd.read_csv(f"{data_path}ukb_cancer_test_male.csv", low_memory=False)
    cancer_sets = {
        "": (df_train, df_valid, df_test),
        "_female": (df_train_f, df_valid_f, df_test_f),
        "_male": (df_train_m, df_valid_m, df_test_m),
    }
    return df_train, df_valid, df_test, None, cancer_sets


def run_rsf(
    data_path: str,
    output_dir: str,
    mode: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    loaded = load_data(data_path, mode)
    if mode == "disease":
        df_train_base, df_valid_base, df_test_base, df_external_base, cancer_sets = loaded
    else:
        df_train_base, df_valid_base, df_test_base, df_external_base, cancer_sets = loaded
        assert cancer_sets is not None

    outcomes = get_outcomes(df_train_base, mode)
    olink_cols, blood_cols, demo_cols = get_feature_columns(df_train_base)

    logger.info(f"Mode={mode}, outcomes={len(outcomes)}, feature sets={FEATURE_SETS}")

    results = []
    for outcome in outcomes:
        # For cancer, use sex-specific data when applicable
        if mode == "cancer" and cancer_sets is not None:
            suffix = get_cancer_data_suffix(outcome)
            df_train_base, df_valid_base, df_test_base = cancer_sets[suffix]
            # Outcome must exist in this cohort (e.g. prostate not in female data)
            if outcome not in df_train_base.columns or f"{outcome}_time_to_diagnosis" not in df_train_base.columns:
                logger.info(f"Skipping {outcome} (not in {suffix or 'default'} cohort)")
                continue
            if suffix:
                logger.info(f"Using {suffix} cohort for {outcome}")

        for fs in FEATURE_SETS:
            logger.info("=" * 80)
            logger.info(f"Outcome: {outcome} | feature_set: {fs}")

            df_train = clean_future(df_train_base.copy(), outcome, mode)
            df_test = clean_future(df_test_base.copy(), outcome, mode)
            df_valid = clean_future(df_valid_base.copy(), outcome, mode)
            df_external = (
                clean_future(df_external_base.copy(), outcome, mode)
                if df_external_base is not None
                else None
            )

            feature_cols = get_feature_set(
                df_train, outcome, fs, olink_cols, blood_cols, demo_cols, logger
            )
            X_train = df_train[feature_cols]
            X_test = df_test[feature_cols]
            X_train, X_test = encode_demo(X_train, X_test)

            # Align valid (and external) with same columns for writing predictions later.
            # X_train is already encoded; only encode valid/external and align to X_train.columns.
            X_valid = encode_and_align_to(df_valid[feature_cols], X_train.columns)
            X_ext = None
            if df_external is not None and len(df_external) > 0:
                X_ext = encode_and_align_to(df_external[feature_cols], X_train.columns)

            logger.info(f"Using {len(feature_cols)} features ({fs})")

            y_train = Surv.from_dataframe(
                f"{outcome}_future", f"{outcome}_time_to_diagnosis", df_train
            )
            y_test = Surv.from_dataframe(
                f"{outcome}_future", f"{outcome}_time_to_diagnosis", df_test
            )

            rsf, imputer, c_index, train_auc_5y, test_auc_5y = train_and_evaluate(
                X_train.values,
                X_test.values,
                y_train,
                y_test,
                df_train,
                df_test,
                outcome,
                logger,
            )

            overall = pd.concat([df_train, df_test], ignore_index=True)
            overall_rate = overall[f"{outcome}_future"].mean()
            mean_event_time = overall.loc[
                overall[f"{outcome}_future"] == 1, f"{outcome}_time_to_diagnosis"
            ].mean()
            logger.info(
                f"[{outcome}] event_rate={overall_rate:.4f} | mean_event_time={mean_event_time:.3f}"
            )

            results.append({
                "outcome": outcome,
                "feature_set": fs,
                "c_index": c_index,
                "train_auc_5y": train_auc_5y,
                "test_auc_5y": test_auc_5y,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "event_rate": round(overall_rate, 4),
                "n_features": len(feature_cols),
            })

            # Write predicted probabilities at 5y for valid, test, and (disease) external
            write_risk_files(
                data_path,
                mode,
                outcome,
                fs,
                X_valid.values,
                df_valid["eid"].values,
                X_test.values,
                df_test["eid"].values,
                X_ext.values if X_ext is not None else None,
                df_external["eid"].values if df_external is not None and len(df_external) > 0 else None,
                imputer,
                rsf,
                logger,
            )

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="RSF for disease or cancer outcomes with 5y risk outputs."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["disease", "cancer"],
        help="Run for disease or cancer outcomes",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Data directory (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results CSV (default: results)",
    )
    args = parser.parse_args()

    data_path = args.data_path.rstrip("/") + "/"
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger("logs", name=f"rsf_{args.mode}")

    logger.info("=" * 60)
    logger.info(f"RSF run_rsf.py | mode={args.mode} | data_path={data_path}")
    logger.info("=" * 60)

    results_df = run_rsf(data_path, args.output_dir, args.mode, logger)

    out_csv = os.path.join(
        args.output_dir,
        f"rsf_{args.mode}_results.csv",
    )
    results_df.to_csv(out_csv, index=False)
    logger.info(f"Results saved to {out_csv}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
