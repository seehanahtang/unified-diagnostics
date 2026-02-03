"""
Simple K-Fold Cross-Validation for Cancer Prediction using xgb.cv()
====================================================================
Runs XGBoost with built-in CV to quickly evaluate AUC performance.

Usage:
    python run_prediction_kfold.py
    python run_prediction_kfold.py --prediction_horizon 2.0
"""

import argparse
import logging
import random
import sys
from typing import Dict

import pandas as pd
import numpy as np
import xgboost as xgb

from config import config, setup_directories
from data_loader import (
    load_tabtext_embeddings, preprocess, 
    get_feature_columns, merge_tabtext_embeddings, get_X,
    filter_cohort_by_time, filter_by_sex, get_y
)

# Setup logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def load_data_for_cv():
    """Load and combine train+valid for CV, keep test separate."""
    logger.info("Loading datasets...")
    train_df = pd.read_csv(f'{config.data_dir}ukb_diag_train.csv')
    test_df = pd.read_csv(f'{config.data_dir}ukb_diag_test.csv')
    
    return train_df, test_df


def run_kfold_cv(
    cancer_type: str,
    prediction_horizon: float = 1.0,
    nfold: int = 3,
    num_boost_round: int = 50,
    early_stopping_rounds: int = 10
) -> Dict:
    """
    Run xgb.cv() for a single cancer type.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {nfold}-fold CV for: {cancer_type}")
    logger.info(f"Prediction horizon: {prediction_horizon} years")
    logger.info(f"{'='*60}")
    
    # Load and preprocess data
    train_df, test_df = load_data_for_cv()
    
    train_df = preprocess(train_df, onehot=True)
    test_df = preprocess(test_df, onehot=True)
    
    train_df = merge_tabtext_embeddings(train_df)
    test_df = merge_tabtext_embeddings(test_df)
    
    train_df = filter_by_sex(train_df, cancer_type)
    test_df = filter_by_sex(test_df, cancer_type)
    
    train_df = filter_cohort_by_time(train_df, cancer_type, start_year=0.0)
    test_df = filter_cohort_by_time(test_df, cancer_type, start_year=0.0)
    
    # Get features and labels
    feature_cols = get_feature_columns(
        train_df,
        use_olink=config.use_olink,
        use_blood=config.use_blood,
        use_demo=config.use_demo,
        use_tabtext=config.use_tabtext,
        cancer_type=cancer_type
    )
    logger.info(f"Features: {len(feature_cols)}")
    
    X = get_X(train_df, feature_cols)
    y = get_y(train_df, cancer_type, prediction_horizon)
    X_test = get_X(test_df, feature_cols)
    y_test = get_y(test_df, cancer_type, prediction_horizon)
    
    # Check for sufficient samples
    if y.sum() < nfold:
        logger.warning(f"Insufficient positive samples ({y.sum()}), skipping")
        return {'cancer_type': cancer_type, 'status': 'skipped'}
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
    
    # XGBoost parameters
    n_neg, n_pos = (y == 0).sum(), (y == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'device': 'cuda' if config.use_gpu else 'cpu',
        'scale_pos_weight': scale_pos_weight,
        'seed': config.random_state,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'verbosity': 0,
    }
    
    # Run xgb.cv
    logger.info(f"Running xgb.cv with nfold={nfold}, num_boost_round={num_boost_round}")
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        nfold=nfold,
        metrics='auc',
        early_stopping_rounds=early_stopping_rounds,
        seed=config.random_state,
        stratified=True,
        verbose_eval=True
    )
    
    # Get best CV AUC
    best_idx = cv_results['test-auc-mean'].idxmax()
    cv_auc_mean = cv_results.loc[best_idx, 'test-auc-mean']
    cv_auc_std = cv_results.loc[best_idx, 'test-auc-std']
    best_round = best_idx + 1
    
    logger.info(f"Best CV AUC: {cv_auc_mean:.4f} (+/- {cv_auc_std:.4f}) at round {best_round}")
    
    # Train final model and evaluate on test set
    logger.info("Training final model on all data...")
    final_model = xgb.train(params, dtrain, num_boost_round=best_round)
    
    y_test_pred = final_model.predict(dtest)
    from sklearn.metrics import roc_auc_score
    test_auc = roc_auc_score(y_test, y_test_pred)
    
    logger.info(f"Test AUC: {test_auc:.4f}")
    
    return {
        'cancer_type': cancer_type,
        'n_samples': len(y),
        'n_positive': int(y.sum()),
        'prevalence': y.mean(),
        'cv_auc_mean': cv_auc_mean,
        'cv_auc_std': cv_auc_std,
        'best_round': best_round,
        'test_auc': test_auc,
        'status': 'completed'
    }


def main():
    parser = argparse.ArgumentParser(description='Simple K-Fold CV using xgb.cv()')
    parser.add_argument('--prediction_horizon', '-p', type=float, default=1.0)
    parser.add_argument('--nfold', '-k', type=int, default=3)
    parser.add_argument('--num_boost_round', '-n', type=int, default=50)
    parser.add_argument('--early_stopping_rounds', '-e', type=int, default=10)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--cancer-type', '-c', type=str, default=None,
                        help='Single cancer type to run (default: all)')
    args = parser.parse_args()
    
    config.use_gpu = args.gpu
    setup_directories()
    
    random.seed(config.random_state)
    np.random.seed(config.random_state)
    
    load_tabtext_embeddings()
    
    logger.info(f"Running {args.nfold}-fold CV, horizon={args.prediction_horizon}yr")
    
    cancer_types = [args.cancer_type] if args.cancer_type else config.cancer_types
    
    all_results = []
    for cancer_type in cancer_types:
        try:
            result = run_kfold_cv(
                cancer_type, 
                args.prediction_horizon, 
                args.nfold,
                args.num_boost_round,
                args.early_stopping_rounds
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error with {cancer_type}: {e}")
            all_results.append({'cancer_type': cancer_type, 'status': 'error', 'error': str(e)})
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    
    summary_df = pd.DataFrame(all_results)
    completed = summary_df[summary_df['status'] == 'completed']
    
    if not completed.empty:
        for _, row in completed.iterrows():
            logger.info(f"{row['cancer_type']:20s} CV AUC: {row['cv_auc_mean']:.4f} (+/-{row['cv_auc_std']:.4f})  Test AUC: {row['test_auc']:.4f}")
        
        summary_df.to_csv(f"{config.output_dir}kfold_cv_summary.csv", index=False)
        logger.info(f"\nSaved summary to {config.output_dir}kfold_cv_summary.csv")
    
    logger.info("\nDone!")


if __name__ == '__main__':
    main()
