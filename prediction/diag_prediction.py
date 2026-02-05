"""
Cancer Prediction Pipeline
===========================
Main entry point for running cancer prediction models on UK Biobank data.

Usage:
    python run_prediction.py --prediction_horizon 1.0
    python run_prediction.py --model lightgbm --no-tuning
    python run_prediction.py 
"""

import argparse
import logging
import random
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

from config import config, setup_directories, setup_logging
from data_loader import (
    load_datasets, load_tabtext_embeddings, preprocess, 
    get_feature_columns, merge_tabtext_embeddings, get_X,
    filter_cohort_by_time, filter_by_sex, filter_high_missingness,
    get_y, get_y_multiclass
)
from models import train_model
from evaluator import (
    evaluate_model, evaluate_model_multiclass, get_feature_importance, plot_results,
    log_metrics, summarize_results,
    plot_calibration_curve
)

logger = logging.getLogger(__name__)


def run_diag_prediction_pipeline(
    train_valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    diag_type: str,
    prediction_horizon: float = 1.0
) -> pd.DataFrame:
    """
    Run prediction pipeline for single diagnosis type.
    
    Args:
        diag_type: Diagnosis type to predict
        prediction_horizon: Time window in years
        
    Returns:
        DataFrame with results for single diagnosis type
    """
    logger.info("=" * 60)
    logger.info(f"Diagnosis Prediction Pipeline: {diag_type}")
    logger.info(f"Prediction horizon: {prediction_horizon} years")
    logger.info("=" * 60)
    
    # Preprocess
    logger.info("Preprocessing data...")
    train_valid_df = preprocess(train_valid_df, onehot=True)
    test_df = preprocess(test_df, onehot=True)
    
    # Merge tabtext embeddings
    train_valid_df = merge_tabtext_embeddings(train_valid_df)
    test_df = merge_tabtext_embeddings(test_df)
    
    # Filter by time (exclude patients already diagnosed)
    train_valid_df = filter_cohort_by_time(train_valid_df, diag_type)
    test_df = filter_cohort_by_time(test_df, diag_type)
    
    # Get feature columns
    feature_cols = get_feature_columns(
        train_df,
        use_olink=config.use_olink,
        use_blood=config.use_blood,
        use_demo=config.use_demo,
        use_tabtext=config.use_tabtext,
        diag_type=diag_type
    )
    logger.info(f"Total features: {len(feature_cols)}")
    
    train_df, valid_df = train_test_split(train_valid_df, test_size=0.15, random_state=config.random_state, stratify=get_y(train_valid_df, diag_type, prediction_horizon))
    
    # Extract features and labels
    X_train = get_X(train_df, feature_cols)
    X_valid = get_X(valid_df, feature_cols)
    X_test = get_X(test_df, feature_cols)
    
    y_train = get_y(train_df, diag_type, prediction_horizon)
    y_valid = get_y(valid_df, diag_type, prediction_horizon)
    y_test = get_y(test_df, diag_type, prediction_horizon)

    # Log class distribution for entire dataset (combined train + valid + test)
    logger.info(f"Train: {y_train.sum()}, Valid: {y_valid.sum()}, Test: {y_test.sum()}")
    
    # Train model
    model, best_params = train_model(
        X_train, y_train, X_valid, y_valid,
        model_type=model_type,
        use_tuning=use_tuning,
        n_trials=n_trials
    )
    
    # Evaluate on validation set
    y_valid_pred = model.predict_proba(X_valid)[:, 1]
    valid_metrics = evaluate_model(
        y_valid, y_valid_pred,
        threshold_method=threshold_method,
        target_recall=target_recall,
        prefix='valid_'
    )
    
    logger.info("\nValidation Set Results:")
    log_metrics(valid_metrics, prefix='valid_')
    
    # Evaluate on test set
    y_test_pred = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_model(
        y_test, y_test_pred, 
        threshold=valid_metrics['valid_threshold'],  # Use threshold from validation
        prefix='test_'
    )
    
    logger.info("\nTest Set Results:")
    log_metrics(test_metrics, prefix='test_')
    
    # Feature importance
    feature_importance = get_feature_importance(model, feature_cols)
    if not feature_importance.empty:
        logger.info("\nTop 10 Features:")
        for i, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Plot results
    if save_results:
        plot_path = f"{config.output_dir}{diag_type}_{model_type}_results.png"
        plot_results(
            y_test, y_test_pred, feature_importance, 
            diag_type, model_type, save_path=plot_path
        )
        
        # Save model
        model_filename = f"{config.models_dir}{model_type}_model_{prediction_horizon}yr_{diag_type}"
        if model_type == 'xgboost':
            model.save_model(f"{model_filename}.json")
        else:
            model.booster_.save_model(f"{model_filename}.txt")
        logger.info(f"Saved model to {model_filename}")

        plot_calibration_curve(
            y_test, y_test_pred, n_bins=5,
            save_path=f"{config.output_dir}{diag_type}_{model_type}_calibration.png"
        )
        logger.info(f"Saved calibration curve to {config.output_dir}{diag_type}_{model_type}_calibration.png")
    
    # Compile results
    results = {
        'diag_type': diag_type,
        'model_type': model_type,
        'prediction_horizon': prediction_horizon,
        'n_train': len(y_train),
        'n_valid': len(y_valid),
        'n_test': len(y_test),
        'train_prevalence': y_train.mean(),
        'test_prevalence': y_test.mean(),
        'n_features': len(feature_cols),
        **valid_metrics,
        **test_metrics,
        'status': 'completed'
    }
    
    if best_params:
        results['best_params'] = str(best_params)
    
    return results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Diagnosis Prediction Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--prediction_horizon', '-p',
        type=float,
        default=1.0,
        help='Prediction horizon in years (e.g., 1.0 for 1-year prediction)'
    )
    
    parser.add_argument(
        '--no-tuning',
        action='store_true',
        help='Disable hyperparameter tuning (use default parameters)'
    )
    
    parser.add_argument(
        '--n-trials', '-n',
        type=int,
        default=50,
        help='Number of Optuna trials for hyperparameter tuning'
    )
    
    parser.add_argument(
        '--no-tabtext',
        action='store_true',
        help='Disable tabtext embeddings'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='out/',
        help='Output directory for results and plots'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU for model training (requires CUDA)'
    )
    
    args = parser.parse_args()
    
    # Update config
    config.output_dir = args.output_dir
    config.use_gpu = args.gpu
    
    # Setup directories and logging with model/horizon in filename
    setup_directories()
    setup_logging(prediction_horizon=args.prediction_horizon)
    
    # Set random seeds for reproducibility
    import os
    os.environ['PYTHONHASHSEED'] = str(config.random_state)
    random.seed(config.random_state)
    np.random.seed(config.random_state)
    
    # Set seeds for XGBoost/LightGBM determinism (if using GPU)
    if config.use_gpu:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    load_tabtext_embeddings()
    
    logger.info("Diagnosis Prediction Pipeline Started")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Diagnosis types: {config.diag_types}")

    train_valid_df, test_df = load_diag_datasets()
    
    result = run_diag_prediction_pipeline(  
        train_valid_df=train_valid_df,
        test_df=test_df,
        prediction_horizon=args.prediction_horizon,
        diag_type=args.diag_type
    )
    logger.info(f"\nFinal Result: {result}")
    logger.info("\nPipeline completed!")

if __name__ == '__main__':
    main()
