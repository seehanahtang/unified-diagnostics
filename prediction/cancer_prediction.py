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
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from config import config, setup_directories, setup_logging
from data_loader import (
    load_cancer_datasets, load_tabtext_embeddings, preprocess, 
    get_feature_columns, merge_tabtext_embeddings, get_X,
    filter_cohort_by_time,
    filter_cohort_by_time_multiclass,
    get_y,
    get_y_multiclass
)
from models import train_model
from evaluator import (
    evaluate_model, evaluate_model_multiclass, get_feature_importance, plot_results,
    log_metrics, summarize_results,
    plot_calibration_curve
)

logger = logging.getLogger(__name__)


def run_cancer_prediction_pipeline(
    cancer_type: str,
    prediction_horizon: float = 1.0,
    model_type: str = "xgboost",
    use_tuning: bool = True,
    n_trials: int = 50,
    threshold_method: str = "youden",
    target_recall: float = 0.8,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run the complete cancer prediction pipeline for a single cancer type.
    
    Args:
        cancer_type: Type of cancer to predict (e.g., 'prostate_cancer')
        prediction_horizon: Time window in years for prediction
        model_type: Model to use ('xgboost' or 'lightgbm')
        use_tuning: Whether to use hyperparameter tuning
        n_trials: Number of Optuna trials if tuning
        threshold_method: 'f1', 'youden', 'target_recall', or 'percentile'
        target_recall: Target recall when using 'target_recall' method
        save_results: Whether to save plots and results
        
    Returns:
        Dictionary with results and metrics
    """
    logger.info("=" * 60)
    logger.info(f"Cancer Prediction Pipeline: {cancer_type}")
    logger.info(f"Prediction horizon: {prediction_horizon} years")
    logger.info(f"Model: {model_type}, Tuning: {use_tuning}")
    logger.info(f"Threshold method: {threshold_method}" + (f" (target={target_recall})" if threshold_method == 'target_recall' else ""))
    logger.info("=" * 60)
    
    # Load data
    train_valid_df, test_df = load_cancer_datasets(cancer_type, prediction_horizon)
    
    # Preprocess
    logger.info("Preprocessing data...")
    train_valid_df = preprocess(train_valid_df, onehot=True)
    test_df = preprocess(test_df, onehot=True)
    
    # Merge tabtext embeddings
    train_valid_df = merge_tabtext_embeddings(train_valid_df)
    test_df = merge_tabtext_embeddings(test_df)
    
    # Filter by time (exclude patients already diagnosed)
    train_valid_df = filter_cohort_by_time(train_valid_df, cancer_type)
    test_df = filter_cohort_by_time(test_df, cancer_type)

    y_train_valid = get_y(train_valid_df, cancer_type, prediction_horizon)
    train_df, valid_df = train_test_split(train_valid_df, test_size=0.15, random_state=config.random_state, stratify=y_train_valid)
    
    # Get feature columns
    feature_cols = get_feature_columns(
        train_df,
        use_olink=config.use_olink,
        use_blood=config.use_blood,
        use_demo=config.use_demo,
        use_tabtext=config.use_tabtext,
    )
    logger.info(f"Total features: {len(feature_cols)}")
    
    
    # Extract features and labels
    X_train = get_X(train_df, feature_cols)
    X_valid = get_X(valid_df, feature_cols)
    X_test = get_X(test_df, feature_cols)
    
    y_train = get_y(train_df, cancer_type, prediction_horizon)
    y_valid = get_y(valid_df, cancer_type, prediction_horizon)
    y_test = get_y(test_df, cancer_type, prediction_horizon)

    # Log class distribution for entire dataset (combined train + valid + test)
    n_train_positives = int(y_train.sum())
    logger.info(f"Train: {n_train_positives}, Valid: {y_valid.sum()}, Test: {y_test.sum()}")

    if n_train_positives < 10:
        logger.warning(f"Skipping {cancer_type}: fewer than 10 positives in training (n={n_train_positives})")
        return {
            'cancer_type': cancer_type,
            'model_type': model_type,
            'prediction_horizon': prediction_horizon,
            'n_train': len(y_train),
            'n_valid': len(y_valid),
            'n_test': len(y_test),
            'train_prevalence': y_train.mean(),
            'n_train_positives': n_train_positives,
            'n_features': len(feature_cols),
            'status': 'skipped',
            'skip_reason': f'fewer than 10 positives in training (n={n_train_positives})'
        }

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
        plot_path = f"{config.output_dir}{cancer_type}_{model_type}_results.png"
        plot_results(
            y_test, y_test_pred, feature_importance, 
            cancer_type, model_type, save_path=plot_path
        )
        
        # Save model
        model_filename = f"{config.models_dir}{model_type}_model_{prediction_horizon}yr_{cancer_type}"
        if model_type == 'xgboost':
            model.save_model(f"{model_filename}.json")
        else:
            model.booster_.save_model(f"{model_filename}.txt")
        logger.info(f"Saved model to {model_filename}")

        plot_calibration_curve(
            y_test, y_test_pred, n_bins=5,
            save_path=f"{config.output_dir}{cancer_type}_{model_type}_calibration.png"
        )
        logger.info(f"Saved calibration curve to {config.output_dir}{cancer_type}_{model_type}_calibration.png")
    
    # Compile results
    results = {
        'cancer_type': cancer_type,
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


def run_multiclass_pipeline(
    prediction_horizon: float = 1.0,
    model_type: str = "xgboost",
    use_tuning: bool = True,
    n_trials: int = 50,
    threshold_method: str = "target_recall",
    target_recall: float = 0.8,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run multiclass cancer type prediction pipeline.
    
    Predicts which type of cancer (or other/none) a patient will develop.
    
    Args:
        prediction_horizon: Time window in years for prediction
        model_type: Model to use ('xgboost' or 'lightgbm')
        use_tuning: Whether to use hyperparameter tuning
        n_trials: Number of Optuna trials if tuning
        save_results: Whether to save plots and results
        
    Returns:
        Dictionary with results and metrics
    """
    logger.info("=" * 60)
    logger.info(f"Cancer Prediction Pipeline: Multiclass")
    logger.info(f"Prediction horizon: {prediction_horizon} years")
    logger.info(f"Model: {model_type}, Tuning: {use_tuning}")
    logger.info(f"Threshold method: {threshold_method}" + (f" (target={target_recall})" if threshold_method == 'target_recall' else ""))
    logger.info("=" * 60)
    
    # Load data
    train_valid_df, test_df = load_cancer_datasets()
    
    # Preprocess
    logger.info("Preprocessing data...")
    train_valid_df = preprocess(train_valid_df, onehot=True)
    test_df = preprocess(test_df, onehot=True)
    
    # Merge tabtext embeddings
    train_valid_df = merge_tabtext_embeddings(train_valid_df)
    test_df = merge_tabtext_embeddings(test_df)
    
    # Filter by time (exclude patients already diagnosed)
    train_valid_df = filter_cohort_by_time_multiclass(train_valid_df, config.cancer_types)
    test_df = filter_cohort_by_time_multiclass(test_df, config.cancer_types)
    
    
    # Get feature columns
    feature_cols = get_feature_columns(
        train_valid_df,
        use_olink=config.use_olink,
        use_blood=config.use_blood,
        use_demo=config.use_demo,
        use_tabtext=config.use_tabtext,
    )   
    logger.info(f"Total features: {len(feature_cols)}")

    train_valid_df, class_mapping = get_y_multiclass(train_valid_df, config.cancer_types, prediction_horizon)
    test_df, class_mapping = get_y_multiclass(test_df, config.cancer_types, prediction_horizon)

    train_df, valid_df = train_test_split(train_valid_df, test_size=0.15, random_state=config.random_state, stratify=train_valid_df["outcome"])
    
    
    # Extract features and labels
    X_train = get_X(train_df, feature_cols)
    X_valid = get_X(valid_df, feature_cols)
    X_test = get_X(test_df, feature_cols)
    
    y_train = train_df["outcome"]
    y_valid = valid_df["outcome"]
    y_test = test_df["outcome"]

    # Log class distribution for entire dataset (combined train + valid + test)
    logger.info(f"Train: {y_train.sum()}, Valid: {y_valid.sum()}, Test: {y_test.sum()}")
    
    # Train model
    model, best_params = train_model(
        X_train, y_train, X_valid, y_valid,
        model_type=model_type,
        use_tuning=use_tuning,
        n_trials=n_trials,
        num_classes=len(config.cancer_types)+1
    )
    
    # Evaluate on validation set
    y_valid_pred = model.predict_proba(X_valid)
    valid_metrics = evaluate_model_multiclass(y_valid, y_valid_pred, class_mapping, prefix='valid_')
    
    logger.info("\nValidation Set Results:")
    log_metrics(valid_metrics, prefix='valid_', multiclass=True)
    
    # Evaluate on test set
    y_test_pred = model.predict_proba(X_test)
    test_metrics = evaluate_model_multiclass(y_test, y_test_pred, class_mapping, prefix='test_')
    
    logger.info("\nTest Set Results:")
    log_metrics(test_metrics, prefix='test_', multiclass=True)
    
    # Feature importance
    feature_importance = get_feature_importance(model, feature_cols)
    if not feature_importance.empty:
        logger.info("\nTop 10 Features:")
        for i, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    if save_results:
        model_filename = f"{config.models_dir}{model_type}_multiclass_{prediction_horizon}yr"
        if model_type == 'xgboost':
            model.save_model(f"{model_filename}.json")
        else:
            model.booster_.save_model(f"{model_filename}.txt")
        logger.info(f"Saved model to {model_filename}")

        plot_calibration_curve(
            y_test > 0, 1 - y_test_pred[:, 0], n_bins=5,
            save_path=f"{config.output_dir}cancer_{model_type}_calibration.png"
        )
        logger.info(f"Saved calibration curve to {config.output_dir}cancer_{model_type}_calibration.png")
    
    
    # Compile results
    results = {
        'classification_mode': 'multiclass',
        'model_type': model_type,
        'prediction_horizon': prediction_horizon,
        'num_classes': len(config.cancer_types)+1,
        'class_mapping': str(class_mapping),
        'n_train': len(y_train),
        'n_valid': len(y_valid),
        'n_test': len(y_test),
        'n_features': len(feature_cols),
        **valid_metrics,
        **test_metrics,
        'status': 'completed'
    }
    
    if best_params:
        results['best_params'] = str(best_params)
    
    return results


def run_multi_cancer_pipeline(
    cancer_types: Optional[List[str]] = None,
    prediction_horizon: float = 1.0,
    model_type: str = "xgboost",
    use_tuning: bool = True,
    n_trials: int = 50,
    threshold_method: str = "target_recall",
    target_recall: float = 0.8
) -> pd.DataFrame:
    """
    Run prediction pipeline for multiple cancer types.
    
    Args:
        cancer_types: List of cancer types to predict (None = all)
        prediction_horizon: Time window in years
        model_type: Model to use
        use_tuning: Whether to use hyperparameter tuning
        n_trials: Number of Optuna trials
        threshold_method: Threshold selection method
        target_recall: Target recall for 'target_recall' method
        
    Returns:
        DataFrame with results for all cancer types
    """
    if cancer_types is None:
        cancer_types = config.cancer_types
    
    all_results = []
    
    for cancer_type in cancer_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {cancer_type}")
        logger.info(f"{'='*60}\n")
        
        try:
            result = run_cancer_prediction_pipeline(
                cancer_type=cancer_type,
                prediction_horizon=prediction_horizon,
                model_type=model_type,
                use_tuning=use_tuning,
                n_trials=n_trials,
                threshold_method=threshold_method,
                target_recall=target_recall
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error processing {cancer_type}: {e}")
            all_results.append({
                'cancer_type': cancer_type,
                'status': 'error',
                'error': str(e)
            })
    
    # Summarize results
    summary_df = summarize_results(
        all_results, 
        output_file=f"{config.output_dir}cancer_prediction_summary.csv"
    )
    
    return summary_df


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Cancer Prediction Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--prediction_horizon', '-p',
        type=float,
        default=1.0,
        help='Prediction horizon in years (e.g., 1.0 for 1-year prediction)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['xgboost', 'lightgbm'],
        default='xgboost',
        help='Model type to use'
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
        '--threshold-method', '-t',
        type=str,
        choices=['f1', 'youden', 'target_recall', 'percentile'],
        default='youden',
        help='Threshold selection method. Use target_recall to maximize recall.'
    )
    
    parser.add_argument(
        '--target-recall',
        type=float,
        default=0.8,
        help='Target recall level when using --threshold-method target_recall'
    )
    
    parser.add_argument(
        '--multiclass',
        action='store_true',
        help='Use multiclass classification (predict cancer type) instead of binary'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU for model training (requires CUDA)'
    )
    
    args = parser.parse_args()
    
    # Update config
    config.output_dir = args.output_dir
    config.use_tabtext = not args.no_tabtext
    config.classification_mode = 'multiclass' if args.multiclass else 'binary'
    config.use_gpu = args.gpu
    
    # Setup directories and logging with model/horizon in filename
    setup_directories()
    setup_logging(model_type=args.model, prediction_horizon=args.prediction_horizon)
    
    # Set random seeds for reproducibility
    import os
    os.environ['PYTHONHASHSEED'] = str(config.random_state)
    random.seed(config.random_state)
    np.random.seed(config.random_state)
    
    # Set seeds for XGBoost/LightGBM determinism (if using GPU)
    if config.use_gpu:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Load tabtext embeddings if enabled
    load_tabtext_embeddings()
    
    logger.info("Cancer Prediction Pipeline Started")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Classification mode: {config.classification_mode}")
    logger.info(f"Cancer types: {config.cancer_types}")
    
    if args.multiclass:
        # Run multiclass pipeline
        result = run_multiclass_pipeline(
            prediction_horizon=args.prediction_horizon,
            model_type=args.model,
            use_tuning=not args.no_tuning,
            n_trials=args.n_trials
        )
        logger.info(f"\nFinal Result: {result}")
    else:
        # Run binary pipeline for all cancer types
        results = run_multi_cancer_pipeline(
            cancer_types=None,  # Uses config.cancer_types
            prediction_horizon=args.prediction_horizon,
            model_type=args.model,
            use_tuning=not args.no_tuning,
            n_trials=args.n_trials,
            threshold_method=args.threshold_method,
            target_recall=args.target_recall
        )
        logger.info("\nFinal Summary:")
        logger.info(results.to_string())
    
    logger.info("\nPipeline completed!")


if __name__ == '__main__':
    main()
