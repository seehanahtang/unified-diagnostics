"""
Ensemble Cancer Prediction Pipeline
====================================
Uses probabilities from individual cancer type models as features
for a meta-model that predicts cancer (any type) vs no cancer.

Usage:
    python run_prediction_ensemble.py --prediction_horizon 1.0
    python run_prediction_ensemble.py --model xgboost --no-tuning
"""

import argparse
import logging
import random
import os
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb

from config import config, setup_directories, setup_logging
from data_loader import (
    load_datasets, load_tabtext_embeddings, preprocess, filter_other_cancers,
    get_feature_columns, merge_tabtext_embeddings, get_X,
    filter_cohort_by_time, filter_by_sex, filter_high_missingness,
    get_y
)
from models import train_model
from evaluator import (
    evaluate_model, get_feature_importance, plot_results,
    log_metrics, summarize_results
)

logger = logging.getLogger(__name__)


def load_cancer_model(
    cancer_type: str,
    model_type: str = "xgboost",
    prediction_horizon: float = 1.0
):
    """
    Load a pre-trained cancer model.
    
    Args:
        cancer_type: Type of cancer model to load
        model_type: 'xgboost' or 'lightgbm'
        prediction_horizon: Prediction horizon used during training
        
    Returns:
        Loaded model or None if not found
    """
    if model_type == 'xgboost':
        model_path = f"models/{model_type}_model_{prediction_horizon}yr_{cancer_type}.json"
        if os.path.exists(model_path):
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            logger.info(f"Loaded model: {model_path}")
            return model
    else:
        model_path = f"models/{model_type}_model_{prediction_horizon}yr_{cancer_type}.txt"
        if os.path.exists(model_path):
            model = lgb.Booster(model_file=model_path)
            logger.info(f"Loaded model: {model_path}")
            return model
    
    logger.warning(f"Model not found: {model_path}")
    return None


def get_cancer_probabilities(
    df: pd.DataFrame,
    feature_cols: List[str],
    cancer_types: List[str],
    model_type: str = "xgboost",
    prediction_horizon: float = 1.0
) -> pd.DataFrame:
    """
    Get probability predictions from each cancer-specific model.
    
    Args:
        df: Input dataframe with features
        feature_cols: List of feature columns
        cancer_types: List of cancer types to get predictions for
        model_type: Model type to load
        prediction_horizon: Prediction horizon
        
    Returns:
        DataFrame with probability columns for each cancer type
    """
    prob_df = pd.DataFrame(index=df.index)
    
    for cancer_type in cancer_types:
        model = load_cancer_model(cancer_type, model_type, prediction_horizon)
        
        if model is None:
            logger.warning(f"Skipping {cancer_type} - model not found")
            prob_df[f'{cancer_type}_prob'] = np.nan
            continue
        
        # Get features for this cancer type (apply sex filtering logic)
        X = get_X(df, feature_cols)
        
        # Get probability predictions
        if model_type == 'xgboost':
            probs = model.predict_proba(X)[:, 1]
        else:
            probs = model.predict(X)
        
        prob_df[f'{cancer_type}_prob'] = probs
        logger.info(f"Generated probabilities for {cancer_type}: mean={probs.mean():.4f}, std={probs.std():.4f}")
    
    return prob_df


def run_ensemble_pipeline(
    prediction_horizon: float = 1.0,
    model_type: str = "xgboost",
    use_tuning: bool = True,
    n_trials: int = 50,
    threshold_method: str = "youden",
    target_recall: float = 0.8,
    include_base_features: bool = False,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run ensemble cancer prediction pipeline.
    
    Uses probabilities from individual cancer models as features to predict
    whether a patient will develop any cancer.
    
    Args:
        prediction_horizon: Time window in years for prediction
        model_type: Model to use for ensemble ('xgboost' or 'lightgbm')
        use_tuning: Whether to use hyperparameter tuning
        n_trials: Number of Optuna trials if tuning
        threshold_method: Threshold selection method
        target_recall: Target recall for 'target_recall' method
        include_base_features: Whether to include original features in addition to probabilities
        save_results: Whether to save results
        
    Returns:
        Dictionary with results and metrics
    """
    logger.info("=" * 60)
    logger.info("Ensemble Cancer Prediction Pipeline")
    logger.info(f"Cancer types: {config.cancer_types}")
    logger.info(f"Prediction horizon: {prediction_horizon} years")
    logger.info(f"Model: {model_type}, Tuning: {use_tuning}")
    logger.info(f"Include base features: {include_base_features}")
    logger.info(f"Threshold method: {threshold_method}")
    logger.info("=" * 60)
    
    # Load data
    train_df, valid_df, test_df = load_datasets()
    
    # Preprocess
    logger.info("Preprocessing data...")
    train_df = preprocess(train_df, onehot=True)
    valid_df = preprocess(valid_df, onehot=True)
    test_df = preprocess(test_df, onehot=True)
    
    # Merge tabtext embeddings
    train_df = merge_tabtext_embeddings(train_df)
    valid_df = merge_tabtext_embeddings(valid_df)
    test_df = merge_tabtext_embeddings(test_df)
    
    # Filter by time (exclude patients already diagnosed with any cancer)
    train_df = filter_cohort_by_time(train_df, "cancer", start_year=0.0)
    valid_df = filter_cohort_by_time(valid_df, "cancer", start_year=0.0)
    test_df = filter_cohort_by_time(test_df, "cancer", start_year=0.0)
    
    # Filter out patients with cancers not in our cancer_types
    train_df = filter_other_cancers(train_df, prediction_horizon)
    valid_df = filter_other_cancers(valid_df, prediction_horizon)
    test_df = filter_other_cancers(test_df, prediction_horizon)
    
    # Get feature columns for base models
    feature_cols = get_feature_columns(
        train_df,
        use_olink=config.use_olink,
        use_blood=config.use_blood,
        use_demo=config.use_demo,
        use_tabtext=config.use_tabtext,
        cancer_type=None
    )
    logger.info(f"Base model features: {len(feature_cols)}")
    
    # Generate probability features from each cancer model
    logger.info("\nGenerating probability features from cancer-specific models...")
    train_probs = get_cancer_probabilities(
        train_df, feature_cols, config.cancer_types, model_type, prediction_horizon
    )
    valid_probs = get_cancer_probabilities(
        valid_df, feature_cols, config.cancer_types, model_type, prediction_horizon
    )
    test_probs = get_cancer_probabilities(
        test_df, feature_cols, config.cancer_types, model_type, prediction_horizon
    )
    
    # Build ensemble feature set
    if include_base_features:
        # Combine probability features with base features
        X_train = pd.concat([get_X(train_df, feature_cols), train_probs], axis=1)
        X_valid = pd.concat([get_X(valid_df, feature_cols), valid_probs], axis=1)
        X_test = pd.concat([get_X(test_df, feature_cols), test_probs], axis=1)
        ensemble_features = feature_cols + list(train_probs.columns)
    else:
        # Use only probability features
        X_train = train_probs
        X_valid = valid_probs
        X_test = test_probs
        ensemble_features = list(train_probs.columns)
    
    logger.info(f"Ensemble features: {len(ensemble_features)}")
    logger.info(f"Probability features: {list(train_probs.columns)}")
    
    # Get binary labels: any cancer within prediction window
    # Use the general cancer column
    y_train = get_y(train_df, "cancer", prediction_horizon)
    y_valid = get_y(valid_df, "cancer", prediction_horizon)
    y_test = get_y(test_df, "cancer", prediction_horizon)

    # Filter patients with >50% missing values
    X_train, y_train = filter_high_missingness(X_train, y_train, threshold=0.5)
    X_valid, y_valid = filter_high_missingness(X_valid, y_valid, threshold=0.5)
    X_test, y_test = filter_high_missingness(X_test, y_test, threshold=0.5)
    
    # Train ensemble model
    logger.info("\nTraining ensemble model...")
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
        threshold=valid_metrics['valid_threshold'],
        prefix='test_'
    )
    
    logger.info("\nTest Set Results:")
    log_metrics(test_metrics, prefix='test_')
    
    # Feature importance (probability features)
    feature_importance = get_feature_importance(model, ensemble_features)
    if not feature_importance.empty:
        logger.info("\nFeature Importance (Ensemble):")
        for i, row in feature_importance.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save results
    if save_results:
        # Save model
        model_filename = f"{config.models_dir}{model_type}_ensemble_{prediction_horizon}yr"
        if model_type == 'xgboost':
            model.save_model(f"{model_filename}.json")
        else:
            model.booster_.save_model(f"{model_filename}.txt")
        logger.info(f"Saved ensemble model to {model_filename}")
        
        # Save feature importance
        if not feature_importance.empty:
            fi_path = f"{config.models_dir}feature_importance_ensemble.csv"
            feature_importance.to_csv(fi_path, index=False)
            logger.info(f"Saved feature importance to {fi_path}")
        
        # Plot results
        plot_path = f"{config.output_dir}ensemble_{model_type}_results.png"
        plot_results(
            y_test, y_test_pred, feature_importance,
            "ensemble", model_type, save_path=plot_path
        )
    
    # Compile results
    results = {
        'classification_mode': 'ensemble',
        'model_type': model_type,
        'prediction_horizon': prediction_horizon,
        'cancer_types': str(config.cancer_types),
        'include_base_features': include_base_features,
        'n_train': len(y_train),
        'n_valid': len(y_valid),
        'n_test': len(y_test),
        'train_prevalence': y_train.mean(),
        'test_prevalence': y_test.mean(),
        'n_features': len(ensemble_features),
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
        description='Ensemble Cancer Prediction Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--prediction_horizon', '-p',
        type=float,
        default=1.0,
        help='Prediction horizon in years'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['xgboost', 'lightgbm'],
        default='xgboost',
        help='Model type to use for ensemble'
    )
    
    parser.add_argument(
        '--no-tuning',
        action='store_true',
        help='Disable hyperparameter tuning'
    )
    
    parser.add_argument(
        '--n-trials', '-n',
        type=int,
        default=50,
        help='Number of Optuna trials for hyperparameter tuning'
    )
    
    parser.add_argument(
        '--threshold-method', '-t',
        type=str,
        choices=['f1', 'youden', 'target_recall', 'percentile'],
        default='youden',
        help='Threshold selection method'
    )
    
    parser.add_argument(
        '--target-recall',
        type=float,
        default=0.8,
        help='Target recall level when using --threshold-method target_recall'
    )
    
    parser.add_argument(
        '--include-base-features',
        action='store_true',
        help='Include base features in addition to probability features'
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
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU for model training'
    )
    
    args = parser.parse_args()
    
    # Update config
    config.output_dir = args.output_dir
    config.use_tabtext = not args.no_tabtext
    config.use_gpu = args.gpu
    
    # Setup directories and logging
    setup_directories()
    setup_logging(model_type=f"ensemble_{args.model}", prediction_horizon=args.prediction_horizon)
    
    # Set random seeds
    os.environ['PYTHONHASHSEED'] = str(config.random_state)
    random.seed(config.random_state)
    np.random.seed(config.random_state)
    
    if config.use_gpu:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Load tabtext embeddings if enabled
    load_tabtext_embeddings()
    
    logger.info("Ensemble Cancer Prediction Pipeline Started")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Cancer types: {config.cancer_types}")
    
    # Run ensemble pipeline
    results = run_ensemble_pipeline(
        prediction_horizon=args.prediction_horizon,
        model_type=args.model,
        use_tuning=not args.no_tuning,
        n_trials=args.n_trials,
        threshold_method=args.threshold_method,
        target_recall=args.target_recall,
        include_base_features=args.include_base_features,
        save_results=True
    )
    
    logger.info(f"\nFinal Results: {results}")
    logger.info("\nEnsemble Pipeline completed!")


if __name__ == '__main__':
    main()
