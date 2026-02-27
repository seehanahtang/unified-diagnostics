"""
Model Training
==============
XGBoost and LightGBM training functions with optional hyperparameter tuning.
"""

import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score, log_loss

from config import config

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================================
# XGBoost Training Functions
# ============================================================================

def objective_xgb(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    scale_pos_weight: float,
    num_classes: int = 2
) -> float:
    """Optuna objective for XGBoost hyperparameter optimization."""
    
    if num_classes == 2:
        objective = 'binary:logistic'
        eval_metric = 'auc'
    else:
        objective = 'multi:softprob'
        eval_metric = 'mlogloss'
    param = {
        'objective': objective,
        'eval_metric': eval_metric,
        'tree_method': 'hist',
        'device': 'cuda' if config.use_gpu else 'cpu',
        'enable_categorical': True,
        # 'scale_pos_weight': scale_pos_weight,
        'random_state': config.random_state,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    if num_classes > 2:
        param['num_class'] = num_classes

    model = xgb.XGBClassifier(**param, early_stopping_rounds=50, verbosity=0)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    
    if num_classes == 2:
        preds = model.predict_proba(X_valid)[:, 1]
        score = roc_auc_score(y_valid, preds)
    else:
        preds = model.predict_proba(X_valid)
        # Use negative log loss for multiclass (since we maximize, and lower log_loss is better)
        # This handles missing classes in validation set, unlike roc_auc_score
        score = -log_loss(y_valid, preds, labels=list(range(num_classes)))
    
    return score


def train_xgboost_with_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    n_trials: int = 50,
    num_classes: int = 2
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    Train XGBoost with Optuna hyperparameter tuning.
    
    Returns:
        Tuple of (trained model, best parameters)
    """
    # Calculate class weight for imbalance (binary only)
    n_neg = (y_train == 0).sum()
    n_pos = (y_train != 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    if num_classes == 2:
        logger.info(f"Class imbalance: {n_neg} neg / {n_pos} pos (weight: {scale_pos_weight:.2f})")
    else:
        logger.info(f"Multiclass with {num_classes} classes")
    logger.info(f"Starting Optuna optimization with {n_trials} trials...")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=config.random_state)
    )
    study.optimize(
        lambda trial: objective_xgb(
            trial, X_train, y_train, X_valid, y_valid, scale_pos_weight, num_classes
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    best_params = study.best_params
    best_params['tree_method'] = 'hist'
    best_params['device'] = 'cuda' if config.use_gpu else 'cpu'
    best_params['enable_categorical'] = True
    best_params['random_state'] = config.random_state
    
    if num_classes == 2:
        best_params['scale_pos_weight'] = scale_pos_weight
        best_params['objective'] = 'binary:logistic'
        best_params['eval_metric'] = 'auc'
    else:
        best_params['objective'] = 'multi:softprob'
        best_params['eval_metric'] = 'mlogloss'
        best_params['num_class'] = num_classes
    
    if num_classes == 2:
        logger.info(f"Best AUC: {study.best_value:.4f}")
    else:
        logger.info(f"Best neg log loss: {study.best_value:.4f}")
    logger.info(f"Best params: {best_params}")
    
    # Train final model with best parameters
    final_model = xgb.XGBClassifier(**best_params, early_stopping_rounds=50, verbosity=0)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    
    return final_model, best_params


def train_xgboost_simple(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    num_classes: int = 2
) -> xgb.XGBClassifier:
    """
    Train XGBoost without hyperparameter tuning.
    
    Args:
        X_train, y_train: Training data
        X_valid, y_valid: Validation data
        params: Optional custom parameters
        num_classes: Number of classes (2 for binary)
        
    Returns:
        Trained XGBoost model
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train != 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    if num_classes == 2:
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': scale_pos_weight,
        }
    else:
        default_params = {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'num_class': num_classes,
        }
    
    default_params.update({
        'tree_method': 'hist',
        'device': 'cuda' if config.use_gpu else 'cpu',
        'enable_categorical': True,
        'random_state': config.random_state,
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
    })
    
    if params:
        default_params.update(params)

    if num_classes > 2:
        default_params = {
            'n_estimators': 292, 
            'max_depth': 4, 
            'learning_rate': 0.013667399886421027, 
            'min_child_weight': 9, 
            'subsample': 0.6428809539923007, 
            'colsample_bytree': 0.8342083403558622, 
            'gamma': 1.7538028439369104, 
            'reg_alpha': 1.0220250584337675e-08, 
            'reg_lambda': 0.055746430880450844, 
            'tree_method': 'hist', 
            'device': 'cuda', 
            'enable_categorical': True, 
            'random_state': 42, 
            'objective': 'multi:softprob', 
            'eval_metric': 'mlogloss', 
            'num_class': 8
        }
    
    model = xgb.XGBClassifier(**default_params, verbosity=0)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    use_tuning: bool = True,
    n_trials: int = 50,
    num_classes: int = 2
) -> Tuple[xgb.XGBClassifier, Optional[Dict[str, Any]]]:
    """
    Unified XGBoost training interface.
    
    Args:
        X_train, y_train: Training data
        X_valid, y_valid: Validation data
        use_tuning: Whether to use Optuna tuning
        n_trials: Number of Optuna trials
        num_classes: Number of classes (2 for binary)
        
    Returns:
        Tuple of (trained model, best params or None)
    """
    if use_tuning:
        return train_xgboost_with_tuning(X_train, y_train, X_valid, y_valid, n_trials, num_classes)
    else:
        model = train_xgboost_simple(X_train, y_train, X_valid, y_valid, num_classes=num_classes)
        return model, None


# ============================================================================
# LightGBM Training Functions
# ============================================================================

def objective_lgb(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    scale_pos_weight: float,
    num_classes: int = 2
) -> float:
    """Optuna objective for LightGBM hyperparameter optimization."""
    
    if num_classes == 2:
        objective = 'binary'
        metric = 'auc'
    else:
        objective = 'multiclass'
        metric = 'multi_logloss'
    
    param = {
        'objective': objective,
        'metric': metric,
        'boosting_type': 'gbdt',
        'device': 'gpu' if config.use_gpu else 'cpu',
        'random_state': config.random_state,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'verbosity': -1,
    }
    
    if num_classes == 2:
        param['scale_pos_weight'] = scale_pos_weight
    else:
        param['num_class'] = num_classes
    
    model = lgb.LGBMClassifier(**param)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)]
    )
    
    if num_classes == 2:
        preds = model.predict_proba(X_valid)[:, 1]
        score = roc_auc_score(y_valid, preds)
    else:
        preds = model.predict_proba(X_valid)
        # Use negative log loss for multiclass (since we maximize, and lower log_loss is better)
        # This handles missing classes in validation set, unlike roc_auc_score
        score = -log_loss(y_valid, preds, labels=list(range(num_classes)))
    
    return score


def train_lightgbm_with_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    n_trials: int = 50,
    num_classes: int = 2
) -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
    """
    Train LightGBM with Optuna hyperparameter tuning.
    
    Returns:
        Tuple of (trained model, best parameters)
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train != 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    if num_classes == 2:
        logger.info(f"Class imbalance: {n_neg} neg / {n_pos} pos (weight: {scale_pos_weight:.2f})")
    else:
        logger.info(f"Multiclass with {num_classes} classes")
    logger.info(f"Starting LightGBM Optuna optimization with {n_trials} trials...")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=config.random_state)
    )
    study.optimize(
        lambda trial: objective_lgb(
            trial, X_train, y_train, X_valid, y_valid, scale_pos_weight, num_classes
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    best_params = study.best_params
    best_params['boosting_type'] = 'gbdt'
    best_params['device'] = 'gpu' if config.use_gpu else 'cpu'
    best_params['random_state'] = config.random_state
    best_params['verbosity'] = -1
    
    if num_classes == 2:
        best_params['scale_pos_weight'] = scale_pos_weight
        best_params['objective'] = 'binary'
        best_params['metric'] = 'auc'
    else:
        best_params['objective'] = 'multiclass'
        best_params['metric'] = 'multi_logloss'
        best_params['num_class'] = num_classes
    best_params['verbosity'] = -1
    
    if num_classes == 2:
        logger.info(f"Best AUC: {study.best_value:.4f}")
    else:
        logger.info(f"Best neg log loss: {study.best_value:.4f}")
    logger.info(f"Best params: {best_params}")
    
    # Train final model with best parameters
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)]
    )
    
    return final_model, best_params


def train_lightgbm_simple(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    num_classes: int = 2
) -> lgb.LGBMClassifier:
    """
    Train LightGBM without hyperparameter tuning.
    
    Args:
        X_train, y_train: Training data
        X_valid, y_valid: Validation data
        params: Optional custom parameters
        num_classes: Number of classes (2 for binary)
        
    Returns:
        Trained LightGBM model
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train != 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    if num_classes == 2:
        default_params = {
            'objective': 'binary',
            'metric': 'auc',
            'scale_pos_weight': scale_pos_weight,
        }
    else:
        default_params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'num_class': num_classes,
        }
    
    default_params.update({
        'boosting_type': 'gbdt',
        'device': 'gpu' if config.use_gpu else 'cpu',
        'random_state': config.random_state,
        'n_estimators': 500,
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.05,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbosity': -1,
    })
    
    if params:
        default_params.update(params)
    
    model = lgb.LGBMClassifier(**default_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)]
    )
    
    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    use_tuning: bool = True,
    n_trials: int = 50,
    num_classes: int = 2
) -> Tuple[lgb.LGBMClassifier, Optional[Dict[str, Any]]]:
    """
    Unified LightGBM training interface.
    
    Args:
        X_train, y_train: Training data
        X_valid, y_valid: Validation data
        use_tuning: Whether to use Optuna tuning
        n_trials: Number of Optuna trials
        num_classes: Number of classes (2 for binary)
        
    Returns:
        Tuple of (trained model, best params or None)
    """
    if use_tuning:
        return train_lightgbm_with_tuning(X_train, y_train, X_valid, y_valid, n_trials, num_classes)
    else:
        model = train_lightgbm_simple(X_train, y_train, X_valid, y_valid, num_classes=num_classes)
        return model, None


# ============================================================================
# Model Training Dispatcher
# ============================================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    model_type: str = "xgboost",
    use_tuning: bool = True,
    n_trials: int = 50,
    num_classes: int = 2
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Train a model using the specified algorithm.
    
    Args:
        X_train, y_train: Training data
        X_valid, y_valid: Validation data
        model_type: "xgboost" or "lightgbm"
        use_tuning: Whether to use Optuna tuning
        n_trials: Number of Optuna trials
        num_classes: Number of classes (2 for binary, >2 for multiclass)
        
    Returns:
        Tuple of (trained model, best params or None)
    """
    model_type = model_type.lower()
    
    if model_type == "xgboost":
        logger.info(f"Training XGBoost model ({'multiclass' if num_classes > 2 else 'binary'})...")
        return train_xgboost(X_train, y_train, X_valid, y_valid, use_tuning, n_trials, num_classes)
    elif model_type == "lightgbm":
        logger.info(f"Training LightGBM model ({'multiclass' if num_classes > 2 else 'binary'})...")
        return train_lightgbm(X_train, y_train, X_valid, y_valid, use_tuning, n_trials, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'xgboost' or 'lightgbm'")
