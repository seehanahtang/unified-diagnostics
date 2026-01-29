"""
Cancer Prediction Pipeline - UK Biobank
========================================
Predicts various cancer diagnoses within the next year using XGBoost with:
- Comprehensive hyperparameter tuning via Optuna
- Multiple cancer types with sex-specific filtering
- Class imbalance handling
- Optimal threshold selection
- Comprehensive evaluation metrics
- Feature importance analysis
"""

import argparse
import numpy as np
import xgboost as xgb
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Try to import optuna for hyperparameter tuning
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not installed. Running without hyperparameter tuning.")
    print("Install with: pip install optuna")


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class Config:
    """Configuration for cancer prediction pipeline."""
    # Directories
    data_dir: str = "/../../orcd/pool/003/dbertsim_shared/ukb/"
    models_dir: str = "out/"
    results_dir: str = "results/"
    log_file: str = "out/cancer_prediction.log"
    
    # Feature flags
    use_olink: bool = True
    use_blood: bool = True
    use_demo: bool = True
    use_tabtext: bool = True  # Use tabtext embeddings
    use_family_history: bool = True
    
    # Prediction window (years)
    prediction_window: float = 1.0  # Predict cancer within next year
    min_followup: float = 0.0  # Minimum time before prediction window
    
    # Cancer types to predict
    cancer_types: List[str] = field(default_factory=lambda: [
        "breast",       # Breast cancer (female only)
        "prostate",     # Prostate cancer (male only)
        "lung",         # Lung cancer
        "colorectal",   # Colorectal cancer
        "bladder",      # Bladder cancer
        "pancreatic",   # Pancreatic cancer
    ])
    
    # Hyperparameter tuning
    n_optuna_trials: int = 50
    cv_folds: int = 5
    random_state: int = 42
    
    # Training parameters
    num_boost_rounds: int = 1000
    early_stopping_rounds: int = 50
    
    # Class imbalance handling
    use_scale_pos_weight: bool = True
    
    # Model selection
    model_type: str = "xgboost"  # "xgboost" or "lightgbm"
    
    # Threshold optimization
    optimize_threshold: bool = True
    threshold_metric: str = "f1"  # "f1", "precision", "recall", "youden"


config = Config()

# Create output directories first (before logging setup)
os.makedirs(config.models_dir, exist_ok=True)
os.makedirs(config.results_dir, exist_ok=True)

# Configure logging (after directories exist)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load tabtext embeddings
demo_embeddings = None
embedding_cols = []
if config.use_tabtext:
    try:
        demo_embeddings = pd.read_csv(f"{config.data_dir}demo_embeddings.csv")
        embedding_cols = [col for col in demo_embeddings.columns if col.startswith("cn_")]
        logger.info(f"Loaded {len(embedding_cols)} tabtext embedding columns")
    except FileNotFoundError:
        logger.warning("demo_embeddings.csv not found. Tabtext features disabled.")
        config.use_tabtext = False


# =============================================================================
# Feature Engineering
# =============================================================================
DEMO_FEATURES = [
    'Age at recruitment', 
    'Sex_male',
    'Ethnic background',
    'Body mass index (BMI)', 
    'Systolic blood pressure, automated reading',
    'Diastolic blood pressure, automated reading',
    'Townsend deprivation index at recruitment',
    'Smoking status', 
    'Alcohol intake frequency.',
    'Medication for cholesterol, blood pressure or diabetes'
]

# Sex-specific cancers
FEMALE_ONLY_CANCERS = ["breast"]
MALE_ONLY_CANCERS = ["prostate"]


def preprocess(df: pd.DataFrame, onehot: bool = False) -> pd.DataFrame:
    """
    Preprocess the dataframe with improved handling.
    
    Args:
        df: Input dataframe
        onehot: Whether to one-hot encode family illness history
        
    Returns:
        Preprocessed dataframe
    """
    df = df.copy()
    
    # Convert categorical columns
    categorical_cols = [
        'Ethnic background',
        'Smoking status', 
        'Alcohol intake frequency.',
        'Medication for cholesterol, blood pressure or diabetes'
    ]
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    
    # Handle missing values for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isna().sum() > 0:
            # Use median imputation for numerical features
            df[col] = df[col].fillna(df[col].median())
    
    # Create derived features
    if 'Age at recruitment' in df.columns:
        df['Age_squared'] = df['Age at recruitment'] ** 2
        df['Age_group'] = pd.cut(
            df['Age at recruitment'], 
            bins=[0, 50, 60, 70, 100],
            labels=['<50', '50-60', '60-70', '70+']
        ).astype('category')
    
    if 'Body mass index (BMI)' in df.columns:
        df['BMI_category'] = pd.cut(
            df['Body mass index (BMI)'],
            bins=[0, 18.5, 25, 30, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        ).astype('category')
    
    # Process family history with one-hot encoding
    if onehot:
        family_cols = ['Illnesses of father', 'Illnesses of mother', 'Illnesses of siblings']
        for col in family_cols:
            if col not in df.columns:
                continue
                
            # Clean the column
            df[col] = df[col].fillna('')
            df[col] = (df[col]
                .str.replace(r'None of the above \(group [12]\)', '', regex=True)
                .str.replace(r'Do not know \(group [12]\)', '', regex=True)
                .str.replace(r'Prefer not to answer \(group [12]\)', '', regex=True)
                .str.strip())

            # One hot encoding
            illness_dummies = df[col].str.get_dummies(sep='|')
            if len(illness_dummies.columns) > 0:
                illness_dummies.columns = col + '_' + illness_dummies.columns.str.strip()
                
                # Create family cancer history feature
                cancer_related = [c for c in illness_dummies.columns 
                                 if 'cancer' in c.lower()]
                if cancer_related:
                    df[f'{col}_has_cancer'] = illness_dummies[cancer_related].max(axis=1)

                # Insert dummy columns
                ill_idx = df.columns.get_loc(col)
                df.drop(columns=[col], inplace=True)

                for i, dummy_col in enumerate(illness_dummies.columns):
                    df.insert(ill_idx + i, dummy_col, illness_dummies[dummy_col])
    
    return df


def get_feature_columns(
    df: pd.DataFrame,
    use_olink: bool = True,
    use_blood: bool = True,
    use_demo: bool = True,
    use_tabtext: bool = True,
    cancer_type: Optional[str] = None
) -> List[str]:
    """
    Get feature columns based on configuration.
    
    Args:
        df: Input dataframe
        use_olink: Include Olink proteomics features
        use_blood: Include blood biomarker features
        use_demo: Include demographic features
        use_tabtext: Include tabtext embedding features
        cancer_type: Specific cancer type for cancer-specific features
        
    Returns:
        List of feature column names
    """
    features = []
    
    if use_tabtext and embedding_cols:
        # Tabtext embedding columns (already merged into df)
        available_embedding_cols = [col for col in embedding_cols if col in df.columns]
        features.extend(available_embedding_cols)
        if available_embedding_cols:
            logger.info(f"Added {len(available_embedding_cols)} tabtext embedding features")
    
    if use_olink:
        olink_cols = [col for col in df.columns if col.startswith("olink_")]
        features.extend(olink_cols)
        if olink_cols:
            logger.info(f"Added {len(olink_cols)} Olink features")
    
    if use_blood:
        blood_cols = [col for col in df.columns if col.startswith("blood_")]
        features.extend(blood_cols)
        if blood_cols:
            logger.info(f"Added {len(blood_cols)} blood biomarker features")
    
    if use_demo:
        demo_cols = [col for col in DEMO_FEATURES if col in df.columns]
        features.extend(demo_cols)
        # Add derived features
        derived = ['Age_squared', 'Age_group', 'BMI_category']
        features.extend([col for col in derived if col in df.columns])
        logger.info(f"Added {len(demo_cols)} demographic features")
    
    
    # Remove duplicates while preserving order
    features = list(dict.fromkeys(features))
    
    return features


def merge_tabtext_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge tabtext embeddings into dataframe based on 'eid' column.
    
    Args:
        df: Input dataframe with 'eid' column
        
    Returns:
        Dataframe with embedding columns merged
    """
    if demo_embeddings is None or not embedding_cols:
        return df
    
    if 'eid' not in df.columns:
        logger.warning("'eid' column not found. Cannot merge tabtext embeddings.")
        return df
    
    # Merge embeddings on 'eid' column
    df = df.merge(
        demo_embeddings[['eid'] + embedding_cols], 
        on='eid', 
        how='left'
    )
    logger.info(f"Merged {len(embedding_cols)} tabtext embedding columns")
    
    return df


def get_X(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """Extract feature matrix from dataframe."""
    available_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = set(feature_cols) - set(available_cols)
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
    return df[available_cols].copy()


def filter_cohort_by_time(
    df: pd.DataFrame, 
    cancer_type: str, 
    start_year: float = 0.0
) -> pd.DataFrame:
    """
    Filter cohort by time to diagnosis.
    Excludes patients who were diagnosed before the start year.
    
    Args:
        df: Input dataframe
        cancer_type: Cancer type column prefix
        start_year: Minimum years before diagnosis to include
        
    Returns:
        Filtered dataframe
    """
    time_col = f'{cancer_type}_time_to_diagnosis'
    if time_col not in df.columns:
        logger.warning(f"Time column {time_col} not found")
        return df
        
    # Include if: no diagnosis (NaN) OR diagnosis is after start_year
    mask = (df[time_col] > start_year) | (df[time_col].isna())
    filtered_df = df.loc[mask].copy()
    
    logger.info(f"Filtered {cancer_type}: {len(df)} -> {len(filtered_df)} samples")
    return filtered_df


def filter_by_sex(
    df: pd.DataFrame, 
    cancer_type: str
) -> pd.DataFrame:
    """Filter by sex for sex-specific cancers."""
    if cancer_type in FEMALE_ONLY_CANCERS:
        if 'Sex_male' in df.columns:
            df = df[df['Sex_male'] == 0].copy()
            logger.info(f"Filtered to females only for {cancer_type}")
    elif cancer_type in MALE_ONLY_CANCERS:
        if 'Sex_male' in df.columns:
            df = df[df['Sex_male'] == 1].copy()
            logger.info(f"Filtered to males only for {cancer_type}")
    return df


def get_y(
    df: pd.DataFrame, 
    cancer_type: str,
    prediction_window: float = 1.0
) -> pd.Series:
    """
    Get binary labels for cancer prediction within prediction window.
    
    Args:
        df: Input dataframe
        cancer_type: Cancer type column prefix
        prediction_window: Years within which to predict cancer
        
    Returns:
        Binary labels (1 if diagnosed within window, 0 otherwise)
    """
    time_col = f"{cancer_type}_time_to_diagnosis"
    if time_col not in df.columns:
        raise ValueError(f"Column {time_col} not found")
    
    # Positive if diagnosed within the prediction window
    y = ((df[time_col] > 0) & (df[time_col] <= prediction_window)).astype(int)
    
    n_positive = y.sum()
    n_total = len(y)
    prevalence = n_positive / n_total if n_total > 0 else 0
    
    logger.info(f"{cancer_type}: {n_positive}/{n_total} positive ({prevalence:.4f})")
    
    return y


# =============================================================================
# Model Training with Hyperparameter Optimization
# =============================================================================
def objective(
    trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    scale_pos_weight: float = 1.0
) -> float:
    """Optuna objective function for hyperparameter tuning."""
    
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "aucpr"],
        "tree_method": "hist",
        "random_state": config.random_state,
        
        # Hyperparameters to tune
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": scale_pos_weight if config.use_scale_pos_weight else 1.0,
    }
    
    dtrain = xgb.DMatrix(data=X_train, label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(data=X_valid, label=y_valid, enable_categorical=True)
    
    evals = [(dtrain, "train"), (dvalid, "valid")]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=config.num_boost_rounds,
        evals=evals,
        early_stopping_rounds=config.early_stopping_rounds,
        verbose_eval=False
    )
    
    # Use validation AUC-PR as objective (better for imbalanced data)
    y_pred = model.predict(dvalid)
    auc_pr = average_precision_score(y_valid, y_pred)
    
    return auc_pr


def train_xgboost_with_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cancer_type: str,
    n_trials: int = 50
) -> Tuple[xgb.Booster, Dict]:
    """
    Train XGBoost with Optuna hyperparameter optimization.
    
    Returns:
        Tuple of (trained model, best parameters)
    """
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not available, using default parameters")
        return train_xgboost_simple_wrapper(X_train, y_train, X_valid, y_valid)
    
    # Calculate scale_pos_weight for class imbalance
    n_positive = y_train.sum()
    n_negative = len(y_train) - n_positive
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
    
    logger.info(f"Training {cancer_type} model with scale_pos_weight={scale_pos_weight:.2f}")
    
    # Optuna study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=config.random_state)
    )
    
    study.optimize(
        lambda trial: objective(
            trial, X_train, y_train, X_valid, y_valid, scale_pos_weight
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    logger.info(f"Best trial AUC-PR: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")
    
    # Train final model with best parameters
    best_params = study.best_params.copy()
    best_params.update({
        "objective": "binary:logistic",
        "eval_metric": ["auc", "aucpr"],
        "tree_method": "hist",
        "random_state": config.random_state,
        "scale_pos_weight": scale_pos_weight if config.use_scale_pos_weight else 1.0,
    })
    
    dtrain = xgb.DMatrix(data=X_train, label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(data=X_valid, label=y_valid, enable_categorical=True)
    
    evals = [(dtrain, "train"), (dvalid, "valid")]
    
    model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=config.num_boost_rounds,
        evals=evals,
        early_stopping_rounds=config.early_stopping_rounds,
        verbose_eval=10
    )
    
    return model, best_params


def train_xgboost_simple(
    dtrain: xgb.DMatrix,
    dvalid: xgb.DMatrix,
    scale_pos_weight: float = 1.0,
    num_epochs: int = 500
) -> xgb.Booster:
    """Simple XGBoost training without hyperparameter tuning."""
    
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "aucpr"],
        "tree_method": "hist",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight if config.use_scale_pos_weight else 1.0,
        "random_state": config.random_state,
    }

    evals = [(dtrain, "train"), (dvalid, "valid")]

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_epochs,
        evals=evals,
        early_stopping_rounds=config.early_stopping_rounds,
        verbose_eval=20
    )
    return model


def train_xgboost_simple_wrapper(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series
) -> Tuple[xgb.Booster, Dict]:
    """Wrapper for simple training that returns params too."""
    n_positive = y_train.sum()
    n_negative = len(y_train) - n_positive
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
    
    dtrain = xgb.DMatrix(data=X_train, label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(data=X_valid, label=y_valid, enable_categorical=True)
    
    model = train_xgboost_simple(dtrain, dvalid, scale_pos_weight)
    
    params = {
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    
    return model, params


# =============================================================================
# LightGBM Training Functions
# =============================================================================
def objective_lgb(
    trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    scale_pos_weight: float = 1.0
) -> float:
    """Optuna objective function for LightGBM hyperparameter tuning."""
    
    params = {
        "objective": "binary",
        "metric": ["auc", "average_precision"],
        "boosting_type": "gbdt",
        "random_state": config.random_state,
        "verbosity": -1,
        
        # Hyperparameters to tune
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": scale_pos_weight if config.use_scale_pos_weight else 1.0,
    }
    
    # Convert categorical columns
    categorical_features = X_train.select_dtypes(include=['category']).columns.tolist()
    
    dtrain = lgb.Dataset(
        X_train, label=y_train,
        categorical_feature=categorical_features if categorical_features else 'auto'
    )
    dvalid = lgb.Dataset(
        X_valid, label=y_valid,
        categorical_feature=categorical_features if categorical_features else 'auto',
        reference=dtrain
    )
    
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=config.num_boost_rounds,
        valid_sets=[dtrain, dvalid],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=config.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )
    
    # Use validation AUC-PR as objective
    y_pred = model.predict(X_valid)
    auc_pr = average_precision_score(y_valid, y_pred)
    
    return auc_pr


def train_lightgbm_with_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cancer_type: str,
    n_trials: int = 50
) -> Tuple[lgb.Booster, Dict]:
    """
    Train LightGBM with Optuna hyperparameter optimization.
    
    Returns:
        Tuple of (trained model, best parameters)
    """
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not available, using default parameters")
        return train_lightgbm_simple_wrapper(X_train, y_train, X_valid, y_valid)
    
    # Calculate scale_pos_weight for class imbalance
    n_positive = y_train.sum()
    n_negative = len(y_train) - n_positive
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
    
    logger.info(f"Training {cancer_type} LightGBM model with scale_pos_weight={scale_pos_weight:.2f}")
    
    # Optuna study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=config.random_state)
    )
    
    study.optimize(
        lambda trial: objective_lgb(
            trial, X_train, y_train, X_valid, y_valid, scale_pos_weight
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    logger.info(f"Best trial AUC-PR: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")
    
    # Train final model with best parameters
    best_params = study.best_params.copy()
    best_params.update({
        "objective": "binary",
        "metric": ["auc", "average_precision"],
        "boosting_type": "gbdt",
        "random_state": config.random_state,
        "verbosity": -1,
        "scale_pos_weight": scale_pos_weight if config.use_scale_pos_weight else 1.0,
    })
    
    categorical_features = X_train.select_dtypes(include=['category']).columns.tolist()
    
    dtrain = lgb.Dataset(
        X_train, label=y_train,
        categorical_feature=categorical_features if categorical_features else 'auto'
    )
    dvalid = lgb.Dataset(
        X_valid, label=y_valid,
        categorical_feature=categorical_features if categorical_features else 'auto',
        reference=dtrain
    )
    
    model = lgb.train(
        best_params,
        dtrain,
        num_boost_round=config.num_boost_rounds,
        valid_sets=[dtrain, dvalid],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=config.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=10)
        ]
    )
    
    return model, best_params


def train_lightgbm_simple(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    scale_pos_weight: float = 1.0
) -> lgb.Booster:
    """Simple LightGBM training without hyperparameter tuning."""
    
    params = {
        "objective": "binary",
        "metric": ["auc", "average_precision"],
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight if config.use_scale_pos_weight else 1.0,
        "random_state": config.random_state,
        "verbosity": -1,
    }
    
    categorical_features = X_train.select_dtypes(include=['category']).columns.tolist()
    
    dtrain = lgb.Dataset(
        X_train, label=y_train,
        categorical_feature=categorical_features if categorical_features else 'auto'
    )
    dvalid = lgb.Dataset(
        X_valid, label=y_valid,
        categorical_feature=categorical_features if categorical_features else 'auto',
        reference=dtrain
    )
    
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dtrain, dvalid],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=config.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=20)
        ]
    )
    return model


def train_lightgbm_simple_wrapper(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series
) -> Tuple[lgb.Booster, Dict]:
    """Wrapper for simple LightGBM training that returns params too."""
    n_positive = y_train.sum()
    n_negative = len(y_train) - n_positive
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
    
    model = train_lightgbm_simple(X_train, y_train, X_valid, y_valid, scale_pos_weight)
    
    params = {
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
    }
    
    return model, params


# =============================================================================
# Evaluation Metrics
# =============================================================================
def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "recall"
) -> float:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        metric: Metric to optimize ("f1", "precision", "recall", "youden")
        
    Returns:
        Optimal threshold
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    
    if metric == "f1":
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
    elif metric == "youden":
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        return roc_thresholds[best_idx]
    elif metric == "precision":
        # Find threshold for 80% precision
        target_precision = 0.8
        valid_idx = np.where(precision >= target_precision)[0]
        best_idx = valid_idx[0] if len(valid_idx) > 0 else 0
    elif metric == "recall":
        # Find threshold for 80% recall
        target_recall = 0.8
        valid_idx = np.where(recall >= target_recall)[0]
        best_idx = valid_idx[-1] if len(valid_idx) > 0 else len(thresholds) - 1
    else:
        best_idx = len(thresholds) // 2
    
    return thresholds[min(best_idx, len(thresholds) - 1)]


def evaluate_model(
    model: Union[xgb.Booster, lgb.Booster],
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
    dataset_name: str = "test",
    model_type: str = "xgboost"
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Returns:
        Dictionary of evaluation metrics
    """
    if model_type == "xgboost":
        dmatrix = xgb.DMatrix(data=X, label=y, enable_categorical=True)
        y_pred_proba = model.predict(dmatrix)
    else:  # lightgbm
        y_pred_proba = model.predict(X)
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        "dataset": dataset_name,
        "n_samples": len(y),
        "n_positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "threshold": threshold,
        
        # Probability metrics
        "auc_roc": roc_auc_score(y, y_pred_proba),
        "auc_pr": average_precision_score(y, y_pred_proba),
        
        # Classification metrics at threshold
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        
        # Confusion matrix
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }
    
    # Calculate specificity and NPV
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Number needed to screen
    if metrics["precision"] > 0:
        metrics["nns"] = 1 / metrics["precision"]
    else:
        metrics["nns"] = float('inf')
    
    return metrics


def get_feature_importance(
    model: Union[xgb.Booster, lgb.Booster],
    feature_names: List[str],
    top_n: int = 20,
    model_type: str = "xgboost"
) -> pd.DataFrame:
    """Get feature importance from trained model."""
    if model_type == "xgboost":
        importance = model.get_score(importance_type='gain')
        
        # Map feature indices to names
        importance_list = []
        for k, v in importance.items():
            if k.startswith('f'):
                idx = int(k[1:])
                if idx < len(feature_names):
                    importance_list.append({"feature": feature_names[idx], "importance": v})
            else:
                importance_list.append({"feature": k, "importance": v})
    else:  # lightgbm
        importance_values = model.feature_importance(importance_type='gain')
        importance_list = [
            {"feature": name, "importance": imp}
            for name, imp in zip(feature_names, importance_values)
        ]
    
    importance_df = pd.DataFrame(importance_list)
    
    if len(importance_df) > 0:
        importance_df = importance_df.sort_values("importance", ascending=False)
    return importance_df.head(top_n)


def plot_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cancer_type: str,
    output_path: str
):
    """Create evaluation plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    axes[0].plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'{cancer_type} - ROC Curve')
    axes[0].legend()
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auc_pr = average_precision_score(y_true, y_pred)
    axes[1].plot(recall, precision, label=f'AUC-PR = {auc_pr:.3f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'{cancer_type} - Precision-Recall Curve')
    axes[1].legend()
    
    # Score distribution
    axes[2].hist(y_pred[y_true == 0], bins=50, alpha=0.5, label='Negative', density=True)
    axes[2].hist(y_pred[y_true == 1], bins=50, alpha=0.5, label='Positive', density=True)
    axes[2].set_xlabel('Predicted Probability')
    axes[2].set_ylabel('Density')
    axes[2].set_title(f'{cancer_type} - Score Distribution')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Training Pipeline
# =============================================================================
def run_cancer_prediction_pipeline(use_hyperparameter_tuning: bool = True):
    """Main pipeline for training cancer prediction models."""
    
    logger.info("=" * 60)
    logger.info("Starting Cancer Prediction Pipeline")
    logger.info(f"Prediction window: {config.prediction_window} year(s)")
    logger.info("=" * 60)
    
    # Load data
    logger.info("Loading datasets...")
    train_df = pd.read_csv(f'{config.data_dir}ukb_cancer_train.csv')
    valid_df = pd.read_csv(f'{config.data_dir}ukb_cancer_valid.csv')
    test_df = pd.read_csv(f'{config.data_dir}ukb_cancer_test.csv')
    
    logger.info(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    
    # Preprocess
    logger.info("Preprocessing data...")
    train_df = preprocess(train_df, onehot=True)
    valid_df = preprocess(valid_df, onehot=True)
    test_df = preprocess(test_df, onehot=True)
    
    # Merge tabtext embeddings if enabled
    if config.use_tabtext:
        logger.info("Merging tabtext embeddings...")
        train_df = merge_tabtext_embeddings(train_df)
        valid_df = merge_tabtext_embeddings(valid_df)
        test_df = merge_tabtext_embeddings(test_df)
    
    # Store all results
    all_results = {}
    
    # Train model for each cancer type
    for cancer_type in config.cancer_types:
        logger.info("\n" + "=" * 60)
        logger.info(f"Training model for: {cancer_type.upper()}")
        logger.info("=" * 60)
        
        # Check if time column exists
        time_col = f'{cancer_type}_time_to_diagnosis'
        if time_col not in train_df.columns:
            logger.warning(f"Skipping {cancer_type}: column {time_col} not found")
            continue
        
        # Filter cohorts
        filtered_train = filter_cohort_by_time(train_df, cancer_type, config.min_followup)
        filtered_valid = filter_cohort_by_time(valid_df, cancer_type, config.min_followup)
        filtered_test = filter_cohort_by_time(test_df, cancer_type, config.min_followup)
        
        # Filter by sex for sex-specific cancers
        filtered_train = filter_by_sex(filtered_train, cancer_type)
        filtered_valid = filter_by_sex(filtered_valid, cancer_type)
        filtered_test = filter_by_sex(filtered_test, cancer_type)
        
        # Get features
        feature_cols = get_feature_columns(
            filtered_train,
            use_olink=config.use_olink,
            use_blood=config.use_blood,
            use_demo=config.use_demo,
            use_tabtext=config.use_tabtext,
            cancer_type=cancer_type
        )
        
        X_train = get_X(filtered_train, feature_cols)
        X_valid = get_X(filtered_valid, feature_cols)
        X_test = get_X(filtered_test, feature_cols)
        
        y_train = get_y(filtered_train, cancer_type, config.prediction_window)
        y_valid = get_y(filtered_valid, cancer_type, config.prediction_window)
        y_test = get_y(filtered_test, cancer_type, config.prediction_window)
        
        # Check for sufficient positive cases
        min_positive = 10
        if y_train.sum() < min_positive or y_valid.sum() < min_positive:
            logger.warning(f"Skipping {cancer_type}: insufficient positive cases")
            continue
        
        # Train model
        if config.model_type == "lightgbm":
            logger.info("Using LightGBM")
            if use_hyperparameter_tuning and OPTUNA_AVAILABLE:
                model, best_params = train_lightgbm_with_tuning(
                    X_train, y_train, X_valid, y_valid,
                    cancer_type=cancer_type,
                    n_trials=config.n_optuna_trials
                )
            else:
                model, best_params = train_lightgbm_simple_wrapper(
                    X_train, y_train, X_valid, y_valid
                )
            # Get predictions for threshold optimization
            y_valid_pred = model.predict(X_valid)
        else:  # xgboost
            logger.info("Using XGBoost")
            if use_hyperparameter_tuning and OPTUNA_AVAILABLE:
                model, best_params = train_xgboost_with_tuning(
                    X_train, y_train, X_valid, y_valid,
                    cancer_type=cancer_type,
                    n_trials=config.n_optuna_trials
                )
            else:
                model, best_params = train_xgboost_simple_wrapper(
                    X_train, y_train, X_valid, y_valid
                )
            # Get predictions for threshold optimization
            dvalid = xgb.DMatrix(data=X_valid, label=y_valid, enable_categorical=True)
            y_valid_pred = model.predict(dvalid)
        
        # Save best parameters
        if use_hyperparameter_tuning and OPTUNA_AVAILABLE:
            with open(f"{config.results_dir}/best_params_{cancer_type}.json", 'w') as f:
                json.dump(best_params, f, indent=2, default=str)
        
        if config.optimize_threshold:
            optimal_threshold = find_optimal_threshold(
                y_valid.values, y_valid_pred, 
                metric=config.threshold_metric
            )
            logger.info(f"Optimal threshold: {optimal_threshold:.3f}")
        else:
            optimal_threshold = 0.5
        
        # Evaluate on all sets
        train_metrics = evaluate_model(model, X_train, y_train, optimal_threshold, "train", config.model_type)
        valid_metrics = evaluate_model(model, X_valid, y_valid, optimal_threshold, "valid", config.model_type)
        test_metrics = evaluate_model(model, X_test, y_test, optimal_threshold, "test", config.model_type)
        
        # Print results
        logger.info(f"\n{cancer_type.upper()} Results (threshold={optimal_threshold:.3f}):")
        logger.info(f"  Train - AUC: {train_metrics['auc_roc']:.3f}, AUC-PR: {train_metrics['auc_pr']:.3f}, F1: {train_metrics['f1']:.3f}")
        logger.info(f"  Valid - AUC: {valid_metrics['auc_roc']:.3f}, AUC-PR: {valid_metrics['auc_pr']:.3f}, F1: {valid_metrics['f1']:.3f}")
        logger.info(f"  Test  - AUC: {test_metrics['auc_roc']:.3f}, AUC-PR: {test_metrics['auc_pr']:.3f}, F1: {test_metrics['f1']:.3f}")
        logger.info(f"  Test  - Precision: {test_metrics['precision']:.3f}, Recall: {test_metrics['recall']:.3f}")
        logger.info(f"  Test  - Specificity: {test_metrics['specificity']:.3f}, NPV: {test_metrics['npv']:.3f}")
        logger.info(f"  Number needed to screen: {test_metrics['nns']:.1f}")
        
        # Feature importance
        importance_df = get_feature_importance(model, feature_cols, model_type=config.model_type)
        logger.info(f"\nTop 10 features for {cancer_type}:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.2f}")
        
        # Save feature importance
        importance_df.to_csv(
            f"{config.results_dir}/feature_importance_{cancer_type}.csv", 
            index=False
        )
        
        # Create plots
        if config.model_type == "xgboost":
            dtest = xgb.DMatrix(data=X_test, label=y_test, enable_categorical=True)
            y_test_pred = model.predict(dtest)
        else:
            y_test_pred = model.predict(X_test)
        
        plot_results(
            y_test.values, y_test_pred, cancer_type,
            f"{config.results_dir}/plots_{cancer_type}.png"
        )
        
        # Save model
        model_filename = f"{config.models_dir}/{config.model_type}_model_{config.prediction_window}yr_{cancer_type}"
        if config.model_type == "xgboost":
            model.save_model(f"{model_filename}.json")
        else:
            model.save_model(f"{model_filename}.txt")
        
        # Store results
        all_results[cancer_type] = {
            "optimal_threshold": optimal_threshold,
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
            "n_features": len(feature_cols),
        }
    
    # Save all results
    with open(f"{config.results_dir}/all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: 1-Year Cancer Prediction Performance (Test Set)")
    logger.info("=" * 80)
    logger.info(f"{'Cancer Type':<15} {'AUC-ROC':>8} {'AUC-PR':>8} {'Precision':>10} {'Recall':>8} {'F1':>6} {'NNS':>6}")
    logger.info("-" * 80)
    
    for cancer_type, results in all_results.items():
        m = results['test_metrics']
        logger.info(
            f"{cancer_type:<15} {m['auc_roc']:>8.3f} {m['auc_pr']:>8.3f} "
            f"{m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>6.3f} {m['nns']:>6.1f}"
        )
    
    logger.info("=" * 80)
    logger.info("Pipeline completed successfully!")
    
    return all_results


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cancer Prediction Pipeline - Predict cancer diagnosis within a time horizon"
    )
    parser.add_argument(
        "--prediction_horizon", 
        type=float, 
        default=1.0,
        help="Time horizon in years for cancer prediction (default: 1.0 year)"
    )
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Disable hyperparameter tuning for faster training"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials for hyperparameter tuning (default: 50)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["xgboost", "lightgbm"],
        default="xgboost",
        help="Model type to use: 'xgboost' or 'lightgbm' (default: xgboost)"
    )
    
    args = parser.parse_args()
    
    # Update config with command-line arguments
    config.prediction_window = args.prediction_horizon
    config.n_optuna_trials = args.n_trials
    config.model_type = args.model
    
    logger.info(f"Model type: {config.model_type}")
    logger.info(f"Prediction horizon: {config.prediction_window} year(s)")
    
    results = run_cancer_prediction_pipeline(use_hyperparameter_tuning=not args.no_tuning)
