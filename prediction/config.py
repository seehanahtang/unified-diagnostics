"""
Configuration for Cancer Prediction Pipeline
=============================================
"""

import os
import logging
from typing import List
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration for cancer prediction pipeline."""
    # Directories
    data_dir: str = "/../../orcd/pool/003/dbertsim_shared/ukb/"
    output_dir: str = "out/"
    models_dir: str = "results/"
    
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
    # cancer_types: List[str] = field(default_factory=lambda: [
    #     "skin",
    #     "breast",       # Breast cancer (female only)
    #     "prostate",     # Prostate cancer (male only)
    #     "lung",         # Lung cancer
    #     "colorectal",   # Colorectal cancer
    #     "bladder",      # Bladder cancer
    #     # "pancreatic",   # Pancreatic cancer
    # ])
    # cancer_types: List[str] = field(default_factory=lambda: [
    #     "lung_cancer",
    #     "colorectal_cancer",
    #     # "stomach_cancer",
    #     "ischemia",
    #     "stroke",
    #     # "alzheimers",
    #     "copd",
    #     "lower_resp",
    #     "kidney",
    #     # "hhd",
    #     "diabetes",
    # ])
    cancer_types: List[str] = field(default_factory=lambda: [
        "ischemia",
        "lower_resp",
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
    
    # Classification mode
    classification_mode: str = "binary"  # "binary" or "multiclass"
    
    # Model selection
    model_type: str = "xgboost"  # "xgboost" or "lightgbm"
    use_gpu: bool = False  # Use GPU for training
    
    # Threshold optimization
    optimize_threshold: bool = True
    threshold_method: str = "target_recall"
    target_recall: float = 0.8


# Global config instance
config = Config()


def setup_directories():
    """Create output directories."""
    os.makedirs(config.models_dir, exist_ok=True)


def setup_logging(model_type: str = "xgboost", prediction_horizon: float = 1.0):
    """Configure logging after directories exist."""
    setup_directories()
    
    # Create log filename with model and prediction horizon
    log_file = f"{config.output_dir}cancer_prediction_{model_type}_{config.threshold_method}_{prediction_horizon}yr.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# Feature constants
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
FEMALE_ONLY_CANCERS = ["breast", "breast_cancer", "ovarian", "ovarian_cancer", "uterine", "uterine_cancer"]
MALE_ONLY_CANCERS = ["prostate", "prostate_cancer"]
