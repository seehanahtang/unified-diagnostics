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
    cancer_types: List[str] = field(default_factory=lambda: [
        # "oral_pharynx_cancer",
        # "digestive_organs_cancer",       
        # "respiratory_intrathoracic_cancer",     
        # "skin_cancer",
        # "mesothelial_soft_tissue_cancer",
        # "breast_cancer",
        # "female_genital_cancer",
        # "male_genital_cancer",
        # "urinary_tract_cancer",
        # "eye_brain_cns_cancer",
        # "endocrine_cancer",
        # "ill_defined_secondary_cancer",
        # "in_situ_cancer",
        # "hematologic_cancer",

        # "bone_cartilage_cancer", # get rid of this one for now
        "skin_cancer",
        "breast_cancer",
        "prostate_cancer",
        "colorectal_cancer",
        "lung_cancer",
        "lymphoma_cancer",
        "kidney_cancer",
        "leukemia_cancer",
        "bladder_cancer",
        "pancreatic_cancer",
        "brain_cancer",
        "stomach_cancer"
    ])

    # diag_types: List[str] = field(default_factory=lambda: [
    #     "lung_cancer",
    #     "colorectal_cancer",
    #     "stomach_cancer",
    #     "ischemia",
    #     "stroke",
    #     "alzheimers",
    #     "copd",
    #     "lower_resp",
    #     "kidney",
    #     "t2d",
    #     "hhd",
    #     "lung",
    # ])
    
    diag_types: List[str] = field(default_factory=lambda: [
        "ischemic_heart_disease",
        "stroke",
        "hypertensive_heart_kidney_diseases",
        "heart_failure",
        "atrial_fibrillation",
        "peripheral_vascular_disease",
        "type_2_diabetes",
        "type_1_diabetes",
        "alzheimers_disease",
        "other_dementia",
        "parkinsons",
        "copd",
        "lower_respiratory_disease",
        "chronic_kidney_disease",
        "acute_kidney_injury",
        "end_stage_renal_disease",
        "liver_disease",
        "depression",
        "anxiety_disorders"
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
]
