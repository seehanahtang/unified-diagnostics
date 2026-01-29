"""
Data Loading and Preprocessing
==============================
Functions for loading UK Biobank data, preprocessing, and feature engineering.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Tuple

from config import config, DEMO_FEATURES, FEMALE_ONLY_CANCERS, MALE_ONLY_CANCERS

logger = logging.getLogger(__name__)

# Global variables for tabtext embeddings
demo_embeddings = None
embedding_cols = []


def load_tabtext_embeddings():
    """Load tabtext embeddings from file."""
    global demo_embeddings, embedding_cols
    
    if not config.use_tabtext:
        return
    
    try:
        demo_embeddings = pd.read_csv(f"{config.data_dir}demo_embeddings.csv")
        embedding_cols = [col for col in demo_embeddings.columns if col.startswith("cn_")]
        logger.info(f"Loaded {len(embedding_cols)} tabtext embedding columns")
    except FileNotFoundError:
        logger.warning("demo_embeddings.csv not found. Tabtext features disabled.")
        config.use_tabtext = False


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, validation, and test datasets.
    
    Returns:
        Tuple of (train_df, valid_df, test_df)
    """
    logger.info("Loading datasets...")
    # train_df = pd.read_csv(f'{config.data_dir}ukb_cancer_train_with_skin.csv')
    # valid_df = pd.read_csv(f'{config.data_dir}ukb_cancer_valid_with_skin.csv')
    # test_df = pd.read_csv(f'{config.data_dir}ukb_cancer_test_with_skin.csv')
    
    train_df = pd.read_csv(f'{config.data_dir}ukb_diag_train.csv')
    valid_df = pd.read_csv(f'{config.data_dir}ukb_diag_valid.csv')
    test_df = pd.read_csv(f'{config.data_dir}ukb_diag_test.csv')
    
    logger.info(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    
    return train_df, valid_df, test_df


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


def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize feature names to remove special characters that LightGBM doesn't support.
    
    Args:
        df: DataFrame with potentially problematic column names
        
    Returns:
        DataFrame with sanitized column names
    """
    import re
    
    def clean_name(name: str) -> str:
        # Replace special JSON characters and other problematic chars
        name = re.sub(r'[\[\]{}:,"\'\\]', '_', name)
        # Replace spaces and dots with underscores
        name = name.replace(' ', '_').replace('.', '_')
        # Remove multiple consecutive underscores
        name = re.sub(r'_+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        return name
    
    new_columns = {col: clean_name(col) for col in df.columns}
    return df.rename(columns=new_columns)


def get_X(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """Extract feature matrix from dataframe and sanitize column names."""
    available_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = set(feature_cols) - set(available_cols)
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
    
    X = df[available_cols].copy()
    # Sanitize column names for LightGBM compatibility
    X = sanitize_feature_names(X)
    return X


def filter_high_missingness(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5
) -> tuple:
    """
    Filter out patients (rows) with more than threshold proportion of missing values.
    
    Args:
        X: Feature matrix
        y: Labels (will be filtered to match X)
        threshold: Maximum proportion of missing values allowed (default 0.5 = 50%)
        
    Returns:
        Tuple of (filtered X, filtered y)
    """
    n_features = X.shape[1]
    missing_proportion = X.isna().sum(axis=1) / n_features
    
    # Keep rows with missing proportion <= threshold
    mask = missing_proportion <= threshold
    
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        logger.info(f"Dropped {n_dropped} patients with >{threshold*100:.0f}% missing values")
    
    return X.loc[mask].copy(), y.loc[mask].copy()
    
    


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


def get_y_multiclass(
    df: pd.DataFrame,
    cancer_types: List[str],
    prediction_window: float = 1.0
) -> Tuple[pd.Series, dict]:
    """
    Get multiclass labels for cancer type prediction within prediction window.
    
    Classes:
        0: No cancer within prediction window
        1 to N: Specific cancer types (from cancer_types list)
        N+1: Other cancer (from cancer_time_to_diagnosis but not in cancer_types)
    
    Args:
        df: Input dataframe
        cancer_types: List of cancer type prefixes to predict
        prediction_window: Years within which to predict cancer
        
    Returns:
        Tuple of (multiclass labels, class mapping dict)
    """
    # Initialize all as 0 (no cancer)
    y = pd.Series(0, index=df.index)
    
    # Create class mapping: 0=no_cancer, 1-N=specific cancers, N+1=other
    class_mapping = {0: 'no_cancer'}
    for i, cancer_type in enumerate(cancer_types, start=1):
        class_mapping[i] = cancer_type
    other_class = len(cancer_types) + 1
    class_mapping[other_class] = 'other_cancer'
    
    # Check for "other" cancer using the general cancer column
    other_col = "cancer_time_to_diagnosis"
    if other_col in df.columns:
        other_mask = (df[other_col] > 0) & (df[other_col] <= prediction_window)
        y.loc[other_mask] = other_class
        logger.info(f"Any cancer: {other_mask.sum()} within {prediction_window} years")
    
    # Override with specific cancer types (priority to specific over "other")
    # Process in reverse order so earlier cancers in list have priority
    for i, cancer_type in reversed(list(enumerate(cancer_types, start=1))):
        time_col = f"{cancer_type}_time_to_diagnosis"
        if time_col not in df.columns:
            logger.warning(f"Column {time_col} not found, skipping")
            continue
        
        # Mark as this cancer type if diagnosed within window
        mask = (df[time_col] > 0) & (df[time_col] <= prediction_window)
        y.loc[mask] = i
    
    # Log class distribution
    logger.info("Multiclass distribution:")
    for class_id, class_name in class_mapping.items():
        count = (y == class_id).sum()
        pct = count / len(y) * 100 if len(y) > 0 else 0
        logger.info(f"  {class_id}: {class_name} - {count} ({pct:.2f}%)")
    
    return y, class_mapping
