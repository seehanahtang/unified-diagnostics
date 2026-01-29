"""
Model Evaluation
================
Functions for evaluating model performance and plotting results.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix, roc_curve
)

from config import config

logger = logging.getLogger(__name__)


def find_optimal_threshold(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray,
    method: str = 'f1',
    target_recall: float = 0.8
) -> float:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        method: 'f1', 'youden', 'precision_recall', or 'target_recall'
        target_recall: Target recall level when method='target_recall'
        
    Returns:
        Optimal threshold
    """
    if method == 'f1':
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        # Avoid division by zero
        f1_scores = np.where(
            (precision + recall) > 0,
            2 * (precision * recall) / (precision + recall),
            0
        )
        # Thresholds has one less element
        if len(f1_scores) > len(thresholds):
            f1_scores = f1_scores[:-1]
        best_idx = np.argmax(f1_scores)
        return thresholds[best_idx]
    
    elif method == 'youden':
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        return thresholds[best_idx]
    
    elif method == 'precision_recall':
        # Find threshold where precision and recall are balanced
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        diff = np.abs(precision[:-1] - recall[:-1])
        best_idx = np.argmin(diff)
        return thresholds[best_idx]
    
    elif method == 'target_recall':
        # Find lowest threshold that achieves at least target_recall
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        # recall is in decreasing order, find where it drops below target
        valid_idx = np.where(recall[:-1] >= target_recall)[0]
        if len(valid_idx) == 0:
            # Can't achieve target, use lowest threshold
            return thresholds[0] if len(thresholds) > 0 else 0.5
        # Use highest threshold that still achieves target recall (best precision)
        best_idx = valid_idx[-1]
        return thresholds[best_idx]
    
    elif method == 'percentile':
        # Classify top X% as positive (based on prevalence)
        prevalence = y_true.mean()
        # Be more generous - predict 2-3x the prevalence rate
        target_positive_rate = min(prevalence * 3, 0.1)
        threshold = np.percentile(y_pred_proba, (1 - target_positive_rate) * 100)
        return threshold
    
    else:
        raise ValueError(f"Unknown method: {method}")


def evaluate_model(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: Optional[float] = None,
    threshold_method: str = 'youden',
    target_recall: float = 0.8,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold (auto-selected if None)
        threshold_method: Method for auto-threshold ('f1', 'youden', 'target_recall', 'percentile')
        target_recall: Target recall when using 'target_recall' method
        prefix: Prefix for metric names
        
    Returns:
        Dictionary of metrics
    """
    if threshold is None:
        threshold = find_optimal_threshold(
            y_true, y_pred_proba, 
            method=threshold_method, 
            target_recall=target_recall
        )
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Basic metrics
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    auc_pr = average_precision_score(y_true, y_pred_proba)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix derived metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Number needed to screen
    nns = 1 / precision if precision > 0 else float('inf')
    
    metrics = {
        f'{prefix}auc_roc': auc_roc,
        f'{prefix}auc_pr': auc_pr,
        f'{prefix}precision': precision,
        f'{prefix}recall': recall,
        f'{prefix}f1': f1,
        f'{prefix}specificity': specificity,
        f'{prefix}npv': npv,
        f'{prefix}nns': nns,
        f'{prefix}threshold': threshold,
        f'{prefix}tp': tp,
        f'{prefix}fp': fp,
        f'{prefix}tn': tn,
        f'{prefix}fn': fn,
    }
    
    return metrics


def evaluate_model_multiclass(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_mapping: dict,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Comprehensive multiclass model evaluation.
    
    Args:
        y_true: True labels (class indices)
        y_pred_proba: Predicted probabilities (n_samples x n_classes)
        class_mapping: Dict mapping class indices to names
        prefix: Prefix for metric names
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, 
        classification_report, log_loss
    )
    
    y_pred = np.argmax(y_pred_proba, axis=1)
    num_classes = len(class_mapping)
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Multiclass AUC-ROC (one-vs-rest)
    try:
        auc_roc_weighted = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        auc_roc_macro = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
    except ValueError:
        auc_roc_weighted = 0.0
        auc_roc_macro = 0.0
    
    # One-vs-Rest AUC: No Cancer (class 0) vs Any Cancer (class 1+)
    try:
        y_binary = (y_true > 0).astype(int)  # 0 = no cancer, 1 = any cancer
        prob_cancer = 1 - y_pred_proba[:, 0]  # P(any cancer) = 1 - P(no cancer)
        auc_cancer_vs_no_cancer = roc_auc_score(y_binary, prob_cancer)

        # Fix: y_true and y_pred_proba are usually numpy arrays, not DataFrames
        # y_true: multiclass labels (integers), y_pred_proba: ndarray (n_samples, n_classes)
        # For restricted: any cancer except class 0 (no_cancer) and last class (other_cancer)
        if isinstance(y_true, pd.Series):
            y_true_arr = y_true.values
        else:
            y_true_arr = y_true
        if isinstance(y_pred_proba, pd.DataFrame):
            y_pred_arr = y_pred_proba.values
        else:
            y_pred_arr = y_pred_proba
        # Classes: 0=no_cancer, 1..n-2=cancers, n-1=other_cancer
        y_binary_restricted = ((y_true_arr >= 1) & (y_true_arr < y_pred_arr.shape[1]-1)).astype(int)
        prob_cancer_restricted = y_pred_arr[:,1:-1].max(axis=1)
        auc_cancer_vs_no_cancer_restricted = roc_auc_score(y_binary_restricted, prob_cancer_restricted)
    except (ValueError, IndexError):
        auc_cancer_vs_no_cancer = 0.0
        auc_cancer_vs_no_cancer_restricted = 0.0
    
    # Log loss
    try:
        logloss = log_loss(y_true, y_pred_proba)
    except ValueError:
        logloss = float('inf')
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Weighted averages
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        f'{prefix}accuracy': accuracy,
        f'{prefix}balanced_accuracy': balanced_acc,
        f'{prefix}auc_roc_weighted': auc_roc_weighted,
        f'{prefix}auc_roc_macro': auc_roc_macro,
        f'{prefix}auc_cancer_vs_no_cancer': auc_cancer_vs_no_cancer,
        f'{prefix}auc_cancer_vs_no_cancer_restricted': auc_cancer_vs_no_cancer_restricted,
        f'{prefix}log_loss': logloss,
        f'{prefix}precision_weighted': precision_weighted,
        f'{prefix}recall_weighted': recall_weighted,
        f'{prefix}f1_weighted': f1_weighted,
    }
    
    # Per-class metrics
    for class_idx, class_name in class_mapping.items():
        if class_idx < len(precision_per_class):
            metrics[f'{prefix}precision_{class_name}'] = precision_per_class[class_idx]
            metrics[f'{prefix}recall_{class_name}'] = recall_per_class[class_idx]
            metrics[f'{prefix}f1_{class_name}'] = f1_per_class[class_idx]
    
    return metrics


def get_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """
    Extract feature importance from trained model.
    
    Args:
        model: Trained model (XGBoost or LightGBM)
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature importance
    """
    try:
        importance = model.feature_importances_
    except AttributeError:
        logger.warning("Model does not have feature_importances_ attribute")
        return pd.DataFrame()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df.head(top_n)


def plot_results(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    feature_importance: pd.DataFrame,
    cancer_type: str,
    model_type: str = "model",
    save_path: Optional[str] = None
) -> None:
    """
    Plot ROC curve, PR curve, and feature importance.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        feature_importance: DataFrame with feature importance
        cancer_type: Cancer type name
        model_type: Name of the model type
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    
    axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC-ROC = {auc_roc:.4f}')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'ROC Curve - {cancer_type}')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    auc_pr = average_precision_score(y_true, y_pred_proba)
    
    # Baseline is the prevalence
    baseline = y_true.mean()
    
    axes[1].plot(recall, precision, 'b-', linewidth=2, label=f'AUC-PR = {auc_pr:.4f}')
    axes[1].axhline(y=baseline, color='k', linestyle='--', linewidth=1, label=f'Baseline = {baseline:.4f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'Precision-Recall Curve - {cancer_type}')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Feature Importance
    if not feature_importance.empty:
        top_features = feature_importance.head(15)
        axes[2].barh(
            range(len(top_features)), 
            top_features['importance'].values,
            color='steelblue'
        )
        axes[2].set_yticks(range(len(top_features)))
        axes[2].set_yticklabels(top_features['feature'].values)
        axes[2].invert_yaxis()
        axes[2].set_xlabel('Importance')
        axes[2].set_title(f'Top Features ({model_type})')
    else:
        axes[2].text(0.5, 0.5, 'No feature importance available', 
                     ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Feature Importance')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None
) -> None:
    """
    Plot calibration curve to assess probability calibration.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration
        save_path: Optional path to save the figure
    """
    from sklearn.calibration import calibration_curve
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy='quantile'
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Calibration Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    # Set x-axis limits to min and max of mean_predicted_value
    ax.set_xlim(mean_predicted_value.min(), mean_predicted_value.max())
    ax.set_ylim(0, 0.06)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved calibration plot to {save_path}")
    
    plt.show()


def log_metrics(metrics: Dict[str, float], prefix: str = "", multiclass: bool = False) -> None:
    """Log evaluation metrics in a formatted way."""
    logger.info(f"{prefix}Evaluation Metrics:")
    
    if multiclass:
        logger.info(f"  Accuracy:          {metrics.get(f'{prefix}accuracy', 0):.4f}")
        logger.info(f"  Balanced Accuracy: {metrics.get(f'{prefix}balanced_accuracy', 0):.4f}")
        logger.info(f"  AUC-ROC (weighted):{metrics.get(f'{prefix}auc_roc_weighted', 0):.4f}")
        logger.info(f"  AUC-ROC (macro):   {metrics.get(f'{prefix}auc_roc_macro', 0):.4f}")
        logger.info(f"  AUC Cancer vs None:{metrics.get(f'{prefix}auc_cancer_vs_no_cancer', 0):.4f}")
        logger.info(f" AUC Top Cancers vs None:{metrics.get(f'{prefix}auc_cancer_vs_no_cancer_restricted', 0):.4f}")
        logger.info(f"  Log Loss:          {metrics.get(f'{prefix}log_loss', 0):.4f}")
        logger.info(f"  F1 (weighted):     {metrics.get(f'{prefix}f1_weighted', 0):.4f}")
        logger.info(f"  Precision (wt):    {metrics.get(f'{prefix}precision_weighted', 0):.4f}")
        logger.info(f"  Recall (wt):       {metrics.get(f'{prefix}recall_weighted', 0):.4f}")
    else:
        logger.info(f"  AUC-ROC:     {metrics.get('auc_roc', metrics.get(f'{prefix}auc_roc', 0)):.4f}")
        logger.info(f"  AUC-PR:      {metrics.get('auc_pr', metrics.get(f'{prefix}auc_pr', 0)):.4f}")
        logger.info(f"  Precision:   {metrics.get('precision', metrics.get(f'{prefix}precision', 0)):.4f}")
        logger.info(f"  Recall:      {metrics.get('recall', metrics.get(f'{prefix}recall', 0)):.4f}")
        logger.info(f"  F1 Score:    {metrics.get('f1', metrics.get(f'{prefix}f1', 0)):.4f}")
        logger.info(f"  Specificity: {metrics.get('specificity', metrics.get(f'{prefix}specificity', 0)):.4f}")
        logger.info(f"  NPV:         {metrics.get('npv', metrics.get(f'{prefix}npv', 0)):.4f}")
        nns = metrics.get('nns', metrics.get(f'{prefix}nns', float('inf')))
        logger.info(f"  NNS:         {nns:.1f}" if nns != float('inf') else "  NNS:         N/A")
        logger.info(f"  Threshold:   {metrics.get('threshold', metrics.get(f'{prefix}threshold', 0.5)):.4f}")


def summarize_results(
    all_results: List[Dict[str, Any]],
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Summarize results across multiple cancer types.
    
    Args:
        all_results: List of result dictionaries
        output_file: Optional path to save summary CSV
        
    Returns:
        Summary DataFrame
    """
    summary_df = pd.DataFrame(all_results)
    
    if output_file:
        summary_df.to_csv(output_file, index=False)
        logger.info(f"Saved summary to {output_file}")
    
    return summary_df
