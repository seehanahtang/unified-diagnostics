"""
Ensemble models for combining ~3 base-model probabilities.

Extracted from progression_analysis_hematoma_new.ipynb. Contains:
- probabilities_model: RF meta-model on base-model probabilities
- multimodal_late_fusion: simple average over columns of X_probs
- weighted_multimodal_late_fusion: AUC-normalized weighting
- weighted_softmax_multimodal_late_fusion: temperature-scaled softmax of AUCs
- pnn_multimodal_late_fusion: neural net meta-model on probabilities

All training-style functions optionally evaluate an external validation set
if X_ext_probs and y_ext are provided, and report external AUC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ModelResult:
    """Trained model plus test (and optional external) predictions and metrics."""

    model: object
    y_test: np.ndarray
    proba_test: np.ndarray
    metrics: Dict[str, float]
    best_params: Optional[Dict[str, object]] = None
    y_ext: Optional[np.ndarray] = None
    proba_ext: Optional[np.ndarray] = None


def _evaluate_binary(
    y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    if proba.ndim == 1:
        p_pos = proba
    else:
        p_pos = proba[:, 1]
    y_pred = (p_pos >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, average="macro")
    ll = log_loss(y_true, np.column_stack([1.0 - p_pos, p_pos]), labels=[0, 1])
    acc = accuracy_score(y_true, y_pred)
    counts = np.bincount(y_true.astype(int))
    baseline_acc = counts.max() / counts.sum()
    auc = roc_auc_score(y_true, p_pos)
    return {
        "F1_macro": float(f1),
        "LogLoss": float(ll),
        "Accuracy": float(acc),
        "Baseline_Accuracy": float(baseline_acc),
        "AUC_binary": float(auc),
    }


# ---------------------------------------------------------------------------
# 1. Probabilities model (RF on base-model probabilities)
# ---------------------------------------------------------------------------


def rf_ensemble(
    X_train_probs: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_test_probs: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    X_ext_probs: Optional[pd.DataFrame | np.ndarray] = None,
    y_ext: Optional[pd.Series | np.ndarray] = None,
    random_state: int = 42,
    param_grid: Optional[Dict[str, Sequence]] = None,
) -> ModelResult:
    """
    RandomForest meta-model trained on base-model probabilities only.
    """
    if param_grid is None:
        param_grid = {
            "n_estimators": [300, 600],
            "max_depth": [None, 5, 10, 20],
            "max_features": ["sqrt", 0.8, 1.0],
            "max_samples": [0.8, 1.0],
            "min_samples_leaf": [1, 5],
        }

    rf = RandomForestClassifier(
        n_jobs=-1,
        random_state=random_state,
        bootstrap=True,
    )
    gs = GridSearchCV(
        rf,
        param_grid,
        scoring="f1_macro",
        cv=3,
        verbose=1,
        n_jobs=-1,
    )
    gs.fit(X_train_probs, y_train)
    best_model = gs.best_estimator_
    best_model.fit(X_train_probs, y_train)

    proba_test = best_model.predict_proba(X_test_probs)
    metrics = _evaluate_binary(y_test, proba_test)

    y_ext_arr: Optional[np.ndarray] = None
    proba_ext_arr: Optional[np.ndarray] = None
    if X_ext_probs is not None and y_ext is not None:
        proba_ext_arr = best_model.predict_proba(X_ext_probs)
        ext_metrics = _evaluate_binary(y_ext, proba_ext_arr)
        for key, value in ext_metrics.items():
            metrics[f"Ext_{key}"] = value
        y_ext_arr = np.asarray(y_ext)

    return ModelResult(
        model=best_model,
        y_test=np.asarray(y_test),
        proba_test=proba_test,
        metrics=metrics,
        best_params=gs.best_params_,
        y_ext=y_ext_arr,
        proba_ext=proba_ext_arr,
    )


# ---------------------------------------------------------------------------
# 2. Simple ensemble (unweighted average over X_probs)
# ---------------------------------------------------------------------------


def simple_ensemble(
    X_test_probs: np.ndarray,
    y_test: np.ndarray | pd.Series,
    X_ext_probs: Optional[np.ndarray] = None,
    y_ext: Optional[np.ndarray | pd.Series] = None,
) -> ModelResult:
    """
    Simple unweighted late fusion: row-wise mean over base-model probabilities.

    X_test_probs: (n_samples, n_models) positive-class probabilities.
    y_test: labels for test set (used for metrics).
    Optionally X_ext_probs, y_ext for external validation; metrics get Ext_* keys.
    Returns ModelResult with model=None (no fitted object).
    """
    X_test_probs = np.asarray(X_test_probs, dtype=float)
    y_test_arr = np.asarray(y_test)

    p_pos_test = X_test_probs.mean(axis=1)
    proba_test = np.column_stack([1.0 - p_pos_test, p_pos_test])
    metrics = _evaluate_binary(y_test_arr, proba_test)

    y_ext_arr: Optional[np.ndarray] = None
    proba_ext_arr: Optional[np.ndarray] = None
    if X_ext_probs is not None and y_ext is not None:
        X_ext_probs = np.asarray(X_ext_probs, dtype=float)
        y_ext_arr = np.asarray(y_ext)
        p_pos_ext = X_ext_probs.mean(axis=1)
        proba_ext_arr = np.column_stack([1.0 - p_pos_ext, p_pos_ext])
        ext_metrics = _evaluate_binary(y_ext_arr, proba_ext_arr)
        for key, value in ext_metrics.items():
            metrics[f"Ext_{key}"] = value

    return ModelResult(
        model=None,
        y_test=y_test_arr,
        proba_test=proba_test,
        metrics=metrics,
        best_params=None,
        y_ext=y_ext_arr,
        proba_ext=proba_ext_arr,
    )


# ---------------------------------------------------------------------------
# 3. Weighted ensemble (AUC-normalized weights)
# ---------------------------------------------------------------------------


def weighted_ensemble(
    X_test_probs: np.ndarray,
    y_test: np.ndarray | pd.Series,
    validation_aucs: np.ndarray,
    X_ext_probs: Optional[np.ndarray] = None,
    y_ext: Optional[np.ndarray | pd.Series] = None,
) -> ModelResult:
    """
    Weighted average using validation AUCs normalized to sum to 1.

    X_test_probs: (n_samples, n_models) base-model probabilities.
    y_test: labels for test set (used for metrics).
    validation_aucs: (n_models,) validation AUCs for each base model.
    Optionally X_ext_probs, y_ext for external validation; metrics get Ext_* keys.
    Returns ModelResult with model=weights dict (for reuse).
    """
    X_test_probs = np.asarray(X_test_probs, dtype=float)
    validation_aucs = np.asarray(validation_aucs, dtype=float)
    y_test_arr = np.asarray(y_test)

    weights = validation_aucs / validation_aucs.sum()
    p_pos_test = X_test_probs @ weights
    proba_test = np.column_stack([1.0 - p_pos_test, p_pos_test])
    metrics = _evaluate_binary(y_test_arr, proba_test)

    y_ext_arr: Optional[np.ndarray] = None
    proba_ext_arr: Optional[np.ndarray] = None
    if X_ext_probs is not None and y_ext is not None:
        X_ext_probs = np.asarray(X_ext_probs, dtype=float)
        y_ext_arr = np.asarray(y_ext)
        p_pos_ext = X_ext_probs @ weights
        proba_ext_arr = np.column_stack([1.0 - p_pos_ext, p_pos_ext])
        ext_metrics = _evaluate_binary(y_ext_arr, proba_ext_arr)
        for key, value in ext_metrics.items():
            metrics[f"Ext_{key}"] = value

    return ModelResult(
        model={"weights": weights},
        y_test=y_test_arr,
        proba_test=proba_test,
        metrics=metrics,
        best_params=None,
        y_ext=y_ext_arr,
        proba_ext=proba_ext_arr,
    )


# ---------------------------------------------------------------------------
# 4. Weighted softmax ensemble
# ---------------------------------------------------------------------------


def _softmax_weights(x: np.ndarray, T: float = 0.1) -> np.ndarray:
    """Temperature-scaled softmax over model AUCs for weighting."""
    x = np.asarray(x, dtype=float)
    x = x - x.max()
    e = np.exp(x / T)
    return e / e.sum()


def weighted_softmax_ensemble(
    X_test_probs: np.ndarray,
    y_test: np.ndarray | pd.Series,
    validation_aucs: np.ndarray,
    T: float = 0.1,
    X_ext_probs: Optional[np.ndarray] = None,
    y_ext: Optional[np.ndarray | pd.Series] = None,
) -> ModelResult:
    """
    Weights = softmax(validation_aucs, T), then probs @ weights.

    X_test_probs: (n_samples, n_models) base-model probabilities.
    y_test: labels for test set (used for metrics).
    validation_aucs: (n_models,) validation AUCs. Smaller T gives more peaked weights.
    Optionally X_ext_probs, y_ext for external validation; metrics get Ext_* keys.
    Returns ModelResult with model=dict(weights=..., T=...) for reuse.
    """
    X_test_probs = np.asarray(X_test_probs, dtype=float)
    validation_aucs = np.asarray(validation_aucs, dtype=float)
    y_test_arr = np.asarray(y_test)

    weights = _softmax_weights(validation_aucs, T=T)
    p_pos_test = X_test_probs @ weights
    proba_test = np.column_stack([1.0 - p_pos_test, p_pos_test])
    metrics = _evaluate_binary(y_test_arr, proba_test)

    y_ext_arr: Optional[np.ndarray] = None
    proba_ext_arr: Optional[np.ndarray] = None
    if X_ext_probs is not None and y_ext is not None:
        X_ext_probs = np.asarray(X_ext_probs, dtype=float)
        y_ext_arr = np.asarray(y_ext)
        p_pos_ext = X_ext_probs @ weights
        proba_ext_arr = np.column_stack([1.0 - p_pos_ext, p_pos_ext])
        ext_metrics = _evaluate_binary(y_ext_arr, proba_ext_arr)
        for key, value in ext_metrics.items():
            metrics[f"Ext_{key}"] = value

    return ModelResult(
        model={"weights": weights, "T": T},
        y_test=y_test_arr,
        proba_test=proba_test,
        metrics=metrics,
        best_params=None,
        y_ext=y_ext_arr,
        proba_ext=proba_ext_arr,
    )


# ---------------------------------------------------------------------------
# 5. PNN multimodal late fusion (neural net on probabilities)
# ---------------------------------------------------------------------------


class _PNN(nn.Module):
    """
    Simple feed-forward neural network that maps base-model probabilities
    to a single logit for the positive class.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        # final logit for positive class
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns shape (batch_size, 1)
        return self.net(x)


def pnn_ensemble(
    X_train_probs: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_test_probs: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    X_ext_probs: Optional[pd.DataFrame | np.ndarray] = None,
    y_ext: Optional[pd.Series | np.ndarray] = None,
    hidden_dims: Sequence[int] = (128, 64),
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    epochs: int = 50,
    device: Optional[str] = None,
) -> ModelResult:
    """
    Train a neural network meta-model on base-model probabilities.

    This function is inspired by the PNN (presc_nn) idea in the notebook,
    but simplified to standard supervised binary training:
    - inputs: base-model probabilities per sample
    - target: binary label (0/1)
    - loss: BCEWithLogitsLoss

    If X_ext_probs and y_ext are provided, external metrics are computed
    and added to the metrics dict with the prefix "Ext_".
    """
    X_train = np.asarray(X_train_probs, dtype=np.float32)
    X_test = np.asarray(X_test_probs, dtype=np.float32)
    y_train_arr = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
    y_test_arr = np.asarray(y_test, dtype=np.float32).reshape(-1, 1)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    model = _PNN(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout)
    model.to(torch_device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train_arr)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(torch_device)
            batch_y = batch_y.to(torch_device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    # Inference on test set
    model.eval()
    with torch.no_grad():
        test_logits = model(
            torch.from_numpy(X_test).to(torch_device)
        ).cpu().numpy().reshape(-1)
        proba_pos_test = 1.0 / (1.0 + np.exp(-test_logits))
        proba_test = np.column_stack([1.0 - proba_pos_test, proba_pos_test])

    metrics = _evaluate_binary(y_test_arr.ravel(), proba_test)

    y_ext_arr: Optional[np.ndarray] = None
    proba_ext_arr: Optional[np.ndarray] = None
    if X_ext_probs is not None and y_ext is not None:
        X_ext = np.asarray(X_ext_probs, dtype=np.float32)
        y_ext_arr = np.asarray(y_ext, dtype=np.float32).reshape(-1, 1)
        with torch.no_grad():
            ext_logits = model(
                torch.from_numpy(X_ext).to(torch_device)
            ).cpu().numpy().reshape(-1)
            proba_pos_ext = 1.0 / (1.0 + np.exp(-ext_logits))
            proba_ext_arr = np.column_stack([1.0 - proba_pos_ext, proba_pos_ext])
        ext_metrics = _evaluate_binary(y_ext_arr.ravel(), proba_ext_arr)
        for key, value in ext_metrics.items():
            metrics[f"Ext_{key}"] = value
        y_ext_arr = y_ext_arr.ravel()

    return ModelResult(
        model=model,
        y_test=y_test_arr.ravel(),
        proba_test=proba_test,
        metrics=metrics,
        best_params=None,
        y_ext=y_ext_arr,
        proba_ext=proba_ext_arr,
    )


__all__ = [
    "ModelResult",
    "rf_ensemble",
    "simple_ensemble",
    "weighted_ensemble",
    "weighted_softmax_ensemble",
    "pnn_ensemble",
]
