"""
Core metrics computation functions for classification and multiclass evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
    roc_auc_score,
    log_loss,
)
from scipy.stats import pearsonr, spearmanr
from typing import Dict


def compute_metrics(
    y_score: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
    threshold_l: float = np.inf
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics for a binary classification task.

    Args:
        y_score: Probabilities of class "1" as returned by the classifier.
        y_true: True labels.
        threshold: Lower threshold for positive class prediction.
        threshold_l: Upper threshold for positive class prediction.

    Returns:
        Dictionary containing the evaluation metrics:
            - accuracy
            - balanced_accuracy
            - f1_score
            - precision
            - recall
            - roc_auc
            - precision_recall_auc
            - true_positive_count
            - false_positive_count
            - true_negative_count
            - false_negative_count
            - mathews_correlation_coefficient
            - cohens_kappa
    """
    y_pred = ((y_score > threshold) & (y_score < threshold_l)).astype(float)

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_score)

    # Precision-Recall AUC
    precision_points, recall_points, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall_points, precision_points)

    # Confusion matrix counts
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Matthews correlation coefficient and Cohen's Kappa
    mcc = matthews_corrcoef(y_true, y_pred)
    cohens_kappa = cohen_kappa_score(y_true, y_pred)

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "precision_recall_auc": pr_auc,
        "true_positive_count": tp,
        "false_positive_count": fp,
        "true_negative_count": tn,
        "false_negative_count": fn,
        "mathews_correlation_coefficient": mcc,
        "cohens_kappa": cohens_kappa,
    }

    return metrics


def compute_relevant_metrics(
    y_score: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
    threshold_l: float = np.inf
) -> Dict[str, float]:
    """Compute selected evaluation metrics for a binary classification task.

    A lighter version of compute_metrics that only returns the most commonly
    used metrics.

    Args:
        y_score: Probabilities of class "1" as returned by the classifier.
        y_true: True labels.
        threshold: Lower threshold for positive class prediction.
        threshold_l: Upper threshold for positive class prediction.

    Returns:
        Dictionary containing:
            - balanced_accuracy
            - roc_auc
    """
    y_pred = ((y_score > threshold) & (y_score < threshold_l)).astype(float)

    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_score)

    metrics = {
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": roc_auc,
    }

    return metrics


def compute_prs_metrics(
    prs: np.ndarray,
    targets: np.ndarray,
    quantile_h: float = 0.95,
    quantile_l: float = None,
    verbose: bool = False
) -> Dict[str, float]:
    """Compute metrics for Polygenic Risk Score using quantile thresholds.

    Args:
        prs: Polygenic risk scores.
        targets: True labels.
        quantile_h: Upper quantile threshold (e.g., 0.95 for top 5%).
        quantile_l: Lower quantile threshold (e.g., 0.05 for bottom 5%).
        verbose: Whether to print threshold values.

    Returns:
        Dictionary containing the evaluation metrics.
    """
    assert quantile_h or quantile_l, "At least one quantile should be defined"

    if isinstance(prs, pd.Series):
        prs = prs.to_numpy()

    if quantile_h:
        q1 = np.quantile(prs, quantile_h)
    else:
        q1 = -np.inf

    if quantile_l:
        q2 = np.quantile(prs, quantile_l)
    else:
        q2 = np.inf

    if verbose:
        print(f"q1: {q1}, q2: {q2}")

    return compute_metrics(prs, targets, threshold=q1, threshold_l=q2)


def compute_mcfadden_pseudo_r2(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute McFadden's Pseudo-R² for multiclass classification.

    R² = 1 - (ll_model / ll_null)
    where ll_null is log-likelihood of majority-class baseline model.

    Args:
        y_true: True integer class labels.
        y_prob: Predicted class probabilities of shape (n_samples, n_classes).

    Returns:
        McFadden's Pseudo-R² value.
    """
    # Model log-likelihood (negative log-loss)
    ll_model = -log_loss(y_true, y_prob) * len(y_true)

    # Null model: predict majority class for all samples
    majority_class = np.bincount(y_true.astype(int)).argmax()
    y_null_prob = np.zeros_like(y_prob)
    y_null_prob[:, majority_class] = 1.0
    ll_null = -log_loss(y_true, y_null_prob) * len(y_true)

    pseudo_r2 = 1 - (ll_model / ll_null)
    return pseudo_r2


def compute_multiclass_metrics(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    ordinal: bool = False,
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics for multiclass classification.

    Args:
        y_prob: Probability predictions for each class (n_samples, n_classes).
        y_true: True integer class labels (0 to n_classes-1).
        ordinal: If True, compute ordinal MAE treating classes as ordered.

    Returns:
        Dictionary containing:
            - log_loss: Multinomial cross-entropy (lower is better)
            - accuracy: Overall accuracy
            - balanced_accuracy: Macro-averaged recall (handles imbalance)
            - macro_auc: Macro-averaged one-vs-rest AUC
            - weighted_auc: Sample-weighted one-vs-rest AUC
            - auc_class_N: Per-class AUC (one-vs-rest)
            - mcfadden_pseudo_r2: McFadden's Pseudo-R²
            - f1_macro: Macro-averaged F1 score
            - f1_weighted: Weighted F1 score
            - cohens_kappa: Cohen's Kappa statistic
            - confusion_matrix: 2D array of shape (n_classes, n_classes)
            - ordinal_mae: Mean absolute error treating classes as ordinal (if ordinal=True)
    """
    y_pred = y_prob.argmax(axis=1)
    n_classes = y_prob.shape[1]

    # Core metrics
    logloss = log_loss(y_true, y_prob)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # AUC metrics (one-vs-rest)
    try:
        macro_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        weighted_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
    except ValueError:
        # Handle case where some classes are missing in y_true
        macro_auc = np.nan
        weighted_auc = np.nan

    # Per-class AUC
    per_class_auc = {}
    for class_id in range(n_classes):
        y_true_binary = (y_true == class_id).astype(int)
        if len(np.unique(y_true_binary)) > 1:  # Need both classes present
            try:
                per_class_auc[f'auc_class_{class_id}'] = roc_auc_score(
                    y_true_binary, y_prob[:, class_id]
                )
            except ValueError:
                per_class_auc[f'auc_class_{class_id}'] = np.nan
        else:
            per_class_auc[f'auc_class_{class_id}'] = np.nan

    # McFadden's Pseudo-R²
    pseudo_r2 = compute_mcfadden_pseudo_r2(y_true, y_prob)

    # F1 scores
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "log_loss": logloss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_auc": macro_auc,
        "weighted_auc": weighted_auc,
        **per_class_auc,  # Unpack per-class AUCs
        "mcfadden_pseudo_r2": pseudo_r2,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "cohens_kappa": kappa,
        "confusion_matrix": cm.tolist(),  # Convert to list for JSON serialization
    }

    # Ordinal MAE (if requested)
    if ordinal:
        ordinal_mae = np.mean(np.abs(y_pred - y_true))
        metrics["ordinal_mae"] = ordinal_mae

    return metrics


def compute_relevant_multiclass_metrics(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    ordinal: bool = False,
) -> Dict[str, float]:
    """Compute selected multiclass metrics (lighter version).

    Returns only the most commonly used metrics.

    Args:
        y_prob: Probability predictions for each class (n_samples, n_classes).
        y_true: True integer class labels.
        ordinal: If True, compute ordinal MAE.

    Returns:
        Dictionary containing:
            - log_loss: Multinomial cross-entropy
            - balanced_accuracy: Macro-averaged recall
            - macro_auc: Macro-averaged one-vs-rest AUC
            - ordinal_mae: (if ordinal=True)
    """
    y_pred = y_prob.argmax(axis=1)

    logloss = log_loss(y_true, y_prob)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    try:
        macro_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except ValueError:
        macro_auc = np.nan

    metrics = {
        "log_loss": logloss,
        "balanced_accuracy": balanced_acc,
        "macro_auc": macro_auc,
    }

    if ordinal:
        metrics["ordinal_mae"] = np.mean(np.abs(y_pred - y_true))

    return metrics


def compute_prs_multiclass_metrics(
    prs: np.ndarray,
    y_true: np.ndarray,
    ordinal: bool = True,
) -> Dict[str, float]:
    """Compute metrics for PRS (single continuous score) with multiclass labels.

    Fits a multinomial logistic regression model (y ~ PRS) and evaluates using
    model predictions, matching PLINK/PRSice/LDPred2 evaluation methodology.

    This approach:
    1. Fits multinomial logistic regression with PRS as the single predictor
    2. Uses model-predicted probabilities for AUC computation
    3. Uses argmax of probabilities for class predictions (accuracy, MAE)

    Args:
        prs: Continuous PRS scores (n_samples,).
        y_true: True integer class labels (0 to n_classes-1).
        ordinal: If True, compute ordinal MAE in addition to other metrics.

    Returns:
        Dictionary containing:
            - macro_auc: Macro-averaged one-vs-rest AUC (using model probabilities)
            - weighted_auc: Sample-weighted one-vs-rest AUC
            - auc_class_N: Per-class AUC (one-vs-rest)
            - accuracy: Overall accuracy
            - balanced_accuracy: Balanced accuracy
            - spearman_r: Spearman rank correlation (raw PRS vs true class)
            - pearson_r: Pearson correlation (raw PRS vs true class)
            - ordinal_mae: Mean absolute error (if ordinal=True)
    """
    from sklearn.linear_model import LogisticRegression

    # Ensure arrays
    prs = np.asarray(prs).ravel()
    y_true = np.asarray(y_true).ravel().astype(int)
    n_classes = len(np.unique(y_true))

    # =========================================================================
    # Fit multinomial logistic regression: y ~ PRS (like PLINK's nnet::multinom)
    # =========================================================================
    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        penalty=None,  # No regularization, like nnet::multinom default
    )
    model.fit(prs.reshape(-1, 1), y_true)

    # Get model predictions
    probs = model.predict_proba(prs.reshape(-1, 1))  # (n_samples, n_classes)
    y_pred = model.predict(prs.reshape(-1, 1))

    # =========================================================================
    # AUC metrics using model probabilities (PLINK-style)
    # =========================================================================
    try:
        macro_auc = roc_auc_score(y_true, probs, multi_class='ovr', average='macro')
        weighted_auc = roc_auc_score(y_true, probs, multi_class='ovr', average='weighted')
    except ValueError:
        macro_auc = np.nan
        weighted_auc = np.nan

    # Per-class AUC using model probabilities
    per_class_auc = {}
    for class_id in range(n_classes):
        y_binary = (y_true == class_id).astype(int)
        if len(np.unique(y_binary)) > 1:
            try:
                per_class_auc[f'auc_class_{class_id}'] = roc_auc_score(
                    y_binary, probs[:, class_id]
                )
            except ValueError:
                per_class_auc[f'auc_class_{class_id}'] = np.nan
        else:
            per_class_auc[f'auc_class_{class_id}'] = np.nan

    # =========================================================================
    # Accuracy metrics using model predictions
    # =========================================================================
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # =========================================================================
    # Correlation metrics (raw PRS vs true class - informational)
    # =========================================================================
    spearman_r, _ = spearmanr(prs, y_true)
    pearson_r, _ = pearsonr(prs, y_true)

    metrics = {
        "macro_auc": macro_auc,
        "weighted_auc": weighted_auc,
        **per_class_auc,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "spearman_r": spearman_r,
        "pearson_r": pearson_r,
    }

    # =========================================================================
    # Ordinal MAE (if applicable)
    # =========================================================================
    if ordinal:
        ordinal_mae = np.mean(np.abs(y_pred - y_true))
        metrics["ordinal_mae"] = ordinal_mae

    return metrics
