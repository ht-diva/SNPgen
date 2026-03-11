"""
Utility functions for cross-validation, hyperparameter search, and data splitting.
"""

import itertools
import random
from typing import Callable, Dict, Tuple, Any, Optional

import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def detect_task_type(y: np.ndarray, task_type: str = 'auto') -> str:
    """Detect whether targets represent classification or multiclass task.

    Args:
        y: Target values array.
        task_type: Explicit task type or 'auto' for detection.
            Valid values: 'auto', 'classification', 'multiclass'.

    Returns:
        'classification' (binary) or 'multiclass'.

    Raises:
        ValueError: If task_type is invalid.
    """
    valid_types = ['auto', 'classification', 'multiclass']
    if task_type not in valid_types:
        raise ValueError(f"Invalid task_type: {task_type}. Must be one of {valid_types}")

    if task_type != 'auto':
        return task_type

    # Auto-detection logic
    unique_values = np.unique(y)
    n_unique = len(unique_values)

    # Binary classification: only {0, 1} or {0.0, 1.0}
    if n_unique == 2 and set(unique_values) <= {0, 1, 0.0, 1.0}:
        return 'classification'

    # Check if values are integer-like for multiclass
    if np.allclose(y, np.round(y)):
        y_int = np.round(y).astype(int)
        unique_int = np.unique(y_int)

        # Multiclass: >2 classes, consecutive integers from 0
        if len(unique_int) > 2 and np.array_equal(unique_int, np.arange(len(unique_int))):
            return 'multiclass'

    # Default to classification for unrecognized patterns
    return 'classification'


def get_stratified_kfold(y: np.ndarray, n_splits: int = 5, n_bins: int = 10,
                         shuffle: bool = True, random_state: int = None,
                         task_type: str = 'auto'):
    """Factory function to get appropriate KFold splitter based on task type.

    Returns StratifiedKFold for classification and multiclass tasks.

    Args:
        y: Target values (used to detect task type).
        n_splits: Number of folds.
        n_bins: Number of quantile bins (unused, kept for backward compatibility).
        shuffle: Whether to shuffle before splitting.
        random_state: Random seed.
        task_type: 'auto', 'classification', or 'multiclass'.

    Returns:
        StratifiedKFold instance.

    Example:
        >>> skf = get_stratified_kfold(y_train, n_splits=5)
        >>> for train_idx, test_idx in skf.split(X, y):
        ...     # train and evaluate
    """
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )


def random_cv_search(
    init_clf_fn: Callable[[Dict], Any],
    param_grid: Dict,
    X: np.ndarray,
    y: np.ndarray,
    n_iters: int = -1,
    cv: int = 3,
    task_type: str = 'auto',
    scoring: str = 'auto',
) -> Tuple[float, Dict]:
    """Perform a random search over the hyperparameter grid for a model.

    If n_iters is -1, then the search will be exhaustive (all combinations).

    Args:
        init_clf_fn: Function to initialize the model given a parameter dict.
        param_grid: Hyperparameter grid as {param_name: [values]}.
        X: Features array.
        y: Labels/values array.
        n_iters: Number of iterations for random search. -1 for exhaustive search.
        cv: Number of cross-validation folds.
        task_type: 'auto', 'classification', or 'multiclass'. Determines scoring.
        scoring: Scoring metric for model selection.
            - 'auto': Use 'auc' for classification, 'balanced_accuracy' for multiclass
            - 'auc': ROC-AUC score (classification only)
            - 'balanced_accuracy': Balanced accuracy (classification/multiclass)
            - 'multiclass_logloss': Negative log-loss (multiclass only, higher is better)
            - 'multiclass_macro_auc': Macro-averaged one-vs-rest AUC (multiclass only)

    Returns:
        Tuple of (best_score, best_params).
    """
    # Detect task type
    task = detect_task_type(y, task_type)

    # Determine scoring metric
    if scoring == 'auto':
        if task == 'classification':
            scoring = 'auc'
        else:
            scoring = 'multiclass_macro_auc'

    # Validate scoring metric for task type
    classification_metrics = {'auc', 'balanced_accuracy'}
    multiclass_metrics = {'balanced_accuracy', 'multiclass_logloss', 'multiclass_macro_auc'}

    if task == 'classification' and scoring not in classification_metrics:
        raise ValueError(f"Scoring '{scoring}' not valid for classification. Use: {classification_metrics}")
    if task == 'multiclass' and scoring not in multiclass_metrics:
        raise ValueError(f"Scoring '{scoring}' not valid for multiclass. Use: {multiclass_metrics}")

    # Generate all parameter combinations
    param_dict_list = []
    for p in itertools.product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), p))
        param_dict_list.append(param_dict)

    random.shuffle(param_dict_list)

    if n_iters != -1:
        param_dict_list = random.sample(param_dict_list, min(n_iters, len(param_dict_list)))

    best_params = {}
    # All metrics are "higher is better" (neg_mse is already negated)
    best_score = float('-inf')

    # Extract the indices for the cross-validation folds
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    idx = np.array_split(idx, cv)

    bar = tqdm(total=len(param_dict_list))
    for param_dict in param_dict_list:
        bar.set_description(f"Best {scoring}: {best_score:.4f}")
        clf = init_clf_fn(param_dict)
        scores = []

        for j in range(cv):
            test_idx = idx[j]
            train_idx = np.concatenate([idx[k] for k in range(cv) if k != j])
            x_fit, y_fit = X[train_idx], y[train_idx]
            x_predict, y_true = X[test_idx], y[test_idx]

            # Handle GPU models
            if CUPY_AVAILABLE and getattr(clf, 'device', 'cpu') == 'cuda':
                x_fit, y_fit, x_predict = cp.array(x_fit), cp.array(y_fit), cp.array(x_predict)

            clf.fit(x_fit, y_fit)

            # Compute score based on scoring metric
            if scoring == 'auc':
                # Use predict_proba for AUC
                if hasattr(clf, 'predict_proba'):
                    y_prob = clf.predict_proba(x_predict)[:, 1]
                else:
                    # Fallback for models without predict_proba (e.g., raw scores)
                    y_prob = clf.predict(x_predict)
                score = roc_auc_score(y_true, y_prob)
            elif scoring == 'balanced_accuracy':
                y_pred = clf.predict(x_predict)
                score = balanced_accuracy_score(y_true, y_pred)
            elif scoring == 'multiclass_logloss':
                # Negative log-loss (higher is better)
                y_prob = clf.predict_proba(x_predict)
                score = -log_loss(y_true, y_prob)
            elif scoring == 'multiclass_macro_auc':
                # Macro-averaged one-vs-rest AUC
                y_prob = clf.predict_proba(x_predict)
                score = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')

            scores.append(score)

            # Free GPU memory if using CuPy
            if CUPY_AVAILABLE:
                cp._default_memory_pool.free_all_blocks()

        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_params = param_dict

        bar.update(1)

    bar.close()
    return best_score, best_params


def get_accessory_splits(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    small_splits_size: int = 10000,
    task_type: str = 'auto',
    n_quantile_bins: int = 5,
) -> Dict[str, np.ndarray]:
    """Create specialized dataset splits for different training scenarios.

    For binary classification:
        1. Balanced dataset: Equal number of positive and negative samples
        2. Small dataset: Subset for faster cross-validation
        3. Small balanced dataset: Combination of both

    For multiclass classification:
        1. Balanced dataset: Equal samples from each class
        2. Small dataset: Stratified subset for faster CV
        3. Small balanced dataset: Combination of both

    Args:
        X: Training features.
        y: Training labels/values.
        seed: Random seed for reproducibility.
        small_splits_size: Maximum size for small splits.
        task_type: 'auto', 'classification', or 'multiclass'.
        n_quantile_bins: Number of quantile bins (unused, kept for backward compatibility).

    Returns:
        Dictionary containing:
            - X_train_balanced, y_train_balanced: Balanced distribution
            - X_train_small, y_train_small: Small subset for CV
            - X_train_small_balanced, y_train_small_balanced: Small balanced subset
    """
    np.random.seed(seed)

    # Detect task type
    task = detect_task_type(y, task_type)

    if task == 'classification':
        # ======== BINARY CLASSIFICATION: Class-based balancing ========
        # 1) Balanced dataset
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]

        # Handle case where dataset is already balanced or has fewer negatives
        if len(neg_idx) >= len(pos_idx):
            # Use all positive samples and sample negatives
            neg_idx_sampled = np.random.choice(neg_idx, size=len(pos_idx), replace=False)
            pos_idx_final = pos_idx
        else:
            # Use all negative samples and sample positives
            neg_idx_sampled = neg_idx
            pos_idx_final = np.random.choice(pos_idx, size=len(neg_idx), replace=False)

        balanced_idx = np.concatenate([pos_idx_final, neg_idx_sampled])
        X_train_balanced, y_train_balanced = X[balanced_idx], y[balanced_idx]

        print(f"Shape of the balanced training set: {X_train_balanced.shape}")
        frac = np.sum(y_train_balanced == 1)
        tot = len(y_train_balanced)
        print(f"The fraction of '1' values of y is {frac}/{tot} = {frac / tot:.4f}")

        # 2) Small dataset for cross-validation
        if len(X) > small_splits_size:
            _, X_train_small, _, y_train_small = train_test_split(
                X, y, test_size=small_splits_size, random_state=seed, stratify=y
            )
        else:
            X_train_small = X
            y_train_small = y

        print(f"Shape of the small CV training set: {X_train_small.shape}")
        frac = np.sum(y_train_small == 1)
        tot = len(y_train_small)
        print(f"The fraction of '1' values of y is {frac}/{tot} = {frac / tot:.4f}")

        # 3) Small balanced dataset
        if len(X_train_balanced) > small_splits_size:
            _, X_train_small_balanced, _, y_train_small_balanced = train_test_split(
                X_train_balanced, y_train_balanced,
                test_size=small_splits_size, random_state=seed, stratify=y_train_balanced
            )
        else:
            X_train_small_balanced = X_train_balanced
            y_train_small_balanced = y_train_balanced

        print(f"Shape of the small CV balanced training set: {X_train_small_balanced.shape}")
        frac = np.sum(y_train_small_balanced == 1)
        tot = len(y_train_small_balanced)
        print(f"The fraction of '1' values of y is {frac}/{tot} = {frac / tot:.4f}")

    elif task == 'multiclass':
        # ======== MULTICLASS CLASSIFICATION: Class-based balancing ========
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)

        # Calculate minimum samples per class
        class_counts = [np.sum(y == c) for c in unique_classes]
        min_per_class = min(class_counts)

        # 1) Balanced dataset: sample equally from each class
        balanced_idx = []
        for class_id in unique_classes:
            class_idx = np.where(y == class_id)[0]
            sampled_idx = np.random.choice(class_idx, size=min_per_class, replace=False)
            balanced_idx.extend(sampled_idx)

        balanced_idx = np.array(balanced_idx)
        np.random.shuffle(balanced_idx)
        X_train_balanced = X[balanced_idx]
        y_train_balanced = y[balanced_idx]

        print(f"Shape of the class-balanced training set: {X_train_balanced.shape}")
        print(f"  {n_classes} classes, {min_per_class} samples per class")
        for c in unique_classes:
            count = np.sum(y_train_balanced == c)
            print(f"  Class {c}: {count} samples ({count/len(y_train_balanced)*100:.1f}%)")

        # 2) Small dataset for cross-validation (stratified by class)
        if len(X) > small_splits_size:
            _, X_train_small, _, y_train_small = train_test_split(
                X, y, test_size=small_splits_size, random_state=seed, stratify=y
            )
        else:
            X_train_small = X
            y_train_small = y

        print(f"Shape of the small CV training set: {X_train_small.shape}")

        # 3) Small balanced dataset
        if len(X_train_balanced) > small_splits_size:
            _, X_train_small_balanced, _, y_train_small_balanced = train_test_split(
                X_train_balanced, y_train_balanced,
                test_size=small_splits_size, random_state=seed, stratify=y_train_balanced
            )
        else:
            X_train_small_balanced = X_train_balanced
            y_train_small_balanced = y_train_balanced

        print(f"Shape of the small CV balanced training set: {X_train_small_balanced.shape}")

    return {
        'X_train_balanced': X_train_balanced,
        'y_train_balanced': y_train_balanced,
        'X_train_small': X_train_small,
        'y_train_small': y_train_small,
        'X_train_small_balanced': X_train_small_balanced,
        'y_train_small_balanced': y_train_small_balanced,
    }