"""
High-level pipeline for orchestrating model training.

Supports classification and multiclass tasks. Task type is auto-detected
from target values unless explicitly set in TrainerConfig.
"""

import os
from typing import Dict, List, Optional, Union

import numpy as np

from .trainers import (
    BaseTrainer,
    TrainerConfig,
    XGBoostTrainer,
    CatBoostTrainer,
    PRSTrainer,
    RandomForestTrainer,
    get_trainer,
)
from .results import EvaluationResults, TrainingResult
from .utils import get_accessory_splits, detect_task_type


class ModelPipeline:
    """Orchestrates training of multiple models.

    This class manages the training of multiple models and aggregates
    their results. It supports custom parameter grids per trainer and
    automatically handles balanced vs imbalanced data splits.

    Supports both classification and multiclass tasks. Task type is
    auto-detected from target values.

    Example:
        >>> config = TrainerConfig(use_gpu=True, cv_folds=3)
        >>> pipeline = ModelPipeline(
        ...     trainers={
        ...         'xgboost': XGBoostTrainer(config),
        ...         'xgboost_balanced': XGBoostTrainer(config),
        ...         'catboost': CatBoostTrainer(config),
        ...         'prs': PRSTrainer(config),
        ...     },
        ...     param_grids={
        ...         'xgboost_balanced': {'max_depth': [1, 3], 'scale_pos_weight': [1, 11]},
        ...     }
        ... )
        >>> results = pipeline.train_all(X_train, y_train, X_test, y_test)
    """

    def __init__(
        self,
        trainers: Dict[str, BaseTrainer],
        param_grids: Optional[Dict[str, Dict]] = None,
        compute_splits: bool = True,
        seed: int = 42,
    ):
        """Initialize the pipeline.

        Args:
            trainers: Dictionary mapping model names to trainer instances.
            param_grids: Optional dictionary mapping model names to custom
                hyperparameter grids. If not provided for a model, uses
                the trainer's default_param_grid.
            compute_splits: Whether to automatically compute accessory splits
                if not provided to train_all().
            seed: Random seed for reproducibility.
        """
        self.trainers = trainers
        self.param_grids = param_grids or {}
        self.compute_splits = compute_splits
        self.seed = seed

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train_balanced: np.ndarray = None,
        y_train_balanced: np.ndarray = None,
        X_train_small: np.ndarray = None,
        y_train_small: np.ndarray = None,
        X_train_small_balanced: np.ndarray = None,
        y_train_small_balanced: np.ndarray = None,
        save_path: Optional[str] = None,
        metrics_dict: Optional[Dict] = None,
    ) -> EvaluationResults:
        """Train all configured models.

        Args:
            X_train: Full training features.
            y_train: Full training labels/values.
            X_test: Test features.
            y_test: Test labels/values.
            X_train_balanced: Balanced training features (optional).
            y_train_balanced: Balanced training labels/values (optional).
            X_train_small: Small training set for CV (optional).
            y_train_small: Small training labels/values for CV (optional).
            X_train_small_balanced: Small balanced training features (optional).
            y_train_small_balanced: Small balanced training labels/values (optional).
            save_path: Optional path to save trained models.
            metrics_dict: Optional dict to log metrics to.

        Returns:
            EvaluationResults containing all trained models and metrics.
        """
        # Compute accessory splits if not provided
        if self.compute_splits and X_train_small is None:
            splits = get_accessory_splits(X_train, y_train, seed=self.seed)
            X_train_balanced = splits['X_train_balanced']
            y_train_balanced = splits['y_train_balanced']
            X_train_small = splits['X_train_small']
            y_train_small = splits['y_train_small']
            X_train_small_balanced = splits['X_train_small_balanced']
            y_train_small_balanced = splits['y_train_small_balanced']

        if save_path:
            os.makedirs(save_path, exist_ok=True)

        results = EvaluationResults()

        for name, trainer in self.trainers.items():
            # Determine which splits to use based on trainer name
            if 'balanced' in name.lower():
                X_tr, y_tr = X_train_balanced, y_train_balanced
                X_tr_small, y_tr_small = X_train_small_balanced, y_train_small_balanced
            else:
                X_tr, y_tr = X_train, y_train
                X_tr_small, y_tr_small = X_train_small, y_train_small

            # Get custom param_grid if provided
            param_grid = self.param_grids.get(name, None)

            # Train the model
            if isinstance(trainer, PRSTrainer):
                # PRS returns multiple results
                prs_results = trainer.train(
                    X_train=X_tr,
                    y_train=y_tr,
                    X_test=X_test,
                    y_test=y_test,
                    metrics_dict=metrics_dict,
                    save_path=save_path,
                )
                for variant_name, result in prs_results.items():
                    results.add_result(variant_name, result)
                    # Task-aware printing
                    if 'macro_auc' in result.metrics and 'spearman_r' in result.metrics:  # Multiclass ordinal
                        print(f"  -> {variant_name}: Macro AUC={result.metrics['macro_auc']:.4f}, Spearman r={result.metrics['spearman_r']:.4f}")
                    elif 'log_loss' in result.metrics:  # Multiclass multinomial
                        print(f"  -> {variant_name}: Log-Loss={result.metrics['log_loss']:.4f}, Macro AUC={result.metrics['macro_auc']:.4f}")
                    else:  # Binary classification
                        print(f"  -> {variant_name}: Balanced Acc={result.metrics['balanced_accuracy']:.4f}, ROC-AUC={result.metrics['roc_auc']:.4f}")
            else:
                result = trainer.train(
                    X_train=X_tr,
                    y_train=y_tr,
                    X_train_small=X_tr_small,
                    y_train_small=y_tr_small,
                    X_test=X_test,
                    y_test=y_test,
                    param_grid=param_grid,
                    metrics_dict=metrics_dict,
                    save_path=save_path,
                    name=name,
                )
                results.add_result(name, result)
                # Task-aware printing
                if 'log_loss' in result.metrics:  # Multiclass
                    print(f"  -> {name}: Log-Loss={result.metrics['log_loss']:.4f}, Macro AUC={result.metrics['macro_auc']:.4f}")
                else:  # Binary classification
                    print(f"  -> {name}: Balanced Acc={result.metrics['balanced_accuracy']:.4f}, ROC-AUC={result.metrics['roc_auc']:.4f}")

        return results

    def to_dict(self, results: EvaluationResults) -> Dict:
        """Convert EvaluationResults to backward-compatible dict format.

        Args:
            results: EvaluationResults from train_all().

        Returns:
            Dictionary with format: {model_name: {'clf': model, 'metrics': {...}}}
        """
        return results.to_dict()


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metrics_dict: Optional[Dict] = None,
    save_path: Optional[str] = None,
    on_gpu: bool = False,
    seed: int = 42,
    models: Optional[List[str]] = None,
    task_type: str = 'auto',
) -> Dict:
    """Train multiple models - functional interface using ModelPipeline internally.

    This function provides a simple interface for training multiple models.
    It automatically computes accessory splits (balanced, small) internally.
    Fresh trainer instances are created each call to ensure no stale state
    between CV folds.

    Supports classification and multiclass tasks. Task type is auto-detected
    from target values unless explicitly set.

    Args:
        X_train: Full training features.
        y_train: Full training labels/values.
        X_test: Test features.
        y_test: Test labels/values.
        metrics_dict: Optional dict to log metrics to.
        save_path: Optional path to save trained models.
        on_gpu: Whether to use GPU acceleration.
        seed: Random seed.
        models: List of models to train. If None, trains all default models.
        task_type: 'auto' (detect), 'classification', or 'multiclass'.

    Returns:
        Dictionary with model results in the original format:
        {model_name: {'clf': model, 'metrics': {...}}}
    """
    # Default models if not specified
    if models is None:
        models = ['xgboost', 'xgboost_balanced', 'catboost', 'prs']

    # Detect task type for param grid selection
    detected_task = detect_task_type(y_train, task_type)

    # Create fresh trainer configs with task_type
    config_gpu = TrainerConfig(
        use_gpu=on_gpu, random_state=seed, task_type=task_type,
    )
    config_cpu = TrainerConfig(
        use_gpu=False, random_state=seed, task_type=task_type,
    )

    # Build trainers dict based on requested models
    trainers = {}
    param_grids = {}

    if 'xgboost' in models:
        trainers['xgboost'] = XGBoostTrainer(config_gpu)

    if 'xgboost_balanced' in models:
        trainers['xgboost_balanced'] = XGBoostTrainer(config_gpu)
        # Only add scale_pos_weight for binary classification
        if detected_task == 'classification':
            param_grids['xgboost_balanced'] = {
                "max_depth": [1, 3, 6],
                "n_estimators": [100, 500, 1000],
                "learning_rate": [0.01, 0.1],
                "scale_pos_weight": [1, 11],
            }
        else:  # multiclass doesn't use scale_pos_weight
            param_grids['xgboost_balanced'] = {
                "max_depth": [1, 3, 6],
                "n_estimators": [100, 500, 1000],
                "learning_rate": [0.01, 0.1],
            }

    if 'catboost' in models:
        trainers['catboost'] = CatBoostTrainer(config_gpu)

    if 'prs' in models:
        trainers['prs'] = PRSTrainer(config_gpu)

    if 'random_forest' in models:
        # RandomForest uses CPU to avoid bad results on GPU (investigate?)
        trainers['random_forest'] = RandomForestTrainer(config_cpu)

    # Create pipeline with compute_splits=True to auto-compute accessory splits
    pipeline = ModelPipeline(
        trainers=trainers,
        param_grids=param_grids,
        compute_splits=True,
        seed=seed,
    )

    # Train all models (splits computed internally)
    results = pipeline.train_all(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        save_path=save_path,
        metrics_dict=metrics_dict,
    )

    # Convert to backward-compatible dict format
    return pipeline.to_dict(results)
