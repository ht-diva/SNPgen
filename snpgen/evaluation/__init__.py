"""
SNPgen Evaluation Module

This module provides tools for training and evaluating machine learning models
on genetic data, particularly for comparing real vs synthetic datasets.

Supports classification (binary) and multiclass tasks.
Task type is auto-detected from target values unless explicitly set.

Main components:
- Metrics: Functions for computing classification and multiclass metrics
- Trainers: Class-based trainers for different ML models (XGBoost, CatBoost, PRS, etc.)
- Pipeline: High-level orchestration for training multiple models
- Results: Dataclasses and utilities for handling evaluation results
- Plotting: Visualization functions for metrics comparison
- Analysis: Filtering and analysis utilities
- Utils: Task detection, data splitting, and cross-validation utilities

Example usage:
    >>> from snpgen.evaluation import (
    ...     ModelPipeline,
    ...     TrainerConfig,
    ...     XGBoostTrainer,
    ...     CatBoostTrainer,
    ...     PRSTrainer,
    ...     get_accessory_splits,
    ...     build_multiindex_df,
    ...     plot_metrics_with_ci,
    ... )
    >>>
    >>> # Configure trainers (task_type='auto' detects classification vs multiclass)
    >>> config = TrainerConfig(use_gpu=True, cv_folds=3, task_type='auto')
    >>> pipeline = ModelPipeline({
    ...     'xgboost': XGBoostTrainer(config),
    ...     'catboost': CatBoostTrainer(config),
    ...     'prs': PRSTrainer(config),
    ... })
    >>>
    >>> # Train models (works with binary and multiclass targets)
    >>> results = pipeline.train_all(X_train, y_train, X_test, y_test)
"""

# Metrics
from .metrics import (
    compute_metrics,
    compute_relevant_metrics,
    compute_prs_metrics,
    # Multiclass metrics
    compute_multiclass_metrics,
    compute_relevant_multiclass_metrics,
    compute_mcfadden_pseudo_r2,
    compute_prs_multiclass_metrics,
)

# Trainers
from .trainers import (
    BaseTrainer,
    TrainerConfig,
    RandomForestTrainer,
    XGBoostTrainer,
    CatBoostTrainer,
    PRSTrainer,
    get_trainer,
    # PRS utilities
    fit_univariate_logistic_regression,
    fit_univariate_regression,
    fit_univariate_multinomial_regression,
    get_univariate_coefficients,
    get_univariate_coefficients_multiclass,
)

# Pipeline
from .pipeline import (
    ModelPipeline,
    train_models,
)

# Results
from .results import (
    MetricsResult,
    TrainingResult,
    EvaluationResults,
    build_multiindex_df,
    log_metrics,
    process_cv_results,
    save_cv_results,
    load_cv_results,
    check_results_exist,
    compute_gwas_prs_results,
    # Incremental CV training utilities
    verify_cv_indices,
    get_missing_trainers,
    merge_fold_results,
)

# Analysis
from .analysis import (
    filter_model_list,
)

# Plotting
from .plotting import (
    set_bold,
    add_value_labels,
    calculate_ci,
    plot_metrics_with_ci,
    # Task detection and default metrics
    detect_task_type_from_df,
    get_default_metrics,
    CLASSIFICATION_METRICS,
    CLASSIFICATION_METRIC_PRETTY_NAMES,
    MULTICLASS_METRICS,
    MULTICLASS_METRIC_PRETTY_NAMES,
    ALL_METRIC_PRETTY_NAMES,
    # Multiclass plotting
    plot_confusion_matrix,
    plot_class_proportions_by_decile,
    plot_ordinal_trend_by_decile,
    plot_multiclass_roc_curves,
)

# Utils
from .utils import (
    random_cv_search,
    get_accessory_splits,
    # Task detection utilities
    detect_task_type,
    get_stratified_kfold,
)


__all__ = [
    # Metrics (classification)
    "compute_metrics",
    "compute_relevant_metrics",
    "compute_prs_metrics",
    # Metrics (multiclass)
    "compute_multiclass_metrics",
    "compute_relevant_multiclass_metrics",
    "compute_mcfadden_pseudo_r2",
    "compute_prs_multiclass_metrics",
    # Trainers
    "BaseTrainer",
    "TrainerConfig",
    "RandomForestTrainer",
    "XGBoostTrainer",
    "CatBoostTrainer",
    "PRSTrainer",
    "get_trainer",
    "fit_univariate_logistic_regression",
    "fit_univariate_regression",
    "fit_univariate_multinomial_regression",
    "get_univariate_coefficients",
    "get_univariate_coefficients_multiclass",
    # Pipeline
    "ModelPipeline",
    "train_models",
    # Results
    "MetricsResult",
    "TrainingResult",
    "EvaluationResults",
    "build_multiindex_df",
    "log_metrics",
    "process_cv_results",
    "save_cv_results",
    "load_cv_results",
    "check_results_exist",
    "compute_gwas_prs_results",
    # Incremental CV training utilities
    "verify_cv_indices",
    "get_missing_trainers",
    "merge_fold_results",
    # Analysis
    "filter_model_list",
    # Plotting
    "set_bold",
    "add_value_labels",
    "calculate_ci",
    "plot_metrics_with_ci",
    "detect_task_type_from_df",
    "get_default_metrics",
    "CLASSIFICATION_METRICS",
    "CLASSIFICATION_METRIC_PRETTY_NAMES",
    "MULTICLASS_METRICS",
    "MULTICLASS_METRIC_PRETTY_NAMES",
    "ALL_METRIC_PRETTY_NAMES",
    # Multiclass plotting
    "plot_confusion_matrix",
    "plot_class_proportions_by_decile",
    "plot_ordinal_trend_by_decile",
    "plot_multiclass_roc_curves",
    # Utils
    "random_cv_search",
    "get_accessory_splits",
    "detect_task_type",
    "get_stratified_kfold",
]
