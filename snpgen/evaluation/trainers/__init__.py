"""
Trainers module for model training implementations.

Supports classification and multiclass tasks. Task type is auto-detected
from target values unless explicitly set in TrainerConfig.
"""

from .base import BaseTrainer, TrainerConfig
from .random_forest import RandomForestTrainer
from .xgboost import XGBoostTrainer
from .catboost import CatBoostTrainer
from .prs import (
    PRSTrainer,
    fit_univariate_logistic_regression,
    fit_univariate_regression,
    fit_univariate_multinomial_regression,
    get_univariate_coefficients,
    get_univariate_coefficients_multiclass,
)


def get_trainer(model_type: str, config: TrainerConfig = None) -> BaseTrainer:
    """Factory function to create trainers by name.

    Args:
        model_type: Type of trainer to create. One of:
            'random_forest', 'xgboost', 'catboost', 'prs'
        config: Optional TrainerConfig instance.

    Returns:
        Initialized trainer instance.

    Raises:
        ValueError: If model_type is not recognized.
    """
    trainers = {
        'random_forest': RandomForestTrainer,
        'xgboost': XGBoostTrainer,
        'catboost': CatBoostTrainer,
        'prs': PRSTrainer,
    }

    if model_type not in trainers:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available types: {list(trainers.keys())}"
        )

    return trainers[model_type](config)


__all__ = [
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
]
