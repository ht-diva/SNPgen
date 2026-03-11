"""
CatBoost trainer implementation.
"""

from typing import Any, Dict, Optional

import numpy as np
from catboost import CatBoostClassifier

from .base import BaseTrainer, TrainerConfig
from ..utils import detect_task_type

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class CatBoostTrainer(BaseTrainer):
    """Trainer for CatBoost with optional GPU support.

    Supports classification (binary and multiclass) tasks.
    Task type is auto-detected from target values unless explicitly set.

    For multiclass tasks, uses CatBoostClassifier with auto class weights.
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)

    @property
    def name(self) -> str:
        return "catboost"

    @property
    def default_param_grid(self) -> Dict[str, Any]:
        grid = {
            "depth": [1, 3],
            "learning_rate": [0.01, 0.1],
            "iterations": [100, 300],
            "l2_leaf_reg": [1, 5],
            "border_count": [32, 128],
        }
        # subsample is not available on GPU
        if not self.config.use_gpu:
            grid["subsample"] = [0.5, 1.0]
        return grid

    def _create_model(self, params: Dict, task_type: str = 'classification') -> CatBoostClassifier:
        gpu_task_type = 'GPU' if self.config.use_gpu else 'CPU'

        return CatBoostClassifier(
            verbose=0,
            auto_class_weights="Balanced",
            task_type=gpu_task_type,
            **params
        )

    def _fit_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> Any:
        model.fit(X, y)
        return model

    def _predict_proba(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get probability predictions (for classification/multiclass).

        For binary classification: returns P(class=1) as 1D array.
        For multiclass: returns full probability matrix (n_samples, n_classes).
        """
        proba = model.predict_proba(X)
        # Free GPU memory if using GPU
        if self.config.use_gpu and CUPY_AVAILABLE:
            cp._default_memory_pool.free_all_blocks()

        # Binary: return P(class=1) for backward compatibility
        if proba.shape[1] == 2 and self._task_type == 'classification':
            return proba[:, 1]
        # Multiclass: return all probabilities
        return proba

    def _predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get raw predictions."""
        pred = model.predict(X)
        # Free GPU memory if using GPU
        if self.config.use_gpu and CUPY_AVAILABLE:
            cp._default_memory_pool.free_all_blocks()
        return pred

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_train_small: np.ndarray,
        y_train_small: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        param_grid: Optional[Dict] = None,
        metrics_dict: Optional[Dict] = None,
        save_path: Optional[str] = None,
        name: Optional[str] = None,
    ):
        result = super().train(
            X_train, y_train, X_train_small, y_train_small,
            X_test, y_test, param_grid, metrics_dict, save_path, name
        )

        # Free GPU memory after training
        if self.config.use_gpu and CUPY_AVAILABLE:
            cp._default_memory_pool.free_all_blocks()

        return result
