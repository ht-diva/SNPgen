"""
XGBoost trainer implementation.
"""

from typing import Any, Dict, Optional

import numpy as np
import xgboost as xgb

from .base import BaseTrainer, TrainerConfig
from ..utils import detect_task_type

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class XGBoostTrainer(BaseTrainer):
    """Trainer for XGBoost with optional GPU support.

    Supports classification (binary and multiclass) tasks.
    Task type is auto-detected from target values unless explicitly set.

    For multiclass tasks, uses XGBClassifier with softmax objective.
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self._scale_pos_weight = None

    @property
    def name(self) -> str:
        return "xgboost"

    @property
    def default_param_grid(self) -> Dict[str, Any]:
        grid = {
            "max_depth": [1, 2, 3, 6, 20],
            "n_estimators": [100, 500, 700, 800, 1000],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
        }
        # Add scale_pos_weight only for classification and if we have the class ratio
        if self._task_type == 'classification' and self._scale_pos_weight is not None:
            grid["scale_pos_weight"] = [self._scale_pos_weight]
        return grid

    def _create_model(self, params: Dict, task_type: str = 'classification') -> xgb.XGBClassifier:
        device = 'cuda' if self.config.use_gpu else 'cpu'

        return xgb.XGBClassifier(
            use_label_encoder=False,
            n_jobs=-1,
            random_state=self.config.random_state,
            device=device,
            **params
        )

    def _fit_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> Any:
        if self.config.use_gpu and CUPY_AVAILABLE:
            X_fit = cp.array(X)
            y_fit = cp.array(y)
        else:
            X_fit = X
            y_fit = y

        model.fit(X_fit, y_fit)
        return model

    def _predict_proba(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get probability predictions (for classification/multiclass).

        For binary classification: returns P(class=1) as 1D array.
        For multiclass: returns full probability matrix (n_samples, n_classes).
        """
        if self.config.use_gpu and CUPY_AVAILABLE:
            X_pred = cp.array(X)
            proba = model.predict_proba(X_pred)
            # Free GPU memory
            cp._default_memory_pool.free_all_blocks()
        else:
            proba = model.predict_proba(X)

        # Binary: return P(class=1) for backward compatibility
        if proba.shape[1] == 2 and self._task_type == 'classification':
            return proba[:, 1]
        # Multiclass: return all probabilities
        return proba

    def _predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get raw predictions."""
        if self.config.use_gpu and CUPY_AVAILABLE:
            X_pred = cp.array(X)
            pred = model.predict(X_pred)
            # Free GPU memory
            cp._default_memory_pool.free_all_blocks()
            return pred
        return model.predict(X)

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
        # Detect task type early to set scale_pos_weight only for binary classification
        self._task_type = detect_task_type(y_train, self.config.task_type)

        if self._task_type == 'classification':
            # Compute scale_pos_weight from training data (binary only)
            self._scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
        else:
            # Multiclass doesn't use scale_pos_weight
            self._scale_pos_weight = None

        result = super().train(
            X_train, y_train, X_train_small, y_train_small,
            X_test, y_test, param_grid, metrics_dict, save_path, name
        )

        # Free GPU memory after training
        if self.config.use_gpu and CUPY_AVAILABLE:
            cp._default_memory_pool.free_all_blocks()

        return result
