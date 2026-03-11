"""
Random Forest trainer implementation.
"""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import BaseTrainer, TrainerConfig
from ..utils import detect_task_type

# Try to import cuML for GPU support
try:
    from cuml.ensemble import RandomForestClassifier as RandomForestClassifier_cu
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False


class RandomForestTrainer(BaseTrainer):
    """Trainer for Random Forest with optional GPU support via cuML.

    Supports classification (RandomForestClassifier).
    Task type is auto-detected from target values unless explicitly set.
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self._use_gpu_actual = False

    @property
    def name(self) -> str:
        return "random_forest"

    @property
    def default_param_grid(self) -> Dict[str, Any]:
        grid = {
            "n_estimators": [100, 500, 1000],
            "max_depth": [1, 20],
            "min_samples_split": [2, 10],
            "min_samples_leaf": [1, 16, 24, 32],
        }
        # class_weight is not available in cuML implementation and only for classification
        if not self._use_gpu_actual and self._task_type == 'classification':
            grid["class_weight"] = ["balanced", "balanced_subsample", None]
        return grid

    def _create_model(self, params: Dict, task_type: str = 'classification') -> Any:
        # Remove class_weight if present (not supported in cuML)
        if self._use_gpu_actual:
            params = {k: v for k, v in params.items() if k != 'class_weight'}

        if self._use_gpu_actual and CUML_AVAILABLE:
            return RandomForestClassifier_cu(
                random_state=self.config.random_state,
                n_streams=1,
                **params
            )
        else:
            return RandomForestClassifier(
                random_state=self.config.random_state,
                n_jobs=-1,
                **params
            )

    def _fit_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> Any:
        model.fit(X, y)
        return model

    def _predict_proba(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get probability predictions (for classification)."""
        return model.predict_proba(X)[:, 1]

    def _predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get raw predictions."""
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
        # Detect task type early for default_param_grid
        self._task_type = detect_task_type(y_train, self.config.task_type)

        # Check GPU availability
        if self.config.use_gpu and not CUML_AVAILABLE:
            print("  GPU Model requested but cuML is not available. "
                  "Install cuML from https://docs.rapids.ai/install/. "
                  "Falling back on CPU implementation")
            self._use_gpu_actual = False
        else:
            self._use_gpu_actual = self.config.use_gpu

        return super().train(
            X_train, y_train, X_train_small, y_train_small,
            X_test, y_test, param_grid, metrics_dict, save_path, name
        )
