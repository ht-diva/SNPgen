"""
Base trainer abstract class and configuration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import os
import time

import numpy as np
import joblib

from ..metrics import compute_metrics, compute_multiclass_metrics
from ..results import TrainingResult, log_metrics
from ..utils import random_cv_search, detect_task_type


@dataclass
class TrainerConfig:
    """Configuration for model trainer.

    Attributes:
        use_gpu: Whether to use GPU acceleration.
        hyperparameter_search: Whether to perform hyperparameter search.
        cv_folds: Number of cross-validation folds for hyperparameter search.
        n_cv_iters: Number of CV iterations (-1 for exhaustive search).
        verbose: Whether to print progress information.
        random_state: Random seed for reproducibility.
        task_type: Task type ('auto', 'classification', 'multiclass').
        cv_scoring: Scoring metric for CV hyperparameter search.
            - 'auto': Use 'auc' for classification, 'multiclass_macro_auc' for multiclass
            - 'auc': ROC-AUC score (classification only)
            - 'balanced_accuracy': Balanced accuracy (classification/multiclass)
            - 'multiclass_logloss': Negative log-loss (multiclass only)
            - 'multiclass_macro_auc': Macro-averaged AUC (multiclass only)
    """
    use_gpu: bool = False
    hyperparameter_search: bool = True
    cv_folds: int = 3
    n_cv_iters: int = -1  # -1 for exhaustive search
    verbose: bool = True
    random_state: int = 42
    task_type: str = 'auto'  # 'auto', 'classification', 'multiclass'
    cv_scoring: str = 'auto'  # 'auto', 'auc', 'balanced_accuracy', 'multiclass_logloss'


class BaseTrainer(ABC):
    """Abstract base class for model trainers.

    All trainer implementations should inherit from this class and implement
    the abstract methods.

    Supports both classification and multiclass tasks. Task type is automatically
    detected from target values unless explicitly set in config.

    Attributes:
        config: TrainerConfig instance with training settings.
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        """Initialize the trainer.

        Args:
            config: TrainerConfig instance. If None, uses default config.
        """
        self.config = config or TrainerConfig()
        self._model = None
        self._task_type = None  # Detected during training

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for logging and identification."""
        pass

    @property
    @abstractmethod
    def default_param_grid(self) -> Dict[str, Any]:
        """Default hyperparameter search grid."""
        pass

    @abstractmethod
    def _create_model(self, params: Dict, task_type: str = 'classification') -> Any:
        """Create a model instance with the given parameters.

        Args:
            params: Dictionary of model parameters.
            task_type: 'classification' or 'multiclass'.

        Returns:
            Initialized model instance.
        """
        pass

    @abstractmethod
    def _fit_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> Any:
        """Fit the model on training data.

        Args:
            model: Model instance to train.
            X: Training features.
            y: Training labels.

        Returns:
            Trained model.
        """
        pass

    @abstractmethod
    def _predict_proba(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get probability predictions from the model (for classification).

        Args:
            model: Trained model.
            X: Features to predict.

        Returns:
            Probability predictions for class 1.
        """
        pass

    @abstractmethod
    def _predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get raw predictions from the model.

        Args:
            model: Trained model.
            X: Features to predict.

        Returns:
            Raw predictions.
        """
        pass

    def _get_init_clf_fn(self, param_grid: Dict, task_type: str) -> callable:
        """Get a function that initializes the model for CV search.

        Args:
            param_grid: Parameter grid for search.
            task_type: 'classification' or 'multiclass'.

        Returns:
            Function that takes params dict and returns model instance.
        """
        return lambda p: self._create_model(p, task_type=task_type)

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
    ) -> TrainingResult:
        """Main training pipeline.

        1. Detects task type (classification/multiclass)
        2. Performs hyperparameter search (if enabled)
        3. Trains final model
        4. Makes predictions
        5. Computes appropriate metrics

        Args:
            X_train: Full training features.
            y_train: Full training labels/values.
            X_train_small: Small training set for CV search.
            y_train_small: Small training labels/values for CV search.
            X_test: Test features.
            y_test: Test labels/values.
            param_grid: Custom parameter grid (uses default if None).
            metrics_dict: Optional dict to log metrics to.
            save_path: Optional path to save the trained model.
            name: Optional custom name for this training run.

        Returns:
            TrainingResult containing model, predictions, and metrics.
        """
        model_name = name or self.name

        # Detect task type
        self._task_type = detect_task_type(y_train, self.config.task_type)

        if self.config.verbose:
            device_str = "GPU" if self.config.use_gpu else "CPU"
            print(f"\nTraining {model_name} ({device_str}, {self._task_type})...")

        start_time = time.time()

        # Use provided param_grid or default
        search_grid = param_grid or self.default_param_grid

        # Hyperparameter search
        best_params = {}
        if self.config.hyperparameter_search and search_grid:
            init_clf_fn = self._get_init_clf_fn(search_grid, self._task_type)
            best_score, best_params = random_cv_search(
                init_clf_fn,
                search_grid,
                X_train_small,
                y_train_small,
                n_iters=self.config.n_cv_iters,
                cv=self.config.cv_folds,
                task_type=self._task_type,
                scoring=self.config.cv_scoring,
            )
            if self.config.verbose:
                print(f"  Best parameters found: {best_params}")
                print(f"  Best score found: {best_score:.4f}")
        else:
            best_params = {}

        # Create and fit final model
        model = self._create_model(best_params, task_type=self._task_type)
        model = self._fit_model(model, X_train, y_train)
        self._model = model

        # Make predictions based on task type
        if self._task_type == 'classification':
            y_score = self._predict_proba(model, X_test)
            metrics = compute_metrics(y_score, y_test)
        elif self._task_type == 'multiclass':
            y_score = self._predict_proba(model, X_test)  # (n_samples, n_classes)
            metrics = compute_multiclass_metrics(y_score, y_test)

        training_time = time.time() - start_time

        # Log metrics if container provided
        if metrics_dict is not None:
            log_metrics(metrics_dict, model_name, metrics)

        # Save model if path provided
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            joblib.dump(model, os.path.join(save_path, model_name))

        if self.config.verbose:
            print("Done.")

        return TrainingResult(
            model_name=model_name,
            model=model,
            y_score=y_score,
            y_test=y_test,
            metrics=metrics,
            best_params=best_params,
            training_time=training_time,
        )

    @property
    def model(self) -> Any:
        """Get the trained model."""
        if self._model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self._model

    @property
    def task_type(self) -> Optional[str]:
        """Get the detected task type."""
        return self._task_type
