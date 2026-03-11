"""
Polygenic Risk Score (PRS) trainer implementation using univariate regression.

Supports classification (logistic regression) and multiclass (multinomial logistic regression).
"""

from typing import Any, Dict, Optional
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import joblib
from tqdm.auto import tqdm
from glum import GeneralizedLinearRegressor
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax

from .base import BaseTrainer, TrainerConfig
from ..metrics import (
    compute_metrics, compute_prs_metrics, compute_multiclass_metrics,
    compute_prs_multiclass_metrics
)
from ..results import TrainingResult, log_metrics
from ..utils import detect_task_type


def fit_univariate_regression(
    X: np.ndarray,
    y: np.ndarray,
    column_index: int,
    family: str = 'binomial'
) -> float:
    """Fit univariate regression on a single column/SNP.

    Args:
        X: Feature matrix.
        y: Labels/values.
        column_index: Index of the column to fit.
        family: GLM family - 'binomial' for classification, 'gaussian' for regression.

    Returns:
        Coefficient for the feature.
    """
    model = GeneralizedLinearRegressor(
        family=family,
        alpha=0,
    )

    # Fit model on one feature
    model.fit(X[:, column_index].reshape(-1, 1), y)

    # Return the coefficient for the feature
    return model.coef_[0]


# Backward-compatible alias
def fit_univariate_logistic_regression(X: np.ndarray, y: np.ndarray, column_index: int) -> float:
    """Fit logistic regression on a single column/SNP.

    Backward-compatible alias for fit_univariate_regression with family='binomial'.
    """
    return fit_univariate_regression(X, y, column_index, family='binomial')


def fit_univariate_multinomial_regression(
    X: np.ndarray,
    y: np.ndarray,
    column_index: int,
    n_classes: int
) -> np.ndarray:
    """Fit multinomial logistic regression on a single column/SNP.

    Args:
        X: Feature matrix.
        y: Integer class labels (0 to n_classes-1).
        column_index: Index of the column to fit.
        n_classes: Number of classes.

    Returns:
        Coefficient array of shape (n_classes,) for the feature.
    """
    model = LogisticRegression(
        solver='lbfgs',
        penalty=None,  # No regularization
        max_iter=1000,
    )

    # Fit model on one feature
    model.fit(X[:, column_index].reshape(-1, 1), y)

    # Return coefficients for all classes - shape (n_classes,)
    return model.coef_.flatten()


def get_univariate_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    family: str = 'binomial',
    parallel: bool = True,
    max_workers: Optional[int] = None
) -> np.ndarray:
    """Fit univariate regression models for all SNPs.

    Args:
        X: Feature matrix (samples x SNPs).
        y: Labels/values.
        family: GLM family - 'binomial' for classification, 'gaussian' for regression.
        parallel: Whether to use parallel processing.
        max_workers: Maximum number of worker threads.

    Returns:
        Array of coefficients (betas) for all SNPs.
    """
    D = X.shape[1]
    coefficients = np.zeros(D)  # Preallocate space for coefficients

    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fit_univariate_regression, X, y, i, family): i
                for i in range(D)
            }
            with tqdm(total=D, desc="Processing features") as pbar:
                for future in as_completed(futures):
                    column_index = futures[future]
                    coefficients[column_index] = future.result()
                    pbar.update(1)
            executor.shutdown(wait=True)
    else:
        for i in tqdm(range(D), desc="Processing features"):
            coefficients[i] = fit_univariate_regression(X, y, i, family)

    return coefficients


def get_univariate_coefficients_multiclass(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    parallel: bool = True,
    max_workers: Optional[int] = None
) -> np.ndarray:
    """Fit univariate multinomial regression models for all SNPs.

    Args:
        X: Feature matrix (samples x SNPs).
        y: Integer class labels (0 to n_classes-1).
        n_classes: Number of classes.
        parallel: Whether to use parallel processing.
        max_workers: Maximum number of worker threads.

    Returns:
        Coefficient matrix of shape (D, n_classes) for all SNPs.
    """
    D = X.shape[1]
    coefficients = np.zeros((D, n_classes))

    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fit_univariate_multinomial_regression, X, y, i, n_classes): i
                for i in range(D)
            }
            with tqdm(total=D, desc="Processing features (multiclass)") as pbar:
                for future in as_completed(futures):
                    column_index = futures[future]
                    coefficients[column_index] = future.result()
                    pbar.update(1)
            executor.shutdown(wait=True)
    else:
        for i in tqdm(range(D), desc="Processing features (multiclass)"):
            coefficients[i] = fit_univariate_multinomial_regression(X, y, i, n_classes)

    return coefficients


class PRSTrainer(BaseTrainer):
    """Trainer for Polygenic Risk Score using univariate regression.

    Supports classification and multiclass tasks:
    - Binary classification: Uses logistic regression (family='binomial')
    - Multiclass: Uses multinomial logistic regression

    Unlike other trainers, PRS doesn't use hyperparameter search.
    It fits univariate regression for each SNP and uses the
    coefficients (betas) to compute a weighted sum as the PRS.

    For multiclass:
        - Betas matrix has shape (D, n_classes)
        - Predictions use softmax over class scores
    """

    def __init__(
        self,
        config: Optional[TrainerConfig] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
    ):
        """Initialize PRS trainer.

        Args:
            config: TrainerConfig instance.
            parallel: Whether to use parallel processing for fitting.
            max_workers: Maximum number of worker threads.
        """
        super().__init__(config)
        self.parallel = parallel
        self.max_workers = max_workers
        self._betas = None

    @property
    def name(self) -> str:
        return "prs univariate"

    @property
    def default_param_grid(self) -> Dict[str, Any]:
        # PRS doesn't use hyperparameter search
        return {}

    def _create_model(self, params: Dict, task_type: str = 'classification') -> None:
        # PRS doesn't create a traditional model
        return None

    def _fit_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # For PRS, "fitting" means computing univariate coefficients
        # This is handled in train() to pass the family parameter
        return None

    def _predict_proba(self, model: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Get probability-like predictions (for classification/multiclass).

        For binary classification: returns scaled PRS as P(class=1).
        For multiclass: returns softmax probabilities (n_samples, n_classes).
        """
        # model is actually betas array
        prs = np.dot(X, model)

        # Check if multiclass (betas has shape (D, n_classes))
        if prs.ndim == 2:
            # Multiclass: apply softmax to get probabilities
            return softmax(prs, axis=1)
        else:
            # Binary: scale to [0, 1] for probability-like output
            prs_scaled = (prs - prs.min()) / (prs.max() - prs.min())
            return prs_scaled

    def _predict(self, model: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Get raw predictions."""
        # model is actually betas array
        prs = np.dot(X, model)
        return prs

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_train_small: np.ndarray = None,  # Not used for PRS
        y_train_small: np.ndarray = None,  # Not used for PRS
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
        param_grid: Optional[Dict] = None,  # Not used for PRS
        metrics_dict: Optional[Dict] = None,
        save_path: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, TrainingResult]:
        """Train PRS and compute metrics for multiple quantile thresholds.

        Unlike other trainers, PRS returns multiple TrainingResults for
        different quantile thresholds.

        For classification: Uses logistic regression and binary classification metrics.
        For multiclass: Uses multinomial logistic regression and multiclass metrics.

        Args:
            X_train: Training features.
            y_train: Training labels (discrete for classification/multiclass).
            X_train_small: Not used for PRS.
            y_train_small: Not used for PRS.
            X_test: Test features.
            y_test: Test labels (discrete for classification/multiclass).
            param_grid: Not used for PRS.
            metrics_dict: Dictionary to log metrics.
            save_path: Path to save model.
            name: Model name override.

        Returns:
            Dictionary mapping metric name to TrainingResult.
        """
        model_name = name or self.name

        # Detect task type
        self._task_type = detect_task_type(y_train, self.config.task_type)

        if self.config.verbose:
            if self._task_type == 'classification':
                regression_type = "Logistic"
            else:
                regression_type = "Multinomial Logistic"
            print(f"\nTraining {model_name} ({self._task_type})...")
            print(f"  Fitting Univariate {regression_type} Regression model...")

        start_time = time.time()

        # Fit univariate models to get betas
        if self._task_type == 'multiclass':
            # Fit using multinomial logistic regression on discrete labels
            n_classes = len(np.unique(y_train))
            betas = get_univariate_coefficients_multiclass(
                X_train, y_train.astype(int),
                n_classes=n_classes,
                parallel=self.parallel,
                max_workers=self.max_workers
            )
        else:
            betas = get_univariate_coefficients(
                X_train, y_train,
                family='binomial',
                parallel=self.parallel,
                max_workers=self.max_workers
            )
        self._betas = betas

        if self.config.verbose:
            print("  Done fitting.")

        # Compute PRS
        prs = np.dot(X_test, betas)

        training_time = time.time() - start_time

        results = {}

        if self._task_type == 'classification':
            # Scale PRS for binary
            prs_scaled = (prs - prs.min()) / (prs.max() - prs.min())

            # Compute classification metrics for different quantile thresholds
            metrics_lower = compute_prs_metrics(prs, y_test, quantile_h=None, quantile_l=0.05, verbose=False)
            metrics_middle = compute_prs_metrics(prs, y_test, quantile_h=0.40, quantile_l=0.60, verbose=False)
            metrics_top = compute_prs_metrics(prs, y_test, quantile_h=0.95, quantile_l=None, verbose=False)
            metrics_scaled = compute_metrics(prs_scaled, y_test)

            metric_variants = {
                f'{model_name} (q <= 0.05)': metrics_lower,
                f'{model_name} (0.4 <= q <= 0.6)': metrics_middle,
                f'{model_name} (q >= 0.95)': metrics_top,
                f'{model_name} scaled (threshold 0.5)': metrics_scaled,
            }

            for variant_name, metrics in metric_variants.items():
                if metrics_dict is not None:
                    log_metrics(metrics_dict, variant_name, metrics)

                results[variant_name] = TrainingResult(
                    model_name=variant_name,
                    model=betas,
                    y_score=prs_scaled if 'scaled' in variant_name else prs,
                    y_test=y_test,
                    metrics=metrics,
                    best_params=None,
                    training_time=training_time,
                )

        elif self._task_type == 'multiclass':
            # Betas fitted with multinomial -> softmax probabilities
            prs_proba = softmax(prs, axis=1)  # Shape: (n_samples, n_classes)

            # Compute multiclass metrics
            metrics_overall = compute_multiclass_metrics(prs_proba, y_test.astype(int))

            variant_name = f'{model_name} (overall)'
            if metrics_dict is not None:
                log_metrics(metrics_dict, variant_name, metrics_overall)

            results[variant_name] = TrainingResult(
                model_name=variant_name,
                model=betas,
                y_score=prs_proba,
                y_test=y_test,
                metrics=metrics_overall,
                best_params=None,
                training_time=training_time,
            )

            if self.config.verbose:
                print(f"  -> {variant_name}: Macro AUC={metrics_overall['macro_auc']:.4f}, "
                      f"Balanced Acc={metrics_overall['balanced_accuracy']:.4f}")

        # Save betas if path provided
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            joblib.dump(betas, os.path.join(save_path, f'{model_name}__betas'))

        if self.config.verbose:
            print("Done.")

        return results

    @property
    def betas(self) -> np.ndarray:
        """Get the fitted beta coefficients."""
        if self._betas is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self._betas
