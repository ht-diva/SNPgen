"""
Result dataclasses and result handling utilities for evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import joblib
import os

from snpgen.evaluation.utils import detect_task_type
from snpgen.evaluation.metrics import compute_metrics


@dataclass
class MetricsResult:
    """Container for computed classification metrics."""
    accuracy: float
    balanced_accuracy: float
    f1_score: float
    precision: float
    recall: float
    roc_auc: float
    precision_recall_auc: float
    true_positive_count: int
    false_positive_count: int
    true_negative_count: int
    false_negative_count: int
    mathews_correlation_coefficient: float
    cohens_kappa: float

    @classmethod
    def from_dict(cls, metrics_dict: Dict[str, float]) -> "MetricsResult":
        """Create MetricsResult from a dictionary."""
        return cls(
            accuracy=metrics_dict["accuracy"],
            balanced_accuracy=metrics_dict["balanced_accuracy"],
            f1_score=metrics_dict["f1_score"],
            precision=metrics_dict["precision"],
            recall=metrics_dict["recall"],
            roc_auc=metrics_dict["roc_auc"],
            precision_recall_auc=metrics_dict["precision_recall_auc"],
            true_positive_count=int(metrics_dict["true_positive_count"]),
            false_positive_count=int(metrics_dict["false_positive_count"]),
            true_negative_count=int(metrics_dict["true_negative_count"]),
            false_negative_count=int(metrics_dict["false_negative_count"]),
            mathews_correlation_coefficient=metrics_dict["mathews_correlation_coefficient"],
            cohens_kappa=metrics_dict["cohens_kappa"],
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "f1_score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
            "roc_auc": self.roc_auc,
            "precision_recall_auc": self.precision_recall_auc,
            "true_positive_count": self.true_positive_count,
            "false_positive_count": self.false_positive_count,
            "true_negative_count": self.true_negative_count,
            "false_negative_count": self.false_negative_count,
            "mathews_correlation_coefficient": self.mathews_correlation_coefficient,
            "cohens_kappa": self.cohens_kappa,
        }


@dataclass
class TrainingResult:
    """Container for single model training results."""
    model_name: str
    model: Any  # Trained model object
    y_score: np.ndarray  # Predictions
    y_test: np.ndarray  # Ground truth
    metrics: Dict[str, float]
    best_params: Optional[Dict] = None
    training_time: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary (without model object for serialization)."""
        return {
            "model_name": self.model_name,
            "metrics": self.metrics,
            "best_params": self.best_params,
            "training_time": self.training_time,
        }


@dataclass
class EvaluationResults:
    """Container for full evaluation results across models."""
    results: Dict[str, TrainingResult] = field(default_factory=dict)

    def add_result(self, name: str, result: TrainingResult):
        """Add a training result."""
        self.results[name] = result

    def get_model(self, name: str) -> Any:
        """Get the trained model by name."""
        return self.results[name].model

    def get_metrics(self, name: str) -> Dict[str, float]:
        """Get metrics for a specific model."""
        return self.results[name].metrics

    def to_dict(self) -> Dict:
        """Convert to nested dict format compatible with build_multiindex_df."""
        return {
            name: {"clf": result.model, "metrics": result.metrics}
            for name, result in self.results.items()
        }

    def to_serializable_dict(self) -> Dict:
        """Convert to dict without model objects (for pickle/JSON)."""
        return {
            name: {"metrics": result.metrics, "best_params": result.best_params}
            for name, result in self.results.items()
        }


def build_multiindex_df(results_dict: Dict) -> pd.DataFrame:
    """Convert nested results dictionary to pandas DataFrame with MultiIndex.

    Args:
        results_dict: Dictionary with structure:
            {
                'fold_0': {
                    'models': {
                        'model_name': {'metrics': {...}},
                        ...
                    },
                    'metadata': {...}
                },
                ...
            }

    Returns:
        DataFrame with MultiIndex (fold, model) containing metrics.
    """
    records = []
    for fold, v in results_dict.items():
        for model, v_m in v['models'].items():
            metrics = v_m['metrics']
            records.append({
                'fold': fold,
                'model': model,
                **metrics
            })

    df = pd.DataFrame(records)
    df = df.set_index(['fold', 'model'])
    return df


def log_metrics(container: Dict, model: str, metrics: Dict[str, float]):
    """Log metrics to a container dictionary.

    Args:
        container: Dictionary to store metrics.
        model: Model name.
        metrics: Dictionary of metric values.
    """
    for metric, value in metrics.items():
        if metric not in container:
            container[metric] = []
        container[metric].append(value)
    container["model"].append(model)


def process_cv_results(
    cv_results: Dict,
    data_type: str,
    train_index_key: str = None,
) -> Dict:
    """Process CV results from a single data type (real or synthetic).

    Extracts model results and metadata, removing model objects for serialization.

    Args:
        cv_results: Dictionary with CV fold results.
            Structure: {fold_key: {'metadata': {...}, 'real': {...}, 'syn': {...}}}
        data_type: Which data type to process ('real' or 'syn').
        train_index_key: Key for train index in metadata. Defaults to '{data_type}_train_index'.

    Returns:
        Dictionary with processed results:
            {
                'fold_0': {
                    'models': {model_name: {'metrics': {...}}, ...},
                    'metadata': {'train_index': ..., 'test_index': ...}
                },
                ...
            }
    """
    if train_index_key is None:
        train_index_key = f'{data_type}_train_index'

    processed = {}
    for fold_key, fold_data in cv_results.items():
        # Set up fold entry
        if fold_key not in processed:
            processed[fold_key] = {}

        # Skip if no results for this data type
        if fold_data.get(data_type) is None:
            continue

        # Remove model instances (clf) to avoid saving them with pickle
        models = {}
        for model_name, model_data in fold_data[data_type].items():
            models[model_name] = {
                k: v for k, v in model_data.items() if k != 'clf'
            }

        processed[fold_key]['models'] = models
        processed[fold_key]['metadata'] = {
            'train_index': fold_data['metadata'].get(train_index_key),
            'test_index': fold_data['metadata'].get('test_index'),
        }

    return processed


def save_cv_results(
    cv_results: Dict,
    save_path: str,
    filename: str = 'results.pkl',
) -> None:
    """Save CV results to a pickle file.

    Args:
        cv_results: Processed CV results dictionary.
        save_path: Directory to save the file.
        filename: Name of the pickle file.
    """
    import os
    import pickle

    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, filename)

    with open(filepath, 'wb') as fp:
        pickle.dump(cv_results, fp)

    print(f"  Results saved to: {filepath}")


def load_cv_results(
    load_path: str,
    filename: str = 'results.pkl',
) -> Dict:
    """Load CV results from a pickle file.

    Args:
        load_path: Directory containing the file.
        filename: Name of the pickle file.

    Returns:
        Loaded CV results dictionary.
    """
    import os
    import pickle

    filepath = os.path.join(load_path, filename)

    with open(filepath, 'rb') as fp:
        return pickle.load(fp)


def check_results_exist(save_path: str, filename: str = 'results.pkl') -> bool:
    """Check if results file already exists.

    Args:
        save_path: Directory to check.
        filename: Name of the pickle file.

    Returns:
        True if file exists, False otherwise.
    """
    import os
    return os.path.exists(os.path.join(save_path, filename))


def recover_cv_results_from_saved_models(
    ml_models_path: str,
    real_split: tuple,
    test_split: tuple,
    skf_real,
    skf_test,
    n_folds: int = 5,
    data_type: str = 'real'
) -> dict:
    """
    Recover cv_results from saved models by reloading them and recomputing metrics.
    
    Args:
        ml_models_path: Path to the ml_models directory containing fold_* subdirectories
        real_split: Tuple of (X, y) for the training split
        test_split: Tuple of (X, y) for the test split  
        skf_real: Stratified K-Fold splitter for the training data
        skf_test: Stratified K-Fold splitter for the test data
        n_folds: Number of folds
        data_type: 'real' or 'syn' - used for index key naming
        
    Returns:
        cv_results dictionary in the expected format for process_cv_results
    """
    cv_results = {}
    
    # Detect task type from targets
    task_type = detect_task_type(test_split[1])
    print(f"Detected task type: {task_type}")
    
    for i, ((train_index, _), (_, test_index)) in enumerate(zip(
            skf_real.split(*real_split),
            skf_test.split(*test_split)
        )):
        
        fold_key = f'fold_{i}'
        fold_path = os.path.join(ml_models_path, fold_key)
        
        if not os.path.exists(fold_path):
            print(f"  Warning: {fold_path} does not exist, skipping fold {i}")
            continue
            
        print(f"\nRecovering fold {i}...")
        
        # Get test data for this fold
        X_test, y_test = test_split[0][test_index], test_split[1][test_index]
        
        # Initialize fold results
        cv_results[fold_key] = {
            'metadata': {
                f'{data_type}_train_index': train_index,
                'test_index': test_index
            },
            data_type: {}
        }
        
        # Load each saved model and compute metrics
        saved_files = os.listdir(fold_path)
        
        for model_file in saved_files:
            model_path = os.path.join(fold_path, model_file)
            
            try:
                # Handle PRS models (saved as *__betas)
                if '__betas' in model_file:
                    model_name = model_file.replace('__betas', '')
                    betas = joblib.load(model_path)
                    
                    # Compute PRS scores
                    prs_raw = X_test @ betas
                    
                    if task_type == 'classification':
                        # Scale PRS to [0, 1] range for classification
                        prs_min, prs_max = prs_raw.min(), prs_raw.max()
                        prs_scaled = (prs_raw - prs_min) / (prs_max - prs_min + 1e-10)
                        
                        # Try different thresholds
                        for threshold in [0.5]:
                            variant_name = f'{model_name} scaled (threshold {threshold})'
                            metrics = compute_metrics(prs_scaled, y_test, threshold=threshold)
                            cv_results[fold_key][data_type][variant_name] = {
                                'clf': None,  # Don't store model object
                                'metrics': metrics
                            }
                            print(f"    → {variant_name}: Balanced Acc={metrics['balanced_accuracy']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
                    else:
                        # Regression metrics (not supported, skip)
                        print(f"    → Skipping {model_name}: regression tasks not supported")
                        continue
                else:
                    # Regular sklearn/xgboost/catboost model
                    model_name = model_file
                    model = joblib.load(model_path)

                    if task_type == 'classification':
                        # Get probability predictions
                        if hasattr(model, 'predict_proba'):
                            y_score = model.predict_proba(X_test)[:, 1]
                        else:
                            y_score = model.predict(X_test)
                        metrics = compute_metrics(y_score, y_test)
                        print(f"    → {model_name}: Balanced Acc={metrics['balanced_accuracy']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
                    else:
                        # Non-classification tasks (not supported, skip)
                        print(f"    → Skipping {model_name}: regression tasks not supported")
                        continue
                    
                    cv_results[fold_key][data_type][model_name] = {
                        'clf': None,  # Don't store model object
                        'metrics': metrics
                    }
                    
            except Exception as e:
                print(f"    Warning: Failed to load {model_file}: {e}")
                continue

    return cv_results


def compute_gwas_prs_results(
    gwas_betas: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = 'prs gwas',
    verbose: bool = True,
    auto_flip: bool = True,
    y_eval: np.ndarray = None,
) -> Dict[str, Dict]:
    """Compute PRS metrics using pre-existing GWAS betas.

    This function computes PRS using fixed GWAS betas (not learned during CV)
    and returns results in the format expected by the CV pipeline.

    WARNING: The input gwas_betas MUST be aligned to the effect allele used in
    the genotype matrix (X_test). This alignment should be performed when
    preparing the HDF5 dataset from BED + GWAS files. If the GWAS effect allele
    doesn't match the BED file's A1 allele for a SNP, the beta for that SNP
    should be flipped (negated) during dataset preparation. See
    `snpgen.utils.bed_conversion.align_alleles_and_flip_betas()` for this.

    Unlike learned PRS (where betas are fit to the data), GWAS betas come from
    external studies and may have opposite sign conventions even after allele
    alignment (e.g., if the trait coding differs). When auto_flip=True, the
    function automatically detects if the PRS has an inverted relationship with
    the target (negative correlation) and flips the PRS direction to ensure
    correct metrics.

    For classification tasks:
        - AUC will be >= 0.5 after flipping
        - Interpretation: higher PRS = higher disease risk

    Args:
        gwas_betas: Array of GWAS effect sizes (betas) for each SNP. Must be
            aligned to the effect allele in the genotype matrix.
        X_test: Test set features (samples x SNPs).
        y_test: Test set targets. Used for task type detection. Also used for
            evaluation if y_eval is not provided.
        model_name: Base name for the model variants.
        verbose: Whether to print results.
        auto_flip: If True, automatically flip PRS direction when it has an
            inverted relationship with the target (AUC < 0.5 or negative
            correlation). This ensures metrics are always interpretable
            (AUC >= 0.5, positive correlation). Default True.
        y_eval: Evaluation targets (optional). If provided, metrics are computed
            against y_eval instead of y_test.

    Returns:
        Dictionary mapping model variant names to {'metrics': {...}} dicts.
        Each metrics dict includes 'prs_flipped' (bool) indicating if the
        PRS direction was flipped.
    """
    from snpgen.evaluation.metrics import (
        compute_metrics,
        compute_prs_metrics,
        compute_prs_multiclass_metrics,
    )
    from scipy.stats import spearmanr

    # Compute PRS using GWAS betas
    prs = np.dot(X_test, gwas_betas)

    # Detect task type from y_test; use y_eval for actual metric computation
    task_type = detect_task_type(y_test)
    y_eval = y_eval if y_eval is not None else y_test

    # Auto-flip: check if PRS direction is inverted
    # For GWAS betas, the sign convention may differ from the target coding
    flipped = False
    if auto_flip:
        # Use Spearman correlation to detect direction (robust to outliers)
        corr, _ = spearmanr(prs, y_eval)
        if corr < 0:
            prs = -prs
            flipped = True
            if verbose:
                print(f"  ⚠ PRS direction inverted (Spearman r={corr:.4f}). Flipping PRS for correct metrics.")

    # Scale PRS to [0, 1] for classification metrics
    prs_scaled = (prs - prs.min()) / (prs.max() - prs.min() + 1e-10)

    results = {}

    if task_type == 'multiclass':
        # Compute multiclass metrics for PRS (single continuous score with ordinal classes)
        # Since PRS is a single score (not class probabilities), we use ordinal metrics:
        # - Spearman correlation (rank-based, appropriate for ordinal classes)
        # - Balanced accuracy via quantile-based class assignment
        # - Ordinal MAE (mean absolute error between predicted and true class)
        metrics_overall = compute_prs_multiclass_metrics(prs, y_eval, ordinal=True)
        metrics_overall['prs_flipped'] = flipped

        results[f'{model_name} (overall)'] = {'clf': None, 'metrics': metrics_overall}

        if verbose:
            print(f"  → {model_name} (overall): Spearman r={metrics_overall['spearman_r']:.4f}, "
                  f"Macro AUC={metrics_overall['macro_auc']:.4f}, "
                  f"Balanced Acc={metrics_overall['balanced_accuracy']:.4f}, "
                  f"Ordinal MAE={metrics_overall['ordinal_mae']:.4f}")

    else:
        # Binary classification metrics
        metrics_lower = compute_prs_metrics(prs, y_eval, quantile_h=None, quantile_l=0.05, verbose=False)
        metrics_middle = compute_prs_metrics(prs, y_eval, quantile_h=0.40, quantile_l=0.60, verbose=False)
        metrics_top = compute_prs_metrics(prs, y_eval, quantile_h=0.95, quantile_l=None, verbose=False)
        metrics_scaled = compute_metrics(prs_scaled, y_eval)

        # Add flipped indicator to metrics
        for m in [metrics_lower, metrics_middle, metrics_top, metrics_scaled]:
            m['prs_flipped'] = flipped

        results[f'{model_name} (q <= 0.05)'] = {'clf': None, 'metrics': metrics_lower}
        results[f'{model_name} (0.4 <= q <= 0.6)'] = {'clf': None, 'metrics': metrics_middle}
        results[f'{model_name} (q >= 0.95)'] = {'clf': None, 'metrics': metrics_top}
        results[f'{model_name} scaled (threshold 0.5)'] = {'clf': None, 'metrics': metrics_scaled}

        if verbose:
            print(f"  → {model_name} scaled: Balanced Acc={metrics_scaled['balanced_accuracy']:.4f}, ROC-AUC={metrics_scaled['roc_auc']:.4f}")

    return results


# =============================================================================
# Incremental CV training utilities
# =============================================================================

def verify_cv_indices(
    existing_results: Dict,
    fold_key: str,
    train_index: np.ndarray,
    test_index: np.ndarray,
) -> None:
    """Verify that CV indices match between existing and current splits.

    Args:
        existing_results: Loaded results.pkl dict.
        fold_key: e.g., 'fold_0'.
        train_index: Current train indices for this fold.
        test_index: Current test indices for this fold.

    Raises:
        ValueError: If indices don't match with detailed diagnostic info.
    """
    if fold_key not in existing_results:
        return  # New fold, no verification needed

    metadata = existing_results[fold_key].get('metadata', {})
    saved_train = metadata.get('train_index')
    saved_test = metadata.get('test_index')

    if saved_train is None or saved_test is None:
        print(f"  Warning: No saved indices found for {fold_key}, skipping verification")
        return

    train_match = np.array_equal(train_index, saved_train)
    test_match = np.array_equal(test_index, saved_test)

    if not train_match or not test_match:
        raise ValueError(
            f"CV indices mismatch for {fold_key}!\n"
            f"  Train indices match: {train_match}\n"
            f"  Test indices match: {test_match}\n"
            f"  Current train size: {len(train_index)}, Saved: {len(saved_train)}\n"
            f"  Current test size: {len(test_index)}, Saved: {len(saved_test)}\n"
            f"  This likely means the random seed or data split changed.\n"
            f"  To retrain from scratch: delete the corresponding results.pkl"
        )


def get_missing_trainers(
    existing_results: Dict,
    fold_key: str,
    requested_trainers: List[str],
    force_retrain: Optional[List[str]] = None,
) -> List[str]:
    """Determine which trainers need to be run based on existing results.

    Args:
        existing_results: Loaded results.pkl dict.
        fold_key: e.g., 'fold_0'.
        requested_trainers: List of trainer names, e.g., ['xgboost', 'prs'].
        force_retrain: Optional list of trainer names to always retrain,
            even if their results already exist. e.g., ['prs'].

    Returns:
        List of trainer names whose results are missing or forced to retrain.

    Note:
        Trainer-to-result-key mapping:
        - 'prs' -> checks for any key starting with 'prs univariate'
        - All others -> exact key match (e.g., 'xgboost', 'catboost')
    """
    if fold_key not in existing_results:
        return list(requested_trainers)

    force_set = set(force_retrain) if force_retrain else set()
    existing_models = set(existing_results[fold_key].get('models', {}).keys())
    missing = []

    for trainer in requested_trainers:
        if trainer in force_set:
            missing.append(trainer)
        elif trainer == 'prs':
            if not any(k.startswith('prs univariate') for k in existing_models):
                missing.append(trainer)
        else:
            if trainer not in existing_models:
                missing.append(trainer)

    return missing


def merge_fold_results(
    existing_results: Dict,
    new_results: Dict,
    fold_key: str,
    metadata: Dict,
) -> None:
    """Merge new model results into existing fold results (in place).

    Args:
        existing_results: Complete results dict (modified in place).
        new_results: Raw output from train_models() or compute_gwas_prs_results()
            for one fold: {'model_name': {'clf': ..., 'metrics': {...}}, ...}
        fold_key: e.g., 'fold_0'.
        metadata: Dict with 'train_index' and 'test_index' for this fold.
    """
    if fold_key not in existing_results:
        existing_results[fold_key] = {'models': {}, 'metadata': {}}

    # Strip 'clf' keys from new results (like process_cv_results does)
    for model_name, model_data in new_results.items():
        existing_results[fold_key]['models'][model_name] = {
            k: v for k, v in model_data.items() if k != 'clf'
        }

    # Update metadata
    existing_results[fold_key]['metadata'] = metadata
