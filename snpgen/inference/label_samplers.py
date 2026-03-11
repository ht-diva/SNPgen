"""
Label sampling strategies for augmented dataset generation.

Provides various strategies for sampling labels when generating synthetic
datasets, supporting binary (case/control) and multiclass discrete traits.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, Tuple
import numpy as np
import torch


class LabelSampler(ABC):
    """Abstract base class for label sampling strategies."""

    def __init__(self, name: str):
        self.name = name

    @property
    def dtype(self) -> torch.dtype:
        """Return appropriate torch dtype for labels."""
        return torch.long

    @abstractmethod
    def sample(self, n_samples: Optional[int] = None) -> np.ndarray:
        """
        Generate labels according to the sampling strategy.

        Args:
            n_samples: Number of labels to generate. If provided, overrides
                      any internal sample count parameters. If None, uses
                      internal parameters (e.g., n_controls, samples_per_class).

        Returns:
            Array of sampled labels
        """
        pass
    
    @abstractmethod
    def get_info(self) -> dict:
        """Return sampling parameters for logging/saving."""
        pass


class BinaryBalancedSampler(LabelSampler):
    """
    Sampler for binary traits that creates balanced classes.

    Generates equal numbers of controls (0) and cases (1).

    Args:
        n_controls: Number of control samples to generate.
                    If None, must specify total_samples in sample()
    """

    def __init__(self, n_controls: Optional[int] = None):
        super().__init__(name="binary_balanced")
        self.n_controls = n_controls

    def sample(self, n_samples: Optional[int] = None) -> np.ndarray:
        """
        Generate balanced binary labels.

        If n_samples is provided, generates n_samples/2 of each class.
        Otherwise, if n_controls was specified at init, generates n_controls of each class.
        """
        if n_samples is not None:
            n_per_class = n_samples // 2
        elif self.n_controls is not None:
            n_per_class = self.n_controls
        else:
            raise ValueError("Either n_samples must be provided or n_controls must be set at init")

        controls = np.zeros(n_per_class, dtype=np.int64)
        cases = np.ones(n_per_class, dtype=np.int64)
        labels = np.concatenate([controls, cases])

        return labels
    
    def get_info(self) -> dict:
        """Return sampling parameters for logging/saving."""
        return {
            'strategy': self.name,
            'n_controls': self.n_controls
        }


class MulticlassBalancedSampler(LabelSampler):
    """
    Sampler for multiclass traits that creates balanced classes.

    Generates equal numbers of samples for each class (0, 1, 2, ..., n_classes-1).

    Args:
        n_classes: Number of classes
        samples_per_class: Number of samples to generate per class.
                          If None, computes from total_samples in sample()

    Example:
        >>> sampler = MulticlassBalancedSampler(n_classes=3, samples_per_class=1000)
        >>> labels = sampler.sample(3000)  # Returns 1000 each of [0, 1, 2]
    """

    def __init__(self, n_classes: int, samples_per_class: Optional[int] = None):
        super().__init__(name="multiclass_balanced")
        self.n_classes = n_classes
        self.samples_per_class = samples_per_class

    def sample(self, n_samples: Optional[int] = None) -> np.ndarray:
        """
        Generate balanced multiclass labels.

        If n_samples is provided, generates n_samples // n_classes per class (with remainder distributed).
        Otherwise, if samples_per_class was specified at init, generates samples_per_class of each class.
        """
        if n_samples is not None:
            n_per_class = n_samples // self.n_classes
            use_remainder = True
            total_samples = n_samples
        elif self.samples_per_class is not None:
            n_per_class = self.samples_per_class
            use_remainder = False
            total_samples = n_per_class * self.n_classes
        else:
            raise ValueError("Either n_samples must be provided or samples_per_class must be set at init")

        # Generate labels for each class
        labels_list = [np.full(n_per_class, class_id, dtype=np.int64)
                       for class_id in range(self.n_classes)]

        # Handle remainder samples by distributing to first classes
        if use_remainder:
            remainder = total_samples % self.n_classes
            for i in range(remainder):
                extra_label = np.array([i], dtype=np.int64)
                labels_list[i] = np.concatenate([labels_list[i], extra_label])

        labels = np.concatenate(labels_list)
        return labels

    def get_info(self) -> dict:
        """Return sampling parameters for logging/saving."""
        return {
            'strategy': self.name,
            'n_classes': self.n_classes,
            'samples_per_class': self.samples_per_class
        }


class MulticlassWeightedSampler(LabelSampler):
    """
    Sampler for multiclass traits with custom class weights.

    Generates samples according to specified class probabilities/weights.

    Args:
        n_classes: Number of classes
        class_weights: Weights for each class. Can be:
                      - Dict mapping class_id -> weight: {0: 0.5, 1: 0.3, 2: 0.2}
                      - Array/list of weights: [0.5, 0.3, 0.2]
                      Weights are automatically normalized to sum to 1.0

    Example:
        >>> # Generate 60% class 0, 30% class 1, 10% class 2
        >>> sampler = MulticlassWeightedSampler(n_classes=3, class_weights=[0.6, 0.3, 0.1])
        >>> labels = sampler.sample(10000)
    """

    def __init__(self, n_classes: int, class_weights: Union[Dict[int, float], np.ndarray, list]):
        super().__init__(name="multiclass_weighted")
        self.n_classes = n_classes

        # Convert class_weights to normalized array
        if isinstance(class_weights, dict):
            weights_array = np.zeros(n_classes)
            for class_id, weight in class_weights.items():
                weights_array[class_id] = weight
        else:
            weights_array = np.asarray(class_weights)

        # Normalize weights to sum to 1
        self.class_weights = weights_array / weights_array.sum()

    def sample(self, n_samples: Optional[int] = None) -> np.ndarray:
        """Generate multiclass labels according to class weights."""
        if n_samples is None:
            raise ValueError("n_samples must be provided for MulticlassWeightedSampler")

        labels = np.random.choice(
            self.n_classes,
            size=n_samples,
            p=self.class_weights
        ).astype(np.int64)

        return labels

    def get_info(self) -> dict:
        """Return sampling parameters for logging/saving."""
        return {
            'strategy': self.name,
            'n_classes': self.n_classes,
            'class_weights': self.class_weights.tolist()
        }


class MulticlassOriginalSampler(LabelSampler):
    """
    Sampler that matches the original class distribution from training data.

    Computes class frequencies from original labels and generates new samples
    with the same distribution.

    Args:
        original_labels: Array of original labels to compute distribution from
        n_classes: Number of classes (optional, auto-detected from labels if None)

    Example:
        >>> # If training data has 50% class 0, 30% class 1, 20% class 2
        >>> sampler = MulticlassOriginalSampler(train_labels)
        >>> labels = sampler.sample(10000)  # Generates with same proportions
    """

    def __init__(self, original_labels: np.ndarray, n_classes: Optional[int] = None):
        super().__init__(name="multiclass_original")
        self.original_labels = np.asarray(original_labels)

        # Compute class distribution
        unique_classes, counts = np.unique(self.original_labels, return_counts=True)
        self.n_classes = n_classes if n_classes is not None else len(unique_classes)

        # Create probability distribution
        self.class_weights = np.zeros(self.n_classes)
        for class_id, count in zip(unique_classes, counts):
            self.class_weights[int(class_id)] = count
        self.class_weights = self.class_weights / self.class_weights.sum()

    def sample(self, n_samples: Optional[int] = None) -> np.ndarray:
        """Generate multiclass labels matching original distribution."""
        if n_samples is None:
            raise ValueError("n_samples must be provided for MulticlassOriginalSampler")

        labels = np.random.choice(
            self.n_classes,
            size=n_samples,
            p=self.class_weights
        ).astype(np.int64)

        return labels

    def get_info(self) -> dict:
        """Return sampling parameters for logging/saving."""
        return {
            'strategy': self.name,
            'n_classes': self.n_classes,
            'class_weights': self.class_weights.tolist()
        }


def get_sampler(
    strategy: str,
    original_labels: np.ndarray,
    **kwargs
) -> LabelSampler:
    """
    Factory function to create a label sampler.

    Args:
        strategy: Sampling strategy name
            - 'binary_balanced': For binary traits, equal controls and cases
            - 'multiclass_balanced': For multiclass traits, equal samples per class
            - 'multiclass_weighted': For multiclass traits, custom class weights
            - 'multiclass_original': Match original class distribution

        original_labels: Array of original labels (for computing stats)
        **kwargs: Strategy-specific parameters
            - For binary_balanced: n_controls
            - For multiclass_balanced: n_classes, samples_per_class
            - For multiclass_weighted: n_classes, class_weights
            - For multiclass_original: n_classes (optional)

    Returns:
        LabelSampler instance

    Example:
        >>> # Binary
        >>> sampler = get_sampler('binary_balanced', labels)

        >>> # Multiclass balanced
        >>> sampler = get_sampler('multiclass_balanced', labels,
        ...                        n_classes=3, samples_per_class=1000)
    """
    # Auto-detect number of classes from labels
    unique_labels = np.unique(original_labels)
    n_classes = kwargs.get('n_classes', len(unique_labels))

    if strategy == 'binary_balanced':
        n_controls = kwargs.get('n_controls', (original_labels == 0).sum())
        return BinaryBalancedSampler(n_controls=n_controls)

    elif strategy == 'multiclass_balanced':
        samples_per_class = kwargs.get('samples_per_class', None)
        return MulticlassBalancedSampler(n_classes=n_classes, samples_per_class=samples_per_class)

    elif strategy == 'multiclass_weighted':
        class_weights = kwargs.get('class_weights')
        if class_weights is None:
            raise ValueError("'multiclass_weighted' strategy requires 'class_weights' parameter")
        return MulticlassWeightedSampler(n_classes=n_classes, class_weights=class_weights)

    elif strategy == 'multiclass_original':
        return MulticlassOriginalSampler(original_labels=original_labels, n_classes=n_classes)

    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Available: 'binary_balanced', 'multiclass_balanced', "
            f"'multiclass_weighted', 'multiclass_original'"
        )


def detect_discrete_label_type(labels: np.ndarray) -> Tuple[str, int]:
    """
    Detect whether discrete labels are binary or multiclass.

    Args:
        labels: Array of discrete labels

    Returns:
        Tuple of (label_type, n_classes) where label_type is 'binary' or 'multiclass'

    Example:
        >>> labels = np.array([0, 1, 0, 1, 1])
        >>> detect_discrete_label_type(labels)
        ('binary', 2)

        >>> labels = np.array([0, 1, 2, 0, 1, 2])
        >>> detect_discrete_label_type(labels)
        ('multiclass', 3)
    """
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    if n_classes == 2:
        label_type = 'binary'
    elif n_classes > 2:
        label_type = 'multiclass'
    else:
        raise ValueError(f"Expected at least 2 classes, got {n_classes}")

    return label_type, n_classes


def get_default_strategy_for_labels(labels: np.ndarray) -> str:
    """
    Get the default augmentation strategy for a given label type.

    Args:
        labels: Array of discrete labels

    Returns:
        Default strategy name

    Example:
        >>> # Binary labels -> 'binary_balanced'
        >>> get_default_strategy_for_labels(np.array([0, 1, 0]))
        'binary_balanced'

        >>> # Multiclass labels -> 'multiclass_balanced'
        >>> get_default_strategy_for_labels(np.array([0, 1, 2]))
        'multiclass_balanced'
    """
    label_type, n_classes = detect_discrete_label_type(labels)
    if label_type == 'binary':
        return 'binary_balanced'
    else:
        return 'multiclass_balanced'
