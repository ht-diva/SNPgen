"""
Privacy evaluation module for SNPgen synthetic genomic data.

Provides metrics to assess whether synthetic SNP data preserves patient privacy
by measuring distance-based similarity to real training data.

Quick usage:
    >>> from snpgen.evaluation.privacy import (
    ...     PrivacyEvaluator,
    ...     load_privacy_data_from_checkpoint,
    ... )
    >>> bundle = load_privacy_data_from_checkpoint(checkpoint_dir, model_name="T2D")
    >>> evaluator = PrivacyEvaluator(distance='hamming', per_class=True)
    >>> results = evaluator.evaluate(bundle, output_dir, eval_target='synthetic')
"""

# Core metrics
from .metrics import (
    identical_match_rate,
    dcr_analysis,
    nnaa,
    membership_inference_distance,
    nndr_analysis,
    allele_frequency_comparison,
    PrivacyEvaluator,
    # Result dataclasses
    IMRResult,
    DCRResult,
    NNAAResult,
    MIResult,
    NNDRResult,
    MAFResult,
)

# Data loading and persistence
from .utils import (
    PrivacyDataBundle,
    load_privacy_data_from_checkpoint,
    load_privacy_data_manual,
    save_privacy_results,
    load_privacy_results,
    subsample_by_label,
    subsample_balanced,
)

# Distance computation
from .distances import (
    batched_knn,
    auto_batch_size,
    onehot_encode_snps,
    KNNCache,
)

# Visualization
from .visualization import (
    plot_dcr_distributions,
    plot_nnaa_summary,
    plot_mi_roc,
    plot_nndr_histogram,
    plot_maf_scatter,
    plot_privacy_summary_table,
)


__all__ = [
    # Evaluator
    "PrivacyEvaluator",
    # Metrics
    "identical_match_rate",
    "dcr_analysis",
    "nnaa",
    "membership_inference_distance",
    "nndr_analysis",
    "allele_frequency_comparison",
    # Results
    "IMRResult",
    "DCRResult",
    "NNAAResult",
    "MIResult",
    "NNDRResult",
    "MAFResult",
    # Data
    "PrivacyDataBundle",
    "load_privacy_data_from_checkpoint",
    "load_privacy_data_manual",
    "save_privacy_results",
    "load_privacy_results",
    "subsample_by_label",
    "subsample_balanced",
    # Distances
    "batched_knn",
    "auto_batch_size",
    "onehot_encode_snps",
    "KNNCache",
    # Visualization
    "plot_dcr_distributions",
    "plot_nnaa_summary",
    "plot_mi_roc",
    "plot_nndr_histogram",
    "plot_maf_scatter",
    "plot_privacy_summary_table",
]
