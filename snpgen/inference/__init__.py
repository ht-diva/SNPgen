"""
Inference module for generating synthetic SNP datasets from trained DDPM models
and computing VAE reconstructions.

This module provides utilities for:
- Generating synthetic datasets with various conditioning strategies (DDPM)
- Computing VAE reconstructions without DDPM (for evaluating information preservation)
- Label sampling for binary and multiclass trait augmentation
- Saving and loading generated datasets

Example usage:
    # DDPM-based synthetic generation
    from snpgen.inference import SyntheticDatasetGenerator
    from snpgen.inference.label_samplers import BinaryBalancedSampler

    generator = SyntheticDatasetGenerator(ddpm_model, config)

    # Generate complete dataset (same labels as original)
    generator.generate_complete(dataloader, output_path)

    # Generate augmented dataset
    sampler = BinaryBalancedSampler(n_controls=1000)
    generator.generate_augmented(sampler, total_samples, output_path)

    # VAE-only reconstruction (for ML/PRS training evaluation)
    from snpgen.inference import ReconstructionGenerator, save_synthetic_dataset

    recon_generator = ReconstructionGenerator(vae_model, device='cuda')
    result = recon_generator.generate_reconstructions(dataloader)
    save_synthetic_dataset(result, 'vae_reconstructions.hdf5', mode='reconstruction')
"""

from .generator import SyntheticDatasetGenerator
from .reconstruction import ReconstructionGenerator, ReconstructionResult
from .label_samplers import (
    LabelSampler,
    BinaryBalancedSampler,
    MulticlassBalancedSampler,
    MulticlassWeightedSampler,
    MulticlassOriginalSampler,
    get_sampler,
    detect_discrete_label_type,
    get_default_strategy_for_labels,
)
from .io import save_synthetic_dataset, load_synthetic_dataset, get_output_filename, dataset_exists

__all__ = [
    # DDPM-based generation
    "SyntheticDatasetGenerator",
    # VAE-only reconstruction
    "ReconstructionGenerator",
    "ReconstructionResult",
    # Label samplers
    "LabelSampler",
    "BinaryBalancedSampler",
    "MulticlassBalancedSampler",
    "MulticlassWeightedSampler",
    "MulticlassOriginalSampler",
    "get_sampler",
    "detect_discrete_label_type",
    "get_default_strategy_for_labels",
    # IO utilities
    "save_synthetic_dataset",
    "load_synthetic_dataset",
    "get_output_filename",
    "dataset_exists",
]
