"""
IO utilities for saving and loading synthetic datasets.
"""

import os
from typing import Dict, Optional, Any, Union
from dataclasses import fields

import h5py
import numpy as np


def save_synthetic_dataset(
    result,
    output_path: str,
    mode: str = 'complete',
    augmentation_strategy: Optional[str] = None,
    extra_attrs: Optional[Dict] = None,
    compression: str = 'gzip'
) -> str:
    """
    Save a synthetic dataset to HDF5 file.

    Args:
        result: GenerationResult object or dict with generation results
        output_path: Path to save the HDF5 file
        mode: Generation mode ('complete', 'syn_recon', 'augmented', 'reconstruction')
        augmentation_strategy: Name of augmentation strategy (for augmented mode)
        extra_attrs: Additional attributes to save
        compression: HDF5 compression type

    Returns:
        Path to saved file
    """
    # Convert GenerationResult to dict if needed
    if hasattr(result, '__dataclass_fields__'):
        data = {f.name: getattr(result, f.name) for f in fields(result)}
    else:
        data = result

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    with h5py.File(output_path, "w") as hf:
        # Save main data arrays
        if mode == 'complete':
            hf.create_dataset("targets", data=data['targets'], compression=compression)
            hf.create_dataset("syn_samples", data=data['samples'], compression=compression)

        elif mode == 'syn_recon':
            # Save all reconstruction metadata
            hf.create_dataset("orig_samples", data=data['orig_samples'], compression=compression)
            hf.create_dataset("targets", data=data['targets'], compression=compression)
            hf.create_dataset("syn_samples", data=data['samples'], compression=compression)
            hf.create_dataset("mu", data=data['mu'], compression=compression)
            hf.create_dataset("logvar", data=data['logvar'], compression=compression)
            hf.create_dataset("z_recon", data=data['z_recon'], compression=compression)
            hf.create_dataset("reconstructions", data=data['reconstructions'], compression=compression)
            hf.create_dataset("z_recon_ddpm", data=data['z_recon_ddpm'], compression=compression)
            hf.create_dataset("reconstructions_ddpm", data=data['reconstructions_ddpm'], compression=compression)

            if data.get('z_syn') is not None:
                hf.create_dataset("z_syn", data=data['z_syn'], compression=compression)
            if data.get('z_recon_scaled') is not None:
                hf.create_dataset("z_recon_scaled", data=data['z_recon_scaled'], compression=compression)

        elif mode == 'augmented':
            hf.create_dataset("targets", data=data['targets'], compression=compression)
            hf.create_dataset("syn_samples", data=data['samples'], compression=compression)

            # Save augmentation metadata
            if augmentation_strategy:
                hf.attrs['augmentation_strategy'] = augmentation_strategy

        elif mode == 'reconstruction':
            # VAE-only reconstruction mode for ML/PRS training
            # Uses same keys as 'complete' mode for compatibility with SplitDataset loader
            # (syn_samples key allows reusing existing data loading code)
            hf.create_dataset("targets", data=data['targets'], compression=compression)
            hf.create_dataset("syn_samples", data=data['reconstructions'], compression=compression)

            # Save additional label types if available (for flexible downstream training)
            if data.get('labels') is not None:
                hf.create_dataset("labels", data=data['labels'], compression=compression)
                
            # Save eids if available
            if data.get('eids') is not None:
                hf.create_dataset("eids", data=data['eids'], compression=compression)

            # Optionally save original samples for analysis
            if data.get('orig_samples') is not None:
                hf.create_dataset("orig_samples", data=data['orig_samples'], compression=compression)

            # Optionally save latent space info for analysis
            if data.get('mu') is not None:
                hf.create_dataset("mu", data=data['mu'], compression=compression)
            if data.get('logvar') is not None:
                hf.create_dataset("logvar", data=data['logvar'], compression=compression)
            if data.get('z') is not None:
                hf.create_dataset("z", data=data['z'], compression=compression)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Save extra attributes
        if extra_attrs:
            for k, v in extra_attrs.items():
                hf.attrs[k] = v

    return output_path


def load_synthetic_dataset(
    file_path: str,
    keys: Optional[list] = None
) -> Dict[str, Any]:
    """
    Load a synthetic dataset from HDF5 file.

    Args:
        file_path: Path to the HDF5 file
        keys: Specific keys to load (None loads all)

    Returns:
        Dictionary with loaded data and attributes
    """
    result = {}

    with h5py.File(file_path, "r") as hf:
        # Load datasets
        available_keys = list(hf.keys())
        keys_to_load = keys if keys is not None else available_keys

        for key in keys_to_load:
            if key in hf:
                result[key] = hf[key][:]

        # Load attributes
        result['attrs'] = dict(hf.attrs)

    return result


def get_output_filename(
    base_name: str,
    mode: str,
    augmentation_strategy: Optional[str] = None,
    split: Optional[str] = None
) -> str:
    """
    Generate standardized output filename for synthetic dataset.

    Args:
        base_name: Base name for the file (e.g., 'syn')
        mode: Generation mode ('complete', 'syn_recon', 'augmented', 'reconstruction')
        augmentation_strategy: Name of augmentation strategy
        split: Optional split name to include in filename (e.g., 'train_val', 'test')

    Returns:
        Filename string
    """
    split_suffix = f"_{split}" if split else ""

    if mode == 'complete':
        return f'{base_name}_complete_dataset{split_suffix}.hdf5'
    elif mode == 'syn_recon':
        return f'{base_name}_recon_dataset{split_suffix}.hdf5'
    elif mode == 'augmented':
        if augmentation_strategy:
            return f'{base_name}_augmented_dataset_{augmentation_strategy}{split_suffix}.hdf5'
        return f'{base_name}_augmented_dataset{split_suffix}.hdf5'
    elif mode == 'reconstruction':
        return f'vae_reconstruction_dataset{split_suffix}.hdf5'
    else:
        return f'{base_name}_{mode}_dataset{split_suffix}.hdf5'


def dataset_exists(output_dir: str, filename: str) -> bool:
    """Check if a dataset file already exists."""
    return os.path.exists(os.path.join(output_dir, filename))
