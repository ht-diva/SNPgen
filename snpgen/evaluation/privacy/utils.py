"""
Data loading, persistence, and helper utilities for privacy evaluation.

Provides:
- PrivacyDataBundle: dataclass holding real/synthetic/reconstructed data
- Checkpoint-based and manual data loading
- Result save/load for incremental evaluation
- Subsampling utilities for per-class analysis
"""

import os
import json
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import h5py
import numpy as np

try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data Bundle
# ---------------------------------------------------------------------------

@dataclass
class PrivacyDataBundle:
    """Container for all data needed by privacy metrics.

    Attributes:
        real_train: (N_train, N_snps) int8, SNP values {0,1,2}.
        real_holdout: (N_holdout, N_snps) int8.
        synthetic: (N_syn, N_snps) int8.
        reconstructed: Optional (N_recon, N_snps) int8.
        labels_train: (N_train,) labels (binary or multiclass int).
        labels_holdout: (N_holdout,) labels.
        labels_syn: (N_syn,) labels.
        labels_recon: Optional (N_recon,) labels.
        model_name: Identifier for this model/trait.
        n_snps: Number of SNP features.
        ddpm_checkpoint_dir: Path to DDPM checkpoint (for saving syn results).
        vae_checkpoint_dir: Path to VAE checkpoint (for saving recon results).
        conditioning_type: 'classification' or 'multiclass'. Auto-detected if None.
    """
    real_train: np.ndarray
    real_holdout: np.ndarray
    synthetic: np.ndarray
    reconstructed: Optional[np.ndarray] = None
    labels_train: np.ndarray = field(default_factory=lambda: np.array([]))
    labels_holdout: np.ndarray = field(default_factory=lambda: np.array([]))
    labels_syn: np.ndarray = field(default_factory=lambda: np.array([]))
    labels_recon: Optional[np.ndarray] = None
    model_name: str = ""
    n_snps: int = 0
    ddpm_checkpoint_dir: str = ""
    vae_checkpoint_dir: str = ""
    conditioning_type: Optional[str] = None

    def __post_init__(self):
        if self.n_snps == 0 and self.real_train is not None:
            self.n_snps = self.real_train.shape[1]

    def summary(self):
        """Print a summary table of the data bundle."""
        rows = [
            ("real_train", self.real_train.shape, _label_dist(self.labels_train)),
            ("real_holdout", self.real_holdout.shape, _label_dist(self.labels_holdout)),
            ("synthetic", self.synthetic.shape, _label_dist(self.labels_syn)),
        ]
        if self.reconstructed is not None:
            rows.append(("reconstructed", self.reconstructed.shape,
                         _label_dist(self.labels_recon) if self.labels_recon is not None else "N/A"))

        print(f"\n{'='*60}")
        print(f" Privacy Data Bundle: {self.model_name}")
        print(f" N_SNPs: {self.n_snps}")
        if self.conditioning_type:
            print(f" Conditioning: {self.conditioning_type}")
        print(f"{'='*60}")
        print(f"{'Dataset':<18} {'Shape':<20} {'Label dist':<20}")
        print(f"{'-'*58}")
        for name, shape, dist in rows:
            print(f"{name:<18} {str(shape):<20} {dist:<20}")
        print()


def _label_dist(labels):
    """Format label distribution as a string."""
    if labels is None or len(labels) == 0:
        return "N/A"
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) > 10:
        return f"many values ({len(unique)} unique)"
    parts = [f"{int(u)}:{int(c)}" for u, c in zip(unique, counts)]
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_privacy_data_from_checkpoint(
    ddpm_checkpoint_dir,
    model_name="",
    train_split='train_val',
    holdout_split='test',
    syn_filename='syn_complete_dataset.hdf5',
    recon_filename='vae_reconstruction_dataset_train_val.hdf5',
    verbose=True,
):
    """Load privacy evaluation data from a DDPM checkpoint directory.

    Reads config.yaml to find the dataset path and VAE checkpoint.
    Loads real data (split into train_val and holdout), synthetic data,
    and optionally reconstructed data from the VAE checkpoint.

    Args:
        ddpm_checkpoint_dir: Path to DDPM checkpoint with config.yaml.
        model_name: Name for this model/trait.
        train_split: Which split to use as training data ('train_val' by default).
        holdout_split: Which split to use as holdout ('test' or 'val').
        syn_filename: Name of synthetic data HDF5 file in checkpoint dir.
        recon_filename: Name of reconstruction HDF5 in VAE checkpoint dir.
        verbose: Print loading info.

    Returns:
        PrivacyDataBundle
    """
    if not OMEGACONF_AVAILABLE:
        raise ImportError("omegaconf is required. Install with: pip install omegaconf")

    config_path = os.path.join(ddpm_checkpoint_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.yaml not found in {ddpm_checkpoint_dir}. "
            f"Use load_privacy_data_manual() instead."
        )

    config = OmegaConf.load(config_path)
    h5_path = config.get('dataset_path', config.get('data', {}).get('dataset_path'))
    seed = config.get('seed', 42)

    if h5_path is None:
        raise ValueError("Could not find 'dataset_path' in config.yaml")

    if verbose:
        print(f"Loading real data from: {h5_path}")

    # Load real data using SplitDataset
    from snpgen.utils import instantiate_from_config
    raw_dataset = instantiate_from_config(
        config.data.raw_dataset,
        file_path=h5_path,
        seed=seed,
        onehot=False,
        data_dtype=None,
        metadata=True,
        verbose=verbose,
    )

    # Get train_val and holdout splits
    real_train, labels_train = raw_dataset.get_split(train_split, metadata=False)
    real_holdout, labels_holdout = raw_dataset.get_split(holdout_split, metadata=False)

    # Ensure int8 for memory efficiency
    if np.issubdtype(real_train.dtype, np.floating):
        print(
            f"⚠️ Warning: Real data loaded with float dtype {real_train.dtype}. "
            f"Converting to int8, which may cause issues if data is not already in {{0,1,2}} format."
        )
    real_train = np.asarray(real_train, dtype=np.int8)
    real_holdout = np.asarray(real_holdout, dtype=np.int8)

    if verbose:
        print(f"  real_train: {real_train.shape}, real_holdout: {real_holdout.shape}")

    # Load synthetic data
    syn_path = os.path.join(ddpm_checkpoint_dir, syn_filename)
    synthetic, labels_syn = _load_hdf5_samples(syn_path, verbose=verbose, name="synthetic")

    # Find VAE checkpoint dir and load reconstructions
    vae_checkpoint_dir = _extract_vae_dir(config)
    reconstructed = None
    labels_recon = None
    if vae_checkpoint_dir:
        recon_path = os.path.join(vae_checkpoint_dir, recon_filename)
        if os.path.exists(recon_path):
            reconstructed, labels_recon = _load_hdf5_samples(
                recon_path, verbose=verbose, name="reconstructed"
            )
        elif verbose:
            print(f"  No reconstruction file found at {recon_path}")
    elif verbose:
        print("  Could not determine VAE checkpoint dir from config")

    # Detect conditioning type
    from snpgen.evaluation.utils import detect_task_type
    conditioning_type = detect_task_type(labels_train)

    if verbose:
        print(f"  Conditioning type: {conditioning_type}")

    return PrivacyDataBundle(
        real_train=real_train,
        real_holdout=real_holdout,
        synthetic=synthetic,
        reconstructed=reconstructed,
        labels_train=labels_train,
        labels_holdout=labels_holdout,
        labels_syn=labels_syn,
        labels_recon=labels_recon,
        model_name=model_name,
        ddpm_checkpoint_dir=ddpm_checkpoint_dir,
        vae_checkpoint_dir=vae_checkpoint_dir or "",
        conditioning_type=conditioning_type,
    )


def load_privacy_data_manual(
    dataset_path,
    ddpm_checkpoint_dir,
    vae_checkpoint_dir="",
    model_name="",
    seed=42,
    val_ratio=0.2,
    test_ratio=0.1,
    holdout_split='test',
    syn_filename='syn_complete_dataset.hdf5',
    recon_filename='vae_reconstruction_dataset_train_val.hdf5',
    conditioning_type=None,
    verbose=True,
):
    """Load privacy evaluation data with manually specified paths.

    Use this for models without a config.yaml (e.g., CAD).

    Args:
        dataset_path: Path to the original HDF5 dataset.
        ddpm_checkpoint_dir: Path to DDPM checkpoint dir.
        vae_checkpoint_dir: Path to VAE checkpoint dir.
        model_name: Name for this model/trait.
        seed: Random seed for splitting.
        val_ratio: Validation ratio for split.
        test_ratio: Test ratio for split.
        holdout_split: Which split to use as holdout.
        syn_filename: Synthetic data filename.
        recon_filename: Reconstruction data filename.
        conditioning_type: 'classification' or 'multiclass'.
            If None, auto-detected from labels using detect_task_type().
        verbose: Print loading info.

    Returns:
        PrivacyDataBundle
    """
    from snpgen.data.loader import SplitDataset

    if verbose:
        print(f"Loading real data from: {dataset_path}")

    raw_dataset = SplitDataset(
        file_path=dataset_path,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        onehot=False,
        data_dtype=None,
        metadata=True,
        seed=seed,
        verbose=verbose,
    )

    real_train, labels_train = raw_dataset.get_split('train_val', metadata=False)
    real_holdout, labels_holdout = raw_dataset.get_split(holdout_split, metadata=False)

    real_train = np.asarray(real_train, dtype=np.int8)
    real_holdout = np.asarray(real_holdout, dtype=np.int8)

    if verbose:
        print(f"  real_train: {real_train.shape}, real_holdout: {real_holdout.shape}")

    # Load synthetic
    syn_path = os.path.join(ddpm_checkpoint_dir, syn_filename)
    synthetic, labels_syn = _load_hdf5_samples(syn_path, verbose=verbose, name="synthetic")

    # Load reconstructions
    reconstructed = None
    labels_recon = None
    if vae_checkpoint_dir:
        recon_path = os.path.join(vae_checkpoint_dir, recon_filename)
        if os.path.exists(recon_path):
            reconstructed, labels_recon = _load_hdf5_samples(
                recon_path, verbose=verbose, name="reconstructed"
            )
        elif verbose:
            print(f"  No reconstruction file found at {recon_path}")

    # Auto-detect conditioning type if not specified
    if conditioning_type is None:
        from snpgen.evaluation.utils import detect_task_type
        conditioning_type = detect_task_type(labels_train)

    if verbose:
        print(f"  Conditioning type: {conditioning_type}")

    return PrivacyDataBundle(
        real_train=real_train,
        real_holdout=real_holdout,
        synthetic=synthetic,
        reconstructed=reconstructed,
        labels_train=labels_train,
        labels_holdout=labels_holdout,
        labels_syn=labels_syn,
        labels_recon=labels_recon,
        model_name=model_name,
        ddpm_checkpoint_dir=ddpm_checkpoint_dir,
        vae_checkpoint_dir=vae_checkpoint_dir,
        conditioning_type=conditioning_type,
    )


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def _load_hdf5_samples(path, verbose=True, name="data"):
    """Load samples and targets from an HDF5 file.

    Supports both 'syn_samples'/'targets' and 'data'/'labels' key conventions.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    with h5py.File(path, 'r') as f:
        # Detect keys
        if 'syn_samples' in f:
            samples = f['syn_samples'][:]
        elif 'data' in f:
            samples = f['data'][:]
        else:
            raise KeyError(f"No 'syn_samples' or 'data' key in {path}. Keys: {list(f.keys())}")

        if 'targets' in f:
            targets = f['targets'][:]
        elif 'labels' in f:
            targets = f['labels'][:]
        else:
            targets = np.zeros(samples.shape[0], dtype=np.int8)
            if verbose:
                print(f"  Warning: no 'targets' or 'labels' key in {path}")

    samples = np.asarray(samples, dtype=np.int8)

    if verbose:
        print(f"  {name}: {samples.shape}, dtype={samples.dtype}")

    return samples, targets


def _extract_vae_dir(config):
    """Extract VAE checkpoint directory from a DDPM config."""
    try:
        vae_ckpt_path = config.model.params.first_stage_config.params.get('ckpt_path', None)
        if vae_ckpt_path is None:
            # Try top-level
            vae_ckpt_path = config.get('vae_ckpt_path', None)
        if vae_ckpt_path:
            return os.path.dirname(str(vae_ckpt_path))
    except (AttributeError, KeyError):
        pass

    # Try top-level vae_ckpt_path
    try:
        vae_ckpt_path = config.get('vae_ckpt_path', None)
        if vae_ckpt_path:
            return os.path.dirname(str(vae_ckpt_path))
    except (AttributeError, KeyError):
        pass

    return None


# ---------------------------------------------------------------------------
# Persistence (incremental saving)
# ---------------------------------------------------------------------------

RESULTS_FILENAME = 'privacy_results.pkl'
SUMMARY_FILENAME = 'privacy_summary.json'


def save_privacy_results(results, output_dir):
    """Save privacy results dict to disk (pickle + JSON summary).

    Args:
        results: Dict mapping metric keys to result dataclass instances.
        output_dir: Directory to save into (created if needed).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save full pickle
    pkl_path = os.path.join(output_dir, RESULTS_FILENAME)
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save JSON summary (scalars only)
    summary = {}
    for key, result in results.items():
        if hasattr(result, 'to_summary_dict'):
            summary[key] = result.to_summary_dict()
        elif isinstance(result, dict):
            summary[key] = {k: _to_json_safe(v) for k, v in result.items()}
        else:
            summary[key] = str(result)

    json_path = os.path.join(output_dir, SUMMARY_FILENAME)
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=_to_json_safe)


def load_privacy_results(output_dir):
    """Load existing privacy results from disk.

    Args:
        output_dir: Directory containing privacy_results.pkl.

    Returns:
        Dict of results, or empty dict if nothing saved yet.
    """
    pkl_path = os.path.join(output_dir, RESULTS_FILENAME)
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    return {}


def _to_json_safe(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    return str(obj)


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------

def subsample_by_label(data, labels, label_value):
    """Extract samples with a specific label value.

    Args:
        data: np.ndarray (N, D).
        labels: np.ndarray (N,).
        label_value: The label to filter for.

    Returns:
        Subset of data where labels == label_value.
    """
    mask = labels == label_value
    return data[mask]


def subsample_balanced(data, labels, n_per_class, seed=42):
    """Subsample data with balanced classes.

    Args:
        data: np.ndarray (N, D).
        labels: np.ndarray (N,).
        n_per_class: Number of samples per class.
        seed: Random seed.

    Returns:
        (subsampled_data, subsampled_labels)
    """
    rng = np.random.RandomState(seed)
    unique_labels = np.unique(labels)
    indices = []
    for label in unique_labels:
        label_idx = np.where(labels == label)[0]
        n = min(n_per_class, len(label_idx))
        chosen = rng.choice(label_idx, n, replace=False)
        indices.append(chosen)
    indices = np.concatenate(indices)
    rng.shuffle(indices)
    return data[indices], labels[indices]
