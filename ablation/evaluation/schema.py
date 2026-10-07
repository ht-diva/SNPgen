"""Standardized HDF5 schema for ablation-generated datasets.

The main SNPgen notebooks use both ``syn_samples``/``targets`` and
``data``/``labels`` conventions.  Ablation outputs write both key pairs so the
same file can be used by the existing inference, privacy, and downstream
evaluation loaders.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import h5py
import numpy as np


@dataclass(frozen=True)
class StandardDataset:
    """Container for standardized genotype datasets."""

    samples: np.ndarray
    labels: np.ndarray
    attrs: Dict[str, Any]


def _as_numpy_int8(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError(f"Expected samples with shape (N, L), got {samples.shape}")
    if not np.isin(samples, [0, 1, 2]).all():
        bad = np.unique(samples[~np.isin(samples, [0, 1, 2])])
        raise ValueError(f"Samples contain values outside {{0,1,2}}: {bad[:10]}")
    return samples.astype(np.int8, copy=False)


def _as_labels(labels: np.ndarray, n_samples: int) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if labels.shape[0] != n_samples:
        raise ValueError(f"Expected {n_samples} labels, got {labels.shape[0]}")
    if np.issubdtype(labels.dtype, np.integer):
        return labels.astype(np.int32, copy=False)
    return labels.astype(np.float32, copy=False)


def save_standard_hdf5(
    output_path: str,
    samples: np.ndarray,
    labels: np.ndarray,
    attrs: Optional[Dict[str, Any]] = None,
    compression: str = "gzip",
) -> str:
    """Write ablation output using all repo-compatible dataset keys."""

    samples = _as_numpy_int8(samples)
    labels = _as_labels(labels, samples.shape[0])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("syn_samples", data=samples, compression=compression)
        hf.create_dataset("targets", data=labels, compression=compression)
        hf.create_dataset("data", data=samples, compression=compression)
        hf.create_dataset("labels", data=labels, compression=compression)
        hf.attrs["schema"] = "snpgen_ablation_v1"
        hf.attrs["n_samples"] = int(samples.shape[0])
        hf.attrs["n_snps"] = int(samples.shape[1])

        for key, value in (attrs or {}).items():
            if value is None:
                continue
            hf.attrs[key] = value

    return output_path


def load_standard_hdf5(file_path: str) -> StandardDataset:
    """Load either SNPgen synthetic or raw HDF5 genotype files."""

    with h5py.File(file_path, "r") as hf:
        if "syn_samples" in hf:
            samples = hf["syn_samples"][:]
        elif "data" in hf:
            samples = hf["data"][:]
        else:
            raise KeyError(f"No 'syn_samples' or 'data' key in {file_path}")

        if "targets" in hf:
            labels = hf["targets"][:]
        elif "labels" in hf:
            labels = hf["labels"][:]
        else:
            raise KeyError(f"No 'targets' or 'labels' key in {file_path}")

        attrs = dict(hf.attrs)

    return StandardDataset(
        samples=_as_numpy_int8(samples),
        labels=_as_labels(labels, samples.shape[0]),
        attrs=attrs,
    )


def validate_standard_hdf5(file_path: str) -> StandardDataset:
    """Load and validate an ablation HDF5 file."""

    data = load_standard_hdf5(file_path)
    required = {"syn_samples", "targets", "data", "labels"}
    with h5py.File(file_path, "r") as hf:
        missing = sorted(required.difference(hf.keys()))
    if missing:
        raise KeyError(f"Missing standardized keys in {file_path}: {missing}")
    return data
