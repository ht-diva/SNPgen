"""Shared cohort loading and matched-sampling protocol for scientific audits."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import numpy as np
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Cohort:
    """A named genotype/phenotype cohort loaded from HDF5."""

    name: str
    path: str
    samples: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class CohortSelection:
    """A cohort path plus deterministic selected rows, before loading genotypes."""

    name: str
    path: str
    indices: np.ndarray
    labels: np.ndarray


def parse_cohort_specs(specs: Iterable[str]) -> dict[str, str]:
    """Parse repeatable ``NAME=HDF5`` command-line values."""

    cohorts: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Cohort must be NAME=HDF5, got: {spec!r}")
        name, path = spec.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f"Cohort must be NAME=HDF5, got: {spec!r}")
        if name == "real":
            raise ValueError("'real' is reserved for the reference cohort")
        if name in cohorts:
            raise ValueError(f"Duplicate cohort name: {name}")
        cohorts[name] = path
    if not cohorts:
        raise ValueError("At least one --cohort NAME=HDF5 is required")
    return cohorts


def _dataset_keys(hf: h5py.File) -> tuple[str, str]:
    sample_key = "syn_samples" if "syn_samples" in hf else "data"
    label_key = "targets" if "targets" in hf else "labels"
    if sample_key not in hf:
        raise KeyError(f"No genotype dataset ('syn_samples' or 'data') in {hf.filename}")
    if label_key not in hf:
        raise KeyError(f"No phenotype dataset ('targets' or 'labels') in {hf.filename}")
    return sample_key, label_key


def read_labels(path: str) -> np.ndarray:
    with h5py.File(path, "r") as hf:
        _, label_key = _dataset_keys(hf)
        labels = np.asarray(hf[label_key][:]).reshape(-1)
    return labels


def read_samples(path: str, indices: np.ndarray | None = None) -> np.ndarray:
    """Read dosage samples, optionally at sorted row indices."""

    with h5py.File(path, "r") as hf:
        sample_key, _ = _dataset_keys(hf)
        ds = hf[sample_key]
        samples = ds[:] if indices is None else ds[np.asarray(indices, dtype=np.int64)]
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError(f"Expected a two-dimensional genotype matrix in {path}, got {samples.shape}")
    if not np.isin(samples, [0, 1, 2]).all():
        bad = np.unique(samples[~np.isin(samples, [0, 1, 2])])
        raise ValueError(f"Unsupported genotype values in {path}: {bad[:10]}")
    return samples.astype(np.float32, copy=False)


def sample_shape(path: str) -> tuple[int, int]:
    with h5py.File(path, "r") as hf:
        sample_key, _ = _dataset_keys(hf)
        shape = tuple(hf[sample_key].shape)
    if len(shape) != 2:
        raise ValueError(f"Expected a two-dimensional genotype matrix in {path}, got {shape}")
    return int(shape[0]), int(shape[1])


def iter_selected_sample_chunks(path: str, indices: np.ndarray, chunk_size: int):
    """Yield validated selected HDF5 rows without loading a full cohort."""

    indices = np.asarray(indices, dtype=np.int64)
    if np.any(np.diff(indices) <= 0):
        raise ValueError("Selected HDF5 indices must be strictly increasing")
    with h5py.File(path, "r") as hf:
        sample_key, _ = _dataset_keys(hf)
        dataset = hf[sample_key]
        for start in range(0, indices.size, chunk_size):
            rows = indices[start : start + chunk_size]
            samples = np.asarray(dataset[rows])
            if not np.isin(samples, [0, 1, 2]).all():
                bad = np.unique(samples[~np.isin(samples, [0, 1, 2])])
                raise ValueError(f"Unsupported genotype values in {path}: {bad[:10]}")
            yield samples


def read_panel_metadata(path: str, n_snps: int) -> dict[str, np.ndarray]:
    """Read variant metadata from the real HDF5, preserving original order."""

    aliases = {
        "snp_id": ("snp_ids", "snp_id"),
        "chrom": ("chrom", "chromosome", "chromosomes"),
        "pos": ("pos", "position", "positions"),
        "metadata_beta": ("betas", "beta"),
    }
    result: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as hf:
        for output_name, candidates in aliases.items():
            for key in candidates:
                if key in hf and hf[key].shape[0] == n_snps:
                    result[output_name] = np.asarray(hf[key][:])
                    break
    result.setdefault("snp_id", np.asarray([f"snp_{i}" for i in range(n_snps)]))
    # Missing physical coordinates must stay visibly missing; inventing an
    # index-based distance would make an LD-decay curve look genomic when it is
    # not. Pairwise LD remains available without coordinates.
    result.setdefault("chrom", np.full(n_snps, "", dtype=object))
    result.setdefault("pos", np.full(n_snps, np.nan, dtype=float))
    return result


def panel_order_hash(metadata: Mapping[str, np.ndarray], n_snps: int) -> str:
    """Return a stable hash of the reference variant order."""

    digest = hashlib.sha256()
    for key in ("snp_id", "chrom", "pos"):
        values = np.asarray(metadata.get(key, np.arange(n_snps))).reshape(-1)
        digest.update(key.encode())
        digest.update(b"\0")
        digest.update("\n".join(map(str, values.tolist())).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def reference_from_config(config_path: str, split: str) -> tuple[str, np.ndarray, np.ndarray, int]:
    """Resolve the raw HDF5 and reproduce SNPgen's outer train/test split."""

    config = OmegaConf.load(config_path)
    real_path = str(config.dataset_path)
    seed = int(config.get("seed", 42))
    test_ratio = float(config.data.raw_dataset.params.get("test_ratio", 0.1))
    labels = read_labels(real_path)
    all_indices = np.arange(labels.shape[0], dtype=np.int64)
    train_val, test = train_test_split(
        all_indices,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels,
    )
    if split == "train_val":
        indices = np.sort(train_val)
    elif split == "test":
        indices = np.sort(test)
    elif split == "full":
        indices = all_indices
    else:
        raise ValueError(f"Unknown reference split: {split}")
    return real_path, indices, labels[indices], seed


def matched_class_counts(
    label_sets: Mapping[str, np.ndarray],
    reference_name: str = "real",
    max_samples: int | None = None,
) -> dict[int, int]:
    """Choose common per-class counts, preserving reference prevalence when capped."""

    classes = np.unique(label_sets[reference_name])
    if classes.size != 2 or not np.array_equal(classes, classes.astype(int)):
        raise ValueError(f"This protocol currently requires binary integer labels; found {classes}")
    classes = classes.astype(int)
    for name, labels in label_sets.items():
        present = np.unique(labels).astype(int)
        if not np.array_equal(present, classes):
            raise ValueError(f"Cohort {name} has classes {present}, expected {classes}")

    available = {
        int(cls): min(int(np.sum(np.asarray(labels) == cls)) for labels in label_sets.values())
        for cls in classes
    }
    if max_samples is None or sum(available.values()) <= max_samples:
        return available

    ref = np.asarray(label_sets[reference_name])
    proportions = {int(cls): float(np.mean(ref == cls)) for cls in classes}
    target = {int(cls): min(available[int(cls)], int(np.floor(max_samples * proportions[int(cls)]))) for cls in classes}
    # Allocate rounding residue without exceeding any cohort's availability.
    while sum(target.values()) < max_samples:
        candidates = [int(cls) for cls in classes if target[int(cls)] < available[int(cls)]]
        if not candidates:
            break
        cls = max(candidates, key=lambda c: proportions[c] - target[c] / max_samples)
        target[cls] += 1
    if any(count == 0 for count in target.values()):
        raise ValueError(f"max_samples={max_samples} yields an empty class: {target}")
    return target


def select_indices_by_class(labels: np.ndarray, counts: Mapping[int, int], seed: int) -> np.ndarray:
    """Select an exact number from every class, deterministically and without replacement."""

    labels = np.asarray(labels).reshape(-1)
    rng = np.random.default_rng(seed)
    selected = []
    for cls, count in sorted(counts.items()):
        candidates = np.flatnonzero(labels == cls)
        if candidates.size < count:
            raise ValueError(f"Requested {count} samples for class {cls}; only {candidates.size} available")
        selected.append(rng.choice(candidates, size=count, replace=False))
    return np.sort(np.concatenate(selected).astype(np.int64))


def prepare_matched_cohort_selections(
    reference_config: str,
    cohort_paths: Mapping[str, str],
    reference_split: str,
    seed: int | None,
    max_samples: int | None,
) -> tuple[dict[str, CohortSelection], dict[str, object]]:
    """Resolve paths and deterministic matched rows without loading genotypes."""

    real_path, real_base_indices, real_labels, config_seed = reference_from_config(reference_config, reference_split)
    chosen_seed = config_seed if seed is None else int(seed)
    labels_by_name = {"real": real_labels}
    for name, path in cohort_paths.items():
        labels_by_name[name] = read_labels(path)
    counts = matched_class_counts(labels_by_name, max_samples=max_samples)

    selections: dict[str, CohortSelection] = {}
    real_relative = select_indices_by_class(real_labels, counts, chosen_seed)
    real_indices = np.sort(real_base_indices[real_relative])
    all_real_labels = read_labels(real_path)
    selections["real"] = CohortSelection("real", real_path, real_indices, all_real_labels[real_indices])

    expected_snps = sample_shape(real_path)[1]
    for offset, (name, path) in enumerate(cohort_paths.items(), start=1):
        labels = labels_by_name[name]
        indices = select_indices_by_class(labels, counts, chosen_seed + offset)
        cohort_snps = sample_shape(path)[1]
        if cohort_snps != expected_snps:
            raise ValueError(
                f"SNP-count mismatch: real has {expected_snps}, {name} has {cohort_snps} ({path}). "
                "Generated files lack sufficient metadata to prove SNP order, so matching dimensions are mandatory."
            )
        selections[name] = CohortSelection(name, path, indices, labels[indices])

    metadata = read_panel_metadata(real_path, expected_snps)
    manifest = {
        "reference_config": str(Path(reference_config).resolve()),
        "reference_hdf5": str(Path(real_path).resolve()),
        "reference_split": reference_split,
        "seed": chosen_seed,
        "target_class_counts": {str(k): int(v) for k, v in counts.items()},
        "n_snps": expected_snps,
        "panel_order_hash": panel_order_hash(metadata, expected_snps),
        "physical_coordinates_available": bool(
            np.all(np.asarray(metadata["chrom"]).astype(str) != "")
            and np.all(np.isfinite(np.asarray(metadata["pos"], dtype=float)))
        ),
        "panel_order_note": "Synthetic panel identity/order is inherited from the reference config and cannot be independently verified from generated HDF5 files.",
        "cohorts": {name: str(Path(selection.path).resolve()) for name, selection in selections.items()},
    }
    return selections, {"manifest": manifest, "metadata": metadata}


def load_matched_cohorts(
    reference_config: str,
    cohort_paths: Mapping[str, str],
    reference_split: str,
    seed: int | None,
    max_samples: int | None,
) -> tuple[dict[str, Cohort], dict[str, object]]:
    """Load reference and generated cohorts under one class-count protocol."""

    selections, provenance = prepare_matched_cohort_selections(
        reference_config, cohort_paths, reference_split, seed, max_samples
    )
    cohorts = {
        name: Cohort(name, selection.path, read_samples(selection.path, selection.indices), selection.labels)
        for name, selection in selections.items()
    }
    return cohorts, provenance
