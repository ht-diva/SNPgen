"""Class-conditional allele-frequency baseline.

This baseline intentionally ignores LD and samples each SNP independently from
the empirical class-specific genotype frequencies.  It is a useful lower bound:
if a complex generator only matches this model, it is mostly preserving marginal
case/control frequency shifts rather than haplotypic structure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

from ablation.evaluation.schema import save_standard_hdf5
from snpgen.data.loader import SplitDataset


@dataclass
class ClassConditionalAFSampler:
    """Independent per-SNP genotype sampler fitted separately per class."""

    alpha: float = 1.0
    classes_: Optional[np.ndarray] = None
    probs_: Optional[np.ndarray] = None

    def fit(self, genotypes: np.ndarray, labels: np.ndarray) -> "ClassConditionalAFSampler":
        genotypes = np.asarray(genotypes)
        labels = np.asarray(labels).reshape(-1)
        if genotypes.ndim != 2:
            raise ValueError(f"Expected genotypes with shape (N, L), got {genotypes.shape}")
        if genotypes.shape[0] != labels.shape[0]:
            raise ValueError("genotypes and labels have incompatible sample counts")
        if not np.isin(genotypes, [0, 1, 2]).all():
            raise ValueError("genotypes must contain only 0, 1, and 2")

        classes = np.unique(labels)
        probs = np.zeros((len(classes), genotypes.shape[1], 3), dtype=np.float64)
        for class_idx, class_value in enumerate(classes):
            x_class = genotypes[labels == class_value]
            if x_class.shape[0] == 0:
                raise ValueError(f"No samples for class {class_value}")
            for genotype_value in range(3):
                probs[class_idx, :, genotype_value] = np.sum(x_class == genotype_value, axis=0)
            probs[class_idx] += self.alpha
            probs[class_idx] /= probs[class_idx].sum(axis=1, keepdims=True)

        self.classes_ = classes
        self.probs_ = probs
        return self

    def sample(self, labels: Iterable, seed: int = 42) -> np.ndarray:
        if self.classes_ is None or self.probs_ is None:
            raise RuntimeError("Sampler has not been fitted")

        labels = np.asarray(list(labels)).reshape(-1)
        rng = np.random.default_rng(seed)
        out = np.empty((labels.shape[0], self.probs_.shape[1]), dtype=np.int8)
        class_to_idx = {value: idx for idx, value in enumerate(self.classes_)}

        for row_idx, label in enumerate(labels):
            if label not in class_to_idx:
                raise ValueError(f"Requested label {label!r} was not observed during fitting")
            p = self.probs_[class_to_idx[label]]
            u = rng.random(p.shape[0])
            cdf = np.cumsum(p, axis=1)
            out[row_idx] = (u[:, None] > cdf[:, :2]).sum(axis=1)

        return out


def _labels_for_strategy(train_labels: np.ndarray, strategy: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    train_labels = np.asarray(train_labels).reshape(-1)
    classes, counts = np.unique(train_labels, return_counts=True)

    if strategy == "matched":
        return train_labels.copy()
    if strategy == "prevalence":
        return rng.choice(classes, size=train_labels.shape[0], p=counts / counts.sum())
    if strategy == "balanced":
        n_per_class = int(counts.max())
        return np.repeat(classes, n_per_class)
    raise ValueError(f"Unknown label strategy: {strategy}")


def generate_from_real_hdf5(
    real_h5: str,
    output_h5: str,
    *,
    seed: int = 42,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    label_strategy: str = "matched",
    x_key: str = "data",
    y_key: str = "labels",
    attrs: Optional[Dict] = None,
) -> str:
    """Fit on the real training split and write a synthetic HDF5 dataset."""

    raw = SplitDataset(
        file_path=real_h5,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        onehot=False,
        data_dtype=np.int8,
        channel_first=False,
        metadata=False,
        x_key=x_key,
        y_key=y_key,
        seed=seed,
        verbose=False,
    )
    x_train, y_train = raw.get_split("train", metadata=False)
    labels = _labels_for_strategy(y_train, label_strategy, seed)
    sampler = ClassConditionalAFSampler().fit(x_train, y_train)
    samples = sampler.sample(labels, seed=seed)
    attrs = {
        "model_name": "class_conditional_af_sampler",
        "label_strategy": label_strategy,
        "train_seed": seed,
        "generation_seed": seed,
        **(attrs or {}),
    }
    return save_standard_hdf5(output_h5, samples=samples, labels=labels, attrs=attrs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-h5", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--label-strategy", choices=["matched", "prevalence", "balanced"], default="matched")
    parser.add_argument("--x-key", default="data")
    parser.add_argument("--y-key", default="labels")
    args = parser.parse_args()

    generate_from_real_hdf5(
        args.real_h5,
        args.output_h5,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        label_strategy=args.label_strategy,
        x_key=args.x_key,
        y_key=args.y_key,
    )


if __name__ == "__main__":
    main()
