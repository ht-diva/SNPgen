"""Yelmen-compatible nearest-neighbour privacy and geometry diagnostics.

The published Yelmen privacy loss is ``AA_TS(test) - AA_TS(train)``.  It is
not the arithmetic-mean ``privacy_score`` historically exposed by SNPgen's
NNAA result.  This module keeps the names and protocols deliberately distinct.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .distances import batched_knn


@dataclass
class MatchedPrivacyCohorts:
    train: np.ndarray
    test: np.ndarray
    synthetic: np.ndarray
    class_counts: dict[str, int]
    seed: int


@dataclass
class YelmenPrivacyLossResult:
    train_aa_truth: float
    train_aa_syn: float
    train_aa_ts: float
    test_aa_truth: float
    test_aa_syn: float
    test_aa_ts: float
    privacy_loss: float
    n_samples_per_cohort: int
    class_counts: dict[str, int]
    metric: str
    seed: int

    def to_summary_dict(self):
        return {
            "train_aa_truth": float(self.train_aa_truth),
            "train_aa_syn": float(self.train_aa_syn),
            "train_aa_ts": float(self.train_aa_ts),
            "test_aa_truth": float(self.test_aa_truth),
            "test_aa_syn": float(self.test_aa_syn),
            "test_aa_ts": float(self.test_aa_ts),
            "privacy_loss": float(self.privacy_loss),
            "n_samples_per_cohort": int(self.n_samples_per_cohort),
            "class_counts": {str(k): int(v) for k, v in self.class_counts.items()},
            "metric": self.metric,
            "seed": int(self.seed),
        }


@dataclass
class NearestNeighborChainResult:
    pattern_counts: dict[str, dict[str, int]]
    pattern_frequencies: dict[str, dict[str, float]]
    expected_equal_mixture_frequency: dict[str, float]
    n_real: int
    n_synthetic: int
    min_length: int
    max_length: int
    metric: str
    neighbor_k: int
    tie_policy: str

    def to_summary_dict(self):
        return {
            "pattern_counts": self.pattern_counts,
            "pattern_frequencies": self.pattern_frequencies,
            "expected_equal_mixture_frequency": self.expected_equal_mixture_frequency,
            "n_real": int(self.n_real),
            "n_synthetic": int(self.n_synthetic),
            "min_length": int(self.min_length),
            "max_length": int(self.max_length),
            "metric": self.metric,
            "neighbor_k": int(self.neighbor_k),
            "tie_policy": self.tie_policy,
        }


def _sample_indices(indices: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if n > len(indices):
        raise ValueError(f"Cannot draw {n} records from a cohort of {len(indices)}")
    if n == len(indices):
        return indices.copy()
    return np.sort(rng.choice(indices, size=n, replace=False))


def matched_privacy_cohorts(
    real_train,
    real_test,
    synthetic,
    labels_train=None,
    labels_test=None,
    labels_synthetic=None,
    max_samples=50000,
    seed=42,
):
    """Deterministically select equal-sized train, test and synthetic cohorts.

    When labels are supplied, every selected cohort has identical per-class
    counts.  The common capacity of each class is retained unless the total
    exceeds ``max_samples``.
    """
    arrays = tuple(np.asarray(x) for x in (real_train, real_test, synthetic))
    if any(x.ndim != 2 for x in arrays):
        raise ValueError("Train, test and synthetic genotypes must be 2D arrays")
    if len({x.shape[1] for x in arrays}) != 1:
        raise ValueError("Train, test and synthetic cohorts must have the same SNP count")
    if min(len(x) for x in arrays) == 0:
        raise ValueError("Matched privacy cohorts cannot be empty")
    if max_samples is not None and int(max_samples) <= 0:
        raise ValueError("max_samples must be positive or None")

    label_sets = (labels_train, labels_test, labels_synthetic)
    have_labels = all(value is not None and len(value) for value in label_sets)
    rngs = [np.random.default_rng(int(seed) + offset) for offset in (0, 1, 2)]

    if not have_labels:
        n = min(len(x) for x in arrays)
        if max_samples is not None:
            n = min(n, int(max_samples))
        selected = [_sample_indices(np.arange(len(x)), n, rng) for x, rng in zip(arrays, rngs)]
        return MatchedPrivacyCohorts(
            *(x[idx] for x, idx in zip(arrays, selected)),
            class_counts={"all": int(n)},
            seed=int(seed),
        )

    labels = tuple(np.asarray(value).reshape(-1) for value in label_sets)
    for name, data, target in zip(("train", "test", "synthetic"), arrays, labels):
        if len(data) != len(target):
            raise ValueError(f"{name} genotypes and labels have different lengths")

    common_classes = sorted(set(labels[0]) & set(labels[1]) & set(labels[2]))
    if not common_classes:
        raise ValueError("Train, test and synthetic labels have no common classes")
    capacities = {
        value: min(int(np.sum(target == value)) for target in labels)
        for value in common_classes
    }
    capacities = {value: count for value, count in capacities.items() if count > 0}
    total_capacity = sum(capacities.values())
    target_total = total_capacity if max_samples is None else min(total_capacity, int(max_samples))

    # Largest-remainder allocation preserves the common class mixture when a cap applies.
    if target_total == total_capacity:
        allocation = capacities.copy()
    else:
        raw = {value: target_total * count / total_capacity for value, count in capacities.items()}
        allocation = {value: min(capacities[value], int(np.floor(raw[value]))) for value in capacities}
        for value in capacities:
            if allocation[value] == 0 and target_total >= len(capacities):
                allocation[value] = 1
        while sum(allocation.values()) > target_total:
            candidates = [v for v in capacities if allocation[v] > 1]
            allocation[min(candidates, key=lambda v: raw[v] - allocation[v])] -= 1
        while sum(allocation.values()) < target_total:
            candidates = [v for v in capacities if allocation[v] < capacities[v]]
            value = max(candidates, key=lambda v: raw[v] - allocation[v])
            allocation[value] += 1

    selected_arrays = []
    for data, target, rng in zip(arrays, labels, rngs):
        pieces = [
            _sample_indices(np.flatnonzero(target == value), allocation[value], rng)
            for value in capacities
            if allocation[value] > 0
        ]
        indices = np.sort(np.concatenate(pieces))
        selected_arrays.append(data[indices])

    return MatchedPrivacyCohorts(
        *selected_arrays,
        class_counts={str(value): int(allocation[value]) for value in capacities},
        seed=int(seed),
    )


def _nearest_other_distances(data, metric, knn_fn, **knn_kwargs):
    k = min(2, len(data))
    if k < 2:
        raise ValueError("AA_TS requires at least two records per cohort")
    distances, indices = knn_fn(data, data, k=k, metric=metric, **knn_kwargs)
    row_index = np.arange(len(data))
    first_is_self = indices[:, 0] == row_index
    return np.where(first_is_self, distances[:, 1], distances[:, 0]).astype(np.float64)


def adversarial_accuracy(truth, synthetic, metric='manhattan', knn_fn=None, **knn_kwargs):
    """Return ``(AA_truth, AA_syn, AA_TS)`` for equal-sized cohorts."""
    truth = np.asarray(truth)
    synthetic = np.asarray(synthetic)
    if truth.shape != synthetic.shape:
        raise ValueError(f"AA_TS requires equal cohort shapes, got {truth.shape} and {synthetic.shape}")
    _knn = knn_fn or batched_knn
    d_ts = _knn(truth, synthetic, k=1, metric=metric, **knn_kwargs)[0][:, 0]
    d_tt = _nearest_other_distances(truth, metric, _knn, **knn_kwargs)
    d_st = _knn(synthetic, truth, k=1, metric=metric, **knn_kwargs)[0][:, 0]
    d_ss = _nearest_other_distances(synthetic, metric, _knn, **knn_kwargs)
    aa_truth = float(np.mean(d_ts > d_tt))
    aa_syn = float(np.mean(d_st > d_ss))
    return aa_truth, aa_syn, float((aa_truth + aa_syn) / 2.0)


def yelmen_privacy_loss(matched, metric='manhattan', knn_fn=None, **knn_kwargs):
    """Compute Yelmen's published ``AA_TS(test) - AA_TS(train)`` quantity."""
    train_values = adversarial_accuracy(
        matched.train, matched.synthetic, metric=metric, knn_fn=knn_fn, **knn_kwargs
    )
    test_values = adversarial_accuracy(
        matched.test, matched.synthetic, metric=metric, knn_fn=knn_fn, **knn_kwargs
    )
    return YelmenPrivacyLossResult(
        train_aa_truth=train_values[0],
        train_aa_syn=train_values[1],
        train_aa_ts=train_values[2],
        test_aa_truth=test_values[0],
        test_aa_syn=test_values[1],
        test_aa_ts=test_values[2],
        privacy_loss=float(test_values[2] - train_values[2]),
        n_samples_per_cohort=len(matched.train),
        class_counts=matched.class_counts,
        metric=metric,
        seed=matched.seed,
    )


def nearest_neighbor_chain_analysis(
    real,
    synthetic,
    min_length=2,
    max_length=5,
    metric='hamming',
    neighbor_k=32,
    knn_fn=None,
    **knn_kwargs,
):
    """Count Yelmen-style T/S nearest-neighbour chains without revisiting nodes."""
    real = np.asarray(real)
    synthetic = np.asarray(synthetic)
    if real.ndim != 2 or synthetic.ndim != 2 or real.shape[1] != synthetic.shape[1]:
        raise ValueError("Real and synthetic cohorts must be 2D with equal SNP counts")
    if len(real) != len(synthetic):
        raise ValueError("Chain analysis requires equal real and synthetic cohort sizes")
    if min_length < 2 or max_length < min_length:
        raise ValueError("Require 2 <= min_length <= max_length")

    combined = np.concatenate((real, synthetic), axis=0)
    if len(combined) < max_length:
        raise ValueError("Combined cohort is too small for the requested chain length")
    source = np.concatenate((np.full(len(real), "T"), np.full(len(synthetic), "S")))
    k = min(len(combined), max(int(neighbor_k), max_length + 1))
    _knn = knn_fn or batched_knn
    distances, indices = _knn(combined, combined, k=k, metric=metric, **knn_kwargs)
    expected_shape = (len(combined), k)
    if distances.shape != expected_shape or indices.shape != expected_shape:
        raise RuntimeError(
            "Nearest-neighbour chain analysis requires full-cohort kNN results; "
            f"expected {expected_shape}, got distances={distances.shape}, "
            f"indices={indices.shape}. Use a GPU or explicitly reduce max_samples."
        )

    # Stabilize ordering within the returned candidate set by distance then index.
    for row in range(len(combined)):
        order = np.lexsort((indices[row], distances[row]))
        indices[row] = indices[row, order]

    counts = {
        str(length): {"".join(bits): 0 for bits in product("ST", repeat=length)}
        for length in range(min_length, max_length + 1)
    }
    for start in range(len(combined)):
        visited = {start}
        current = start
        pattern = source[start]
        for length in range(2, max_length + 1):
            next_node = next((int(idx) for idx in indices[current] if int(idx) not in visited), None)
            if next_node is None:
                raise RuntimeError(
                    f"No unvisited neighbour found with k={k}; increase neighbor_k"
                )
            visited.add(next_node)
            current = next_node
            pattern += source[current]
            if length >= min_length:
                counts[str(length)][pattern] += 1

    frequencies = {
        length: {pattern: count / len(combined) for pattern, count in values.items()}
        for length, values in counts.items()
    }
    expected = {str(length): float(0.5 ** length) for length in range(min_length, max_length + 1)}
    return NearestNeighborChainResult(
        pattern_counts=counts,
        pattern_frequencies=frequencies,
        expected_equal_mixture_frequency=expected,
        n_real=len(real),
        n_synthetic=len(synthetic),
        min_length=min_length,
        max_length=max_length,
        metric=metric,
        neighbor_k=k,
        tie_policy="distance_then_index_within_returned_knn_set",
    )
