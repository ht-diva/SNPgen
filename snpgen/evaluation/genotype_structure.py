"""Matched genotype-structure diagnostics for real and synthetic cohorts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from sklearn.decomposition import PCA


def allele_frequencies(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return alternate-allele frequency and true minor-allele frequency."""

    samples = np.asarray(samples, dtype=float)
    af = samples.mean(axis=0) / 2.0
    return af, np.minimum(af, 1.0 - af)


def genotype_counts(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples)
    return np.stack([(samples == dosage).sum(axis=0) for dosage in (0, 1, 2)], axis=1).astype(np.int64)


def hwe_exact_p_value(hom0: int, het: int, hom2: int) -> float:
    """Exact Hardy--Weinberg p-value (Wigginton et al. recurrence)."""

    obs_homc = min(int(hom0), int(hom2))
    obs_homr = max(int(hom0), int(hom2))
    obs_hets = int(het)
    rare_copies = 2 * obs_homc + obs_hets
    n = obs_homc + obs_homr + obs_hets
    if n == 0 or rare_copies == 0 or rare_copies == 2 * n:
        return 1.0
    probabilities = np.zeros(rare_copies + 1, dtype=float)
    midpoint = int(rare_copies * (2 * n - rare_copies) / (2 * n))
    if (midpoint & 1) != (rare_copies & 1):
        midpoint += 1
    probabilities[midpoint] = 1.0
    total = 1.0

    curr_hets = midpoint
    curr_homr = (rare_copies - midpoint) // 2
    curr_homc = n - curr_hets - curr_homr
    while curr_hets >= 2:
        probability = probabilities[curr_hets] * curr_hets * (curr_hets - 1.0) / (4.0 * (curr_homr + 1.0) * (curr_homc + 1.0))
        probabilities[curr_hets - 2] = probability
        total += probability
        curr_hets -= 2
        curr_homr += 1
        curr_homc += 1

    curr_hets = midpoint
    curr_homr = (rare_copies - midpoint) // 2
    curr_homc = n - curr_hets - curr_homr
    while curr_hets <= rare_copies - 2:
        probability = probabilities[curr_hets] * 4.0 * curr_homr * curr_homc / ((curr_hets + 2.0) * (curr_hets + 1.0))
        probabilities[curr_hets + 2] = probability
        total += probability
        curr_hets += 2
        curr_homr -= 1
        curr_homc -= 1

    probabilities /= total
    observed = probabilities[obs_hets]
    return float(min(1.0, probabilities[probabilities <= observed + 1e-12].sum()))


def hwe_statistics(samples: np.ndarray) -> pd.DataFrame:
    counts = genotype_counts(samples)
    n = counts.sum(axis=1).astype(float)
    af = (counts[:, 1] + 2.0 * counts[:, 2]) / (2.0 * n)
    observed_het = counts[:, 1] / n
    expected_het = 2.0 * af * (1.0 - af)
    hwe_f = np.divide(
        expected_het - observed_het,
        expected_het,
        out=np.full_like(expected_het, np.nan),
        where=expected_het > 0,
    )
    p_values = np.array([hwe_exact_p_value(*row) for row in counts], dtype=float)
    return pd.DataFrame(
        {
            "n0": counts[:, 0],
            "n1": counts[:, 1],
            "n2": counts[:, 2],
            "hwe_f": hwe_f,
            "hwe_exact_p": p_values,
        }
    )


def pairwise_ld_r2(samples: np.ndarray) -> np.ndarray:
    """Compute the dosage Pearson-r-squared matrix, retaining undefined pairs as NaN."""

    x = np.asarray(samples, dtype=np.float32)
    centered = x - x.mean(axis=0, keepdims=True)
    norm = np.sqrt(np.sum(centered * centered, axis=0))
    valid = norm > 0
    standardized = np.zeros_like(centered, dtype=np.float32)
    standardized[:, valid] = centered[:, valid] / norm[valid]
    correlation = standardized.T @ standardized
    r2 = np.square(np.clip(correlation, -1.0, 1.0), dtype=np.float32)
    r2[~valid, :] = np.nan
    r2[:, ~valid] = np.nan
    return r2


def pair_cache(chrom: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return same-chromosome upper-triangle pairs with positive physical distance."""

    chrom = np.asarray(chrom).astype(str)
    pos = np.asarray(pos, dtype=float)
    if np.any(chrom == "") or not np.all(np.isfinite(pos)):
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
    left, right = np.triu_indices(pos.size, k=1)
    distance = np.abs(pos[right] - pos[left])
    keep = (chrom[left] == chrom[right]) & (distance > 0)
    return left[keep], right[keep], distance[keep]


def ld_decay_table(
    matrices: dict[str, np.ndarray],
    chrom: np.ndarray,
    pos: np.ndarray,
    n_bins: int = 50,
) -> pd.DataFrame:
    left, right, distance = pair_cache(chrom, pos)
    columns = ["cohort", "distance_left", "distance_right", "distance_center", "n_pairs", "n_valid", "mean_r2", "se_r2"]
    if distance.size == 0:
        return pd.DataFrame(columns=columns)
    minimum = max(1.0, float(distance.min()))
    maximum = float(distance.max())
    edges = np.geomspace(minimum, maximum, n_bins + 1) if maximum > minimum else np.array([minimum, maximum + 1.0])
    rows = []
    bin_index = np.clip(np.digitize(distance, edges) - 1, 0, edges.size - 2)
    for name, matrix in matrices.items():
        values = matrix[left, right]
        for index in range(edges.size - 1):
            selected = bin_index == index
            valid = selected & np.isfinite(values)
            n_valid = int(valid.sum())
            rows.append(
                {
                    "cohort": name,
                    "distance_left": float(edges[index]),
                    "distance_right": float(edges[index + 1]),
                    "distance_center": float(np.sqrt(edges[index] * edges[index + 1])),
                    "n_pairs": int(selected.sum()),
                    "n_valid": n_valid,
                    "mean_r2": float(np.mean(values[valid])) if n_valid else np.nan,
                    "se_r2": float(np.std(values[valid], ddof=1) / np.sqrt(n_valid)) if n_valid > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=columns)


@dataclass(frozen=True)
class ReferencePCA:
    mean: np.ndarray
    scale: np.ndarray
    model: PCA


def fit_reference_pca(samples: np.ndarray, n_components: int, seed: int) -> tuple[ReferencePCA, np.ndarray]:
    x = np.asarray(samples, dtype=np.float32)
    mean = x.mean(axis=0)
    af = mean / 2.0
    scale = np.sqrt(2.0 * af * (1.0 - af))
    scale[scale <= 1e-8] = 1.0
    standardized = (x - mean) / scale
    n_components = min(n_components, standardized.shape[0] - 1, standardized.shape[1])
    model = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    scores = model.fit_transform(standardized)
    return ReferencePCA(mean, scale, model), scores


def transform_reference_pca(samples: np.ndarray, state: ReferencePCA) -> np.ndarray:
    return state.model.transform((np.asarray(samples, dtype=np.float32) - state.mean) / state.scale)


def compare_ld(reference: np.ndarray, candidate: np.ndarray, chrom: np.ndarray, pos: np.ndarray) -> dict[str, float | int | str]:
    left, right, distance = pair_cache(chrom, pos)
    scope = "same_chromosome" if left.size else "all_pairs_no_physical_coordinates"
    if left.size == 0:
        left, right = np.triu_indices(reference.shape[0], k=1)
    real = reference[left, right]
    synthetic = candidate[left, right]
    valid = np.isfinite(real) & np.isfinite(synthetic)
    real, synthetic = real[valid], synthetic[valid]
    result: dict[str, float | int | str] = {"ld_pair_scope": scope, "ld_n_valid_pairs": int(valid.sum())}
    if real.size >= 2 and np.std(real) > 0 and np.std(synthetic) > 0:
        result["ld_pearson_r"] = float(pearsonr(real, synthetic).statistic)
        result["ld_spearman_rho"] = float(spearmanr(real, synthetic).statistic)
    else:
        result["ld_pearson_r"] = np.nan
        result["ld_spearman_rho"] = np.nan
    denominator = float(np.dot(real, real))
    result["ld_slope_origin"] = float(np.dot(real, synthetic) / denominator) if denominator else np.nan
    result["ld_mae"] = float(np.mean(np.abs(real - synthetic))) if real.size else np.nan
    result["ld_rmse"] = float(np.sqrt(np.mean((real - synthetic) ** 2))) if real.size else np.nan
    result["ld_mean_r2"] = float(np.mean(synthetic)) if synthetic.size else np.nan
    if scope == "same_chromosome":
        valid_distance = np.isfinite(reference[left, right]) & np.isfinite(candidate[left, right])
        candidate_values = candidate[left, right][valid_distance]
        log_distance = np.log10(distance[valid_distance])
        result["ld_distance_spearman_rho"] = (
            float(spearmanr(log_distance, candidate_values).statistic)
            if candidate_values.size >= 2 and np.std(candidate_values) > 0
            else np.nan
        )
    else:
        result["ld_distance_spearman_rho"] = np.nan
    return result


def pca_comparison(reference_scores: np.ndarray, candidate_scores: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for component in range(reference_scores.shape[1]):
        real = reference_scores[:, component]
        synthetic = candidate_scores[:, component]
        lower, upper = np.quantile(real, [0.01, 0.99])
        rows.append(
            {
                "pc": component + 1,
                "real_mean": float(np.mean(real)),
                "cohort_mean": float(np.mean(synthetic)),
                "real_sd": float(np.std(real, ddof=1)),
                "cohort_sd": float(np.std(synthetic, ddof=1)),
                "wasserstein": float(wasserstein_distance(real, synthetic)),
                "wasserstein_over_real_sd": float(wasserstein_distance(real, synthetic) / np.std(real, ddof=1)) if np.std(real, ddof=1) else np.nan,
                "outside_real_1_99_fraction": float(np.mean((synthetic < lower) | (synthetic > upper))),
            }
        )
    # np.cov returns a scalar for a one-component PCA; normalize to matrix
    # shape so the Frobenius norm remains defined for that valid case.
    real_cov = np.atleast_2d(np.cov(reference_scores, rowvar=False))
    syn_cov = np.atleast_2d(np.cov(candidate_scores, rowvar=False))
    denominator = np.linalg.norm(real_cov, ord="fro")
    summary = {"pca_covariance_relative_frobenius": float(np.linalg.norm(syn_cov - real_cov, ord="fro") / denominator) if denominator else np.nan}
    return pd.DataFrame(rows), summary
