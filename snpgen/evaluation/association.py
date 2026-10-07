"""Univariate genotype--phenotype association utilities.

The binary-trait fitter uses genotype-grouped binomial likelihoods.  For an
additive dosage model this is exactly equivalent to fitting an intercept and
one dosage coefficient to every individual, while avoiding repeated expansion
of hundreds of thousands of rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm, pearsonr, spearmanr


@dataclass(frozen=True)
class LogisticFitOptions:
    max_iter: int = 100
    tolerance: float = 1e-9
    coefficient_limit: float = 30.0
    information_tolerance: float = 1e-12


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini--Hochberg adjusted p-values, preserving NaNs."""

    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full(p_values.shape, np.nan, dtype=float)
    valid = np.isfinite(p_values)
    p = p_values[valid]
    if p.size == 0:
        return q_values
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * p.size / np.arange(1, p.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    q_values[valid] = restored
    return q_values


def genotype_case_counts(samples: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return total and case counts for genotypes 0, 1, and 2 at each SNP."""

    x = np.asarray(samples)
    y = np.asarray(labels).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"Incompatible sample/label shapes: {x.shape}, {y.shape}")
    if not np.isin(x, [0, 1, 2]).all():
        raise ValueError("Association fitting requires integer dosages in {0,1,2}")
    if not np.array_equal(np.unique(y), np.array([0, 1])):
        raise ValueError(f"Binary association fitting requires labels 0/1, got {np.unique(y)}")
    case_samples = x[y == 1]
    totals = np.stack([(x == dosage).sum(axis=0) for dosage in (0, 1, 2)], axis=1).astype(float)
    cases = np.stack([(case_samples == dosage).sum(axis=0) for dosage in (0, 1, 2)], axis=1).astype(float)
    return totals, cases


def _grouped_log_likelihood(theta: np.ndarray, totals: np.ndarray, cases: np.ndarray) -> float:
    eta = theta[0] + theta[1] * np.arange(3, dtype=float)
    return float(np.sum(cases * eta - totals * np.logaddexp(0.0, eta)))


def _fit_grouped_logistic_snp(
    totals: np.ndarray,
    cases: np.ndarray,
    options: LogisticFitOptions,
) -> tuple[float, float, float, float, int, str]:
    controls = totals - cases
    if np.count_nonzero(totals) < 2:
        return np.nan, np.nan, np.nan, np.nan, 0, "monomorphic"
    if cases.sum() == 0 or controls.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan, 0, "single_outcome"

    prevalence = np.clip(cases.sum() / totals.sum(), 1e-9, 1 - 1e-9)
    theta = np.array([np.log(prevalence / (1 - prevalence)), 0.0], dtype=float)
    genotype = np.arange(3, dtype=float)
    status = "max_iter"
    iterations = 0

    for iterations in range(1, options.max_iter + 1):
        eta = theta[0] + theta[1] * genotype
        probability = expit(eta)
        residual = cases - totals * probability
        weight = totals * probability * (1.0 - probability)
        information = np.array(
            [
                [weight.sum(), np.dot(weight, genotype)],
                [np.dot(weight, genotype), np.dot(weight, genotype * genotype)],
            ]
        )
        determinant = float(np.linalg.det(information))
        if not np.isfinite(determinant) or determinant <= options.information_tolerance:
            status = "singular_or_separated"
            break
        score = np.array([residual.sum(), np.dot(residual, genotype)])
        step = np.linalg.solve(information, score)

        old_ll = _grouped_log_likelihood(theta, totals, cases)
        scale = 1.0
        candidate = theta + step
        while scale > 2.0**-20 and _grouped_log_likelihood(candidate, totals, cases) < old_ll:
            scale *= 0.5
            candidate = theta + scale * step
        if scale <= 2.0**-20:
            status = "line_search_failed"
            break
        theta = candidate
        if np.max(np.abs(theta)) >= options.coefficient_limit:
            status = "separated"
            break
        if np.max(np.abs(scale * step)) < options.tolerance:
            status = "ok"
            break

    if status != "ok":
        # A finite-looking coefficient at separation or a failed Newton step is
        # not a valid maximum-likelihood estimate. Keep the status and exclude
        # the effect from calibration rather than silently using it.
        return np.nan, np.nan, np.nan, np.nan, iterations, status

    probability = expit(theta[0] + theta[1] * genotype)
    weight = totals * probability * (1.0 - probability)
    information = np.array(
        [
            [weight.sum(), np.dot(weight, genotype)],
            [np.dot(weight, genotype), np.dot(weight, genotype * genotype)],
        ]
    )
    covariance = np.linalg.inv(information)
    se = float(np.sqrt(covariance[1, 1]))
    z = float(theta[1] / se)
    p = float(2.0 * norm.sf(abs(z)))
    return float(theta[1]), se, z, p, iterations, status


def fit_binary_associations(
    samples: np.ndarray,
    labels: np.ndarray,
    options: LogisticFitOptions | None = None,
) -> pd.DataFrame:
    """Fit ``logit(P(y=1)) = intercept + beta * dosage`` for every SNP."""

    totals, cases = genotype_case_counts(samples, labels)
    return fit_binary_associations_from_counts(totals, cases, options)


def fit_binary_associations_from_counts(
    totals: np.ndarray,
    cases: np.ndarray,
    options: LogisticFitOptions | None = None,
) -> pd.DataFrame:
    """Fit all SNPs from pre-aggregated genotype and case counts."""

    options = options or LogisticFitOptions()
    totals = np.asarray(totals, dtype=float)
    cases = np.asarray(cases, dtype=float)
    if totals.ndim != 2 or totals.shape[1] != 3 or cases.shape != totals.shape:
        raise ValueError(f"Expected matching (n_snps, 3) counts, got {totals.shape} and {cases.shape}")
    if np.any(cases < 0) or np.any(totals < cases):
        raise ValueError("Case counts must lie between zero and total genotype counts")
    controls = totals - cases
    records = []
    for snp_index in range(totals.shape[0]):
        beta, se, z, p, iterations, status = _fit_grouped_logistic_snp(
            totals[snp_index], cases[snp_index], options
        )
        records.append(
            {
                "snp_index": snp_index,
                "beta": beta,
                "se": se,
                "z": z,
                "p_value": p,
                "odds_ratio": np.exp(beta) if np.isfinite(beta) else np.nan,
                "iterations": iterations,
                "fit_status": status,
                "n0": int(totals[snp_index, 0]),
                "n1": int(totals[snp_index, 1]),
                "n2": int(totals[snp_index, 2]),
                "case0": int(cases[snp_index, 0]),
                "case1": int(cases[snp_index, 1]),
                "case2": int(cases[snp_index, 2]),
                "af": float((totals[snp_index, 1] + 2.0 * totals[snp_index, 2]) / (2.0 * totals[snp_index].sum())),
                "case_af": float((cases[snp_index, 1] + 2.0 * cases[snp_index, 2]) / (2.0 * cases[snp_index].sum())),
                "control_af": float((controls[snp_index, 1] + 2.0 * controls[snp_index, 2]) / (2.0 * controls[snp_index].sum())),
            }
        )
    result = pd.DataFrame.from_records(records)
    result["maf"] = np.minimum(result.af, 1.0 - result.af)
    result["case_control_delta_af"] = result.case_af - result.control_af
    result["q_value"] = benjamini_hochberg(result["p_value"].to_numpy())
    return result


def compare_association_effects(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Compare fitted effects and reference-based discovery sets."""

    merged = reference[["snp_index", "beta", "se", "q_value"]].merge(
        candidate[["snp_index", "beta", "se", "q_value"]],
        on="snp_index",
        suffixes=("_real", "_synthetic"),
        validate="one_to_one",
    )
    valid = np.isfinite(merged.beta_real) & np.isfinite(merged.beta_synthetic)
    real_beta = merged.loc[valid, "beta_real"].to_numpy()
    syn_beta = merged.loc[valid, "beta_synthetic"].to_numpy()
    result: dict[str, float | int] = {"n_valid_effects": int(valid.sum())}
    if valid.sum() >= 2 and np.std(real_beta) > 0 and np.std(syn_beta) > 0:
        result["beta_pearson_r"] = float(pearsonr(real_beta, syn_beta).statistic)
        result["beta_spearman_rho"] = float(spearmanr(real_beta, syn_beta).statistic)
        result["beta_ols_slope"] = float(np.polyfit(real_beta, syn_beta, 1)[0])
        result["beta_ols_intercept"] = float(np.polyfit(real_beta, syn_beta, 1)[1])
    else:
        result.update(beta_pearson_r=np.nan, beta_spearman_rho=np.nan, beta_ols_slope=np.nan, beta_ols_intercept=np.nan)
    denominator = float(np.dot(real_beta, real_beta))
    result["beta_calibration_slope_origin"] = float(np.dot(real_beta, syn_beta) / denominator) if denominator > 0 else np.nan
    result["beta_mae"] = float(np.mean(np.abs(syn_beta - real_beta))) if valid.any() else np.nan
    result["beta_rmse"] = float(np.sqrt(np.mean((syn_beta - real_beta) ** 2))) if valid.any() else np.nan
    nonzero = valid & (merged.beta_real != 0) & (merged.beta_synthetic != 0)
    result["sign_concordance"] = float(np.mean(np.sign(merged.loc[nonzero, "beta_real"]) == np.sign(merged.loc[nonzero, "beta_synthetic"]))) if nonzero.any() else np.nan

    delta = reference[["snp_index", "case_control_delta_af"]].merge(
        candidate[["snp_index", "case_control_delta_af"]], on="snp_index", suffixes=("_real", "_synthetic")
    )
    real_delta = delta.case_control_delta_af_real.to_numpy(dtype=float)
    syn_delta = delta.case_control_delta_af_synthetic.to_numpy(dtype=float)
    if real_delta.size >= 2 and np.std(real_delta) > 0 and np.std(syn_delta) > 0:
        result["delta_af_pearson_r"] = float(pearsonr(real_delta, syn_delta).statistic)
    else:
        result["delta_af_pearson_r"] = np.nan
    delta_denominator = float(np.dot(real_delta, real_delta))
    result["delta_af_calibration_slope_origin"] = float(np.dot(real_delta, syn_delta) / delta_denominator) if delta_denominator else np.nan

    real_sig = np.isfinite(merged.q_value_real) & (merged.q_value_real <= alpha)
    syn_sig = np.isfinite(merged.q_value_synthetic) & (merged.q_value_synthetic <= alpha)
    same_direction = np.sign(merged.beta_real) == np.sign(merged.beta_synthetic)
    replicated = real_sig & syn_sig & same_direction
    discordant = syn_sig & ~replicated
    n_real, n_syn, n_rep = int(real_sig.sum()), int(syn_sig.sum()), int(replicated.sum())
    result.update(
        reference_discoveries=n_real,
        synthetic_discoveries=n_syn,
        same_direction_overlap=n_rep,
        power_vs_reference=float(n_rep / n_real) if n_real else np.nan,
        fdr_vs_reference=float(discordant.sum() / n_syn) if n_syn else np.nan,
    )
    return result
