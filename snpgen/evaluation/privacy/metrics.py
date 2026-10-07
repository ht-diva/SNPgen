"""
Privacy metrics for evaluating synthetic SNP data.

Implements the standard SNPgen privacy/fidelity metrics:
1. Identical Match Rate (IMR) — exact copy detection
2. Distance to Closest Record (DCR) — min distance distribution analysis
3. NNAA (Nearest Neighbor Adversarial Accuracy) — adapted from GeneDiffusion
4. Distance-based Membership Inference (MI) — re-identification risk
5. NNDR (Nearest Neighbor Distance Ratio) — copying detection
6. Allele Frequency Comparison (MAF Drift) — fidelity sanity check
7. Case-control allele-frequency calibration — conditional fidelity

The Yelmen-compatible AA_TS privacy loss and nearest-neighbour-chain analysis
live in ``yelmen.py`` and are orchestrated here for synthetic cohorts.

Plus PrivacyEvaluator class for orchestrating all metrics with incremental saving.
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score

from .distances import batched_knn, KNNCache
from .utils import (
    PrivacyDataBundle,
    save_privacy_results,
    load_privacy_results,
    save_privacy_manifest,
    load_privacy_manifest,
    subsample_by_label,
)
from .yelmen import (
    matched_privacy_cohorts,
    nearest_neighbor_chain_analysis,
    yelmen_privacy_loss,
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class IMRResult:
    """Result of Identical Match Rate analysis."""
    match_rate: float
    n_matches: int
    n_synthetic: int

    def to_summary_dict(self):
        return {
            'match_rate': self.match_rate,
            'n_matches': self.n_matches,
            'n_synthetic': self.n_synthetic,
        }


@dataclass
class DCRResult:
    """Result of Distance to Closest Record analysis."""
    dcr_syn: np.ndarray          # (N_syn,) min distances syn → real_train
    dcr_holdout: np.ndarray      # (N_holdout,) min distances holdout → real_train
    dcr_syn_median: float
    dcr_holdout_median: float
    dcr_syn_mean: float
    dcr_holdout_mean: float
    ks_statistic: float          # KS test statistic
    ks_pvalue: float             # KS test p-value
    mannwhitney_statistic: float
    mannwhitney_pvalue: float
    frac_below_5th_pct: float    # Fraction of syn DCR below 5th pct of holdout DCR
    metric: str                  # 'hamming' or 'manhattan'

    def to_summary_dict(self):
        return {
            'dcr_syn_median': float(self.dcr_syn_median),
            'dcr_holdout_median': float(self.dcr_holdout_median),
            'dcr_syn_mean': float(self.dcr_syn_mean),
            'dcr_holdout_mean': float(self.dcr_holdout_mean),
            'ks_statistic': float(self.ks_statistic),
            'ks_pvalue': float(self.ks_pvalue),
            'mannwhitney_pvalue': float(self.mannwhitney_pvalue),
            'frac_below_5th_pct': float(self.frac_below_5th_pct),
            'metric': self.metric,
        }


@dataclass
class NNAAResult:
    """Result of Nearest Neighbor Adversarial Accuracy analysis."""
    aa_train: float    # Fraction of train samples whose NN in syn is farther than NN in train
    aa_syn: float      # Fraction of syn samples whose NN in train is farther than NN in syn
    privacy_score: float   # (aa_train + aa_syn) / 2, target ~0.5
    n_train_samples: int
    n_syn_samples: int
    metric: str

    def to_summary_dict(self):
        return {
            'aa_train': float(self.aa_train),
            'aa_syn': float(self.aa_syn),
            'privacy_score': float(self.privacy_score),
            'n_train_samples': self.n_train_samples,
            'n_syn_samples': self.n_syn_samples,
            'metric': self.metric,
        }


@dataclass
class MIResult:
    """Result of distance-based Membership Inference analysis."""
    auc: float                        # ROC-AUC for distinguishing train vs holdout
    dcr_train_to_syn: np.ndarray      # (N_train,) min distances train → syn
    dcr_holdout_to_syn: np.ndarray    # (N_holdout,) min distances holdout → syn
    dcr_train_mean: float
    dcr_holdout_mean: float
    metric: str

    def to_summary_dict(self):
        return {
            'auc': float(self.auc),
            'dcr_train_mean': float(self.dcr_train_mean),
            'dcr_holdout_mean': float(self.dcr_holdout_mean),
            'metric': self.metric,
        }


@dataclass
class NNDRResult:
    """Result of Nearest Neighbor Distance Ratio analysis."""
    nndr_values: np.ndarray       # (N_syn,) d1/d2 for each synthetic sample
    nndr_mean: float
    nndr_median: float
    frac_below_08: float          # Fraction with NNDR < 0.8
    frac_below_05: float          # Fraction with NNDR < 0.5
    metric: str

    def to_summary_dict(self):
        return {
            'nndr_mean': float(self.nndr_mean),
            'nndr_median': float(self.nndr_median),
            'frac_below_0.8': float(self.frac_below_08),
            'frac_below_0.5': float(self.frac_below_05),
            'metric': self.metric,
        }


@dataclass
class MAFResult:
    """Result of Allele Frequency Comparison."""
    maf_real: np.ndarray     # (N_snps,) per-SNP allele frequency in real
    maf_syn: np.ndarray      # (N_snps,) per-SNP allele frequency in synthetic
    pearson_r: float
    pearson_pvalue: float
    slope: float
    intercept: float
    mean_abs_drift: float
    max_abs_drift: float

    def to_summary_dict(self):
        return {
            'pearson_r': float(self.pearson_r),
            'slope': float(getattr(self, 'slope', np.nan)),
            'intercept': float(getattr(self, 'intercept', np.nan)),
            'mean_abs_drift': float(self.mean_abs_drift),
            'max_abs_drift': float(self.max_abs_drift),
        }


@dataclass
class CaseControlDeltaResult:
    """Result of case-control allele-frequency shift comparison."""
    real_delta: np.ndarray
    syn_delta: np.ndarray
    pearson_r: float
    slope: float
    mean_abs_delta_error: float
    max_abs_delta_error: float

    def to_summary_dict(self):
        return {
            'pearson_r': float(self.pearson_r),
            'slope': float(self.slope),
            'mean_abs_delta_error': float(self.mean_abs_delta_error),
            'max_abs_delta_error': float(self.max_abs_delta_error),
        }


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def identical_match_rate(synthetic, real_train, verbose=True):
    """Compute fraction of synthetic samples that exactly match a real training sample.

    Uses hashing for O(N+M) complexity.

    Args:
        synthetic: np.ndarray (N_syn, D) int.
        real_train: np.ndarray (N_train, D) int.
        verbose: Print results.

    Returns:
        IMRResult
    """
    # Hash each real sample into a set
    real_hashes = set()
    for i in range(real_train.shape[0]):
        real_hashes.add(real_train[i].tobytes())

    n_matches = 0
    for i in range(synthetic.shape[0]):
        if synthetic[i].tobytes() in real_hashes:
            n_matches += 1

    rate = n_matches / synthetic.shape[0] if synthetic.shape[0] > 0 else 0.0

    if verbose:
        print(f"  IMR: {rate:.6f} ({n_matches}/{synthetic.shape[0]} exact matches)")

    return IMRResult(
        match_rate=rate,
        n_matches=n_matches,
        n_synthetic=synthetic.shape[0],
    )


def dcr_analysis(synthetic, real_train, real_holdout, metric='hamming',
                 device='auto', verbose=True, knn_fn=None, **knn_kwargs):
    """Distance to Closest Record analysis.

    Computes min distance from each synthetic sample to real_train, and
    compares against holdout → real_train as baseline.

    Args:
        synthetic: np.ndarray (N_syn, D).
        real_train: np.ndarray (N_train, D).
        real_holdout: np.ndarray (N_holdout, D).
        metric: 'hamming' or 'manhattan'.
        device: 'auto', 'gpu', or 'cpu'.
        verbose: Print results.

    Returns:
        DCRResult
    """
    _knn = knn_fn or batched_knn
    if verbose:
        print(f"  Computing DCR syn → real_train ({metric})...")
    dcr_syn_dists, _ = _knn(synthetic, real_train, k=1, metric=metric,
                            device=device, verbose=verbose, **knn_kwargs)
    dcr_syn = dcr_syn_dists[:, 0].astype(np.float64)

    if verbose:
        print(f"  Computing DCR holdout → real_train ({metric})...")
    dcr_hold_dists, _ = _knn(real_holdout, real_train, k=1, metric=metric,
                             device=device, verbose=verbose, **knn_kwargs)
    dcr_holdout = dcr_hold_dists[:, 0].astype(np.float64)

    # Statistical tests
    ks_stat, ks_p = scipy_stats.ks_2samp(dcr_syn, dcr_holdout)
    mw_stat, mw_p = scipy_stats.mannwhitneyu(dcr_syn, dcr_holdout, alternative='two-sided')

    # Fraction of syn DCR below 5th percentile of holdout DCR
    pct_5 = np.percentile(dcr_holdout, 5)
    frac_below = np.mean(dcr_syn < pct_5)

    result = DCRResult(
        dcr_syn=dcr_syn,
        dcr_holdout=dcr_holdout,
        dcr_syn_median=float(np.median(dcr_syn)),
        dcr_holdout_median=float(np.median(dcr_holdout)),
        dcr_syn_mean=float(np.mean(dcr_syn)),
        dcr_holdout_mean=float(np.mean(dcr_holdout)),
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_p),
        mannwhitney_statistic=float(mw_stat),
        mannwhitney_pvalue=float(mw_p),
        frac_below_5th_pct=float(frac_below),
        metric=metric,
    )

    if verbose:
        print(f"  DCR syn  median={result.dcr_syn_median:.1f}, mean={result.dcr_syn_mean:.1f}")
        print(f"  DCR hold median={result.dcr_holdout_median:.1f}, mean={result.dcr_holdout_mean:.1f}")
        print(f"  KS p-value={result.ks_pvalue:.4e}, frac_below_5th_pct={result.frac_below_5th_pct:.4f}")

    return result


def nnaa(synthetic, real_train, metric='hamming', n_samples=None,
         device='auto', seed=42, verbose=True, knn_fn=None, **knn_kwargs):
    """Nearest Neighbor Adversarial Accuracy (adapted from GeneDiffusion).

    For each real training sample, checks if its NN among synthetic samples
    is farther than its NN among other training samples (and vice versa).

    Args:
        synthetic: np.ndarray (N_syn, D).
        real_train: np.ndarray (N_train, D).
        metric: 'hamming' or 'manhattan'.
        n_samples: Number of samples to evaluate (None = all, but capped to avoid OOM).
        device: 'auto', 'gpu', or 'cpu'.
        seed: Random seed for subsampling.
        verbose: Print results.

    Returns:
        NNAAResult
    """
    rng = np.random.RandomState(seed)

    # Subsample if needed
    max_eval = n_samples or min(len(real_train), len(synthetic), 50000)
    if len(real_train) > max_eval:
        idx_train = rng.choice(len(real_train), max_eval, replace=False)
        train_sub = real_train[idx_train]
    else:
        train_sub = real_train
        max_eval = len(real_train)

    if len(synthetic) > max_eval:
        idx_syn = rng.choice(len(synthetic), max_eval, replace=False)
        syn_sub = synthetic[idx_syn]
    else:
        syn_sub = synthetic

    n_train_eval = len(train_sub)
    n_syn_eval = len(syn_sub)

    if verbose:
        print(f"  NNAA: evaluating {n_train_eval} train, {n_syn_eval} syn samples ({metric})")

    _knn = knn_fn or batched_knn

    # AA_train: for each train sample, compare NN distance in syn vs NN distance in train
    # We need k=1 NN in syn, and k=2 NN in train (since the sample itself is in train, we
    # skip self-match by using k=2 when querying within the same set)

    # train → syn (k=1)
    if verbose:
        print(f"  Computing train → syn kNN...")
    dist_train_to_syn, _ = _knn(train_sub, syn_sub, k=1, metric=metric,
                                device=device, verbose=verbose, **knn_kwargs)
    d_ts = dist_train_to_syn[:, 0].astype(np.float64)

    # train → train (k=2, skip self)
    if verbose:
        print(f"  Computing train → train kNN...")
    dist_train_to_train, _ = _knn(train_sub, train_sub, k=2, metric=metric,
                                  device=device, verbose=verbose, **knn_kwargs)
    # First NN might be self (distance 0), use second if so
    d_tt = np.where(
        dist_train_to_train[:, 0] == 0,
        dist_train_to_train[:, 1],
        dist_train_to_train[:, 0]
    ).astype(np.float64)

    # AA_train: fraction where syn is farther than train NN
    aa_train = float(np.mean(d_ts > d_tt))

    # AA_syn: for each syn sample, compare NN distance in train vs NN distance in syn
    if verbose:
        print(f"  Computing syn → train kNN...")
    dist_syn_to_train, _ = _knn(syn_sub, train_sub, k=1, metric=metric,
                                device=device, verbose=verbose, **knn_kwargs)
    d_st = dist_syn_to_train[:, 0].astype(np.float64)

    if verbose:
        print(f"  Computing syn → syn kNN...")
    dist_syn_to_syn, _ = _knn(syn_sub, syn_sub, k=2, metric=metric,
                              device=device, verbose=verbose, **knn_kwargs)
    d_ss = np.where(
        dist_syn_to_syn[:, 0] == 0,
        dist_syn_to_syn[:, 1],
        dist_syn_to_syn[:, 0]
    ).astype(np.float64)

    aa_syn = float(np.mean(d_st > d_ss))

    privacy_score = (aa_train + aa_syn) / 2.0

    if verbose:
        print(f"  NNAA: AA_train={aa_train:.4f}, AA_syn={aa_syn:.4f}, "
              f"privacy_score={privacy_score:.4f} (target ~0.5)")

    return NNAAResult(
        aa_train=aa_train,
        aa_syn=aa_syn,
        privacy_score=privacy_score,
        n_train_samples=n_train_eval,
        n_syn_samples=n_syn_eval,
        metric=metric,
    )


def membership_inference_distance(synthetic, real_train, real_holdout,
                                  metric='hamming', device='auto',
                                  verbose=True, knn_fn=None, **knn_kwargs):
    """Distance-based Membership Inference attack.

    Tests whether training samples are closer to synthetic data than holdout
    samples. If they are, the model has memorized training data.

    Args:
        synthetic: np.ndarray (N_syn, D).
        real_train: np.ndarray (N_train, D).
        real_holdout: np.ndarray (N_holdout, D).
        metric: 'hamming' or 'manhattan'.
        device: 'auto', 'gpu', or 'cpu'.
        verbose: Print results.

    Returns:
        MIResult
    """
    _knn = knn_fn or batched_knn
    if verbose:
        print(f"  MI: Computing train → syn distances ({metric})...")
    dist_train, _ = _knn(real_train, synthetic, k=1, metric=metric,
                         device=device, verbose=verbose, **knn_kwargs)
    dcr_train = dist_train[:, 0].astype(np.float64)

    if verbose:
        print(f"  MI: Computing holdout → syn distances ({metric})...")
    dist_holdout, _ = _knn(real_holdout, synthetic, k=1, metric=metric,
                           device=device, verbose=verbose, **knn_kwargs)
    dcr_holdout = dist_holdout[:, 0].astype(np.float64)

    # ROC-AUC: can we distinguish train (label=1) from holdout (label=0)?
    # Score: negative distance (closer = higher "membership" score)
    labels = np.concatenate([
        np.ones(len(dcr_train)),
        np.zeros(len(dcr_holdout))
    ])
    scores = np.concatenate([-dcr_train, -dcr_holdout])

    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = 0.5
        if verbose:
            print("  MI: Could not compute AUC (constant predictions)")

    if verbose:
        print(f"  MI AUC: {auc:.4f} (target ~0.5)")
        print(f"  MI: mean dist train→syn={np.mean(dcr_train):.1f}, "
              f"holdout→syn={np.mean(dcr_holdout):.1f}")

    return MIResult(
        auc=auc,
        dcr_train_to_syn=dcr_train,
        dcr_holdout_to_syn=dcr_holdout,
        dcr_train_mean=float(np.mean(dcr_train)),
        dcr_holdout_mean=float(np.mean(dcr_holdout)),
        metric=metric,
    )


def nndr_analysis(synthetic, real_train, metric='hamming', device='auto',
                  verbose=True, knn_fn=None, **knn_kwargs):
    """Nearest Neighbor Distance Ratio analysis.

    For each synthetic sample, computes ratio of 1st to 2nd nearest neighbor
    distance in real_train. Low ratio = sample suspiciously close to one
    specific real sample.

    Args:
        synthetic: np.ndarray (N_syn, D).
        real_train: np.ndarray (N_train, D).
        metric: 'hamming' or 'manhattan'.
        device: 'auto', 'gpu', or 'cpu'.
        verbose: Print results.

    Returns:
        NNDRResult
    """
    _knn = knn_fn or batched_knn
    if verbose:
        print(f"  NNDR: Computing syn → real_train kNN k=2 ({metric})...")
    dists, _ = _knn(synthetic, real_train, k=2, metric=metric,
                    device=device, verbose=verbose, **knn_kwargs)

    d1 = dists[:, 0].astype(np.float64)
    d2 = dists[:, 1].astype(np.float64)

    # Avoid division by zero
    valid = d2 > 0
    nndr = np.ones(len(d1), dtype=np.float64)
    nndr[valid] = d1[valid] / d2[valid]

    result = NNDRResult(
        nndr_values=nndr,
        nndr_mean=float(np.mean(nndr)),
        nndr_median=float(np.median(nndr)),
        frac_below_08=float(np.mean(nndr < 0.8)),
        frac_below_05=float(np.mean(nndr < 0.5)),
        metric=metric,
    )

    if verbose:
        print(f"  NNDR: mean={result.nndr_mean:.4f}, median={result.nndr_median:.4f}")
        print(f"  NNDR: frac<0.8={result.frac_below_08:.4f}, frac<0.5={result.frac_below_05:.4f}")

    return result


def allele_frequency_comparison(synthetic, real_train, verbose=True):
    """Per-SNP allele frequency comparison between real and synthetic data.

    Computes mean allele frequency (proportional to MAF) for each SNP.

    Args:
        synthetic: np.ndarray (N_syn, D) int, values in {0,1,2}.
        real_train: np.ndarray (N_train, D) int, values in {0,1,2}.
        verbose: Print results.

    Returns:
        MAFResult
    """
    # Mean allele count per SNP (proportional to allele frequency)
    maf_real = np.mean(real_train.astype(np.float64), axis=0) / 2.0
    maf_syn = np.mean(synthetic.astype(np.float64), axis=0) / 2.0

    drift = np.abs(maf_real - maf_syn)

    r, p = scipy_stats.pearsonr(maf_real, maf_syn)
    if np.allclose(maf_real, maf_real[0]):
        slope = np.nan
        intercept = np.nan
    else:
        slope, intercept = np.polyfit(maf_real, maf_syn, deg=1)

    result = MAFResult(
        maf_real=maf_real,
        maf_syn=maf_syn,
        pearson_r=float(r),
        pearson_pvalue=float(p),
        slope=float(slope),
        intercept=float(intercept),
        mean_abs_drift=float(np.mean(drift)),
        max_abs_drift=float(np.max(drift)),
    )

    if verbose:
        print(f"  MAF: Pearson r={result.pearson_r:.6f}, "
              f"slope={result.slope:.6f}, "
              f"mean_drift={result.mean_abs_drift:.6f}, max_drift={result.max_abs_drift:.6f}")

    return result


def case_control_delta_comparison(synthetic, labels_syn, real_train, labels_train, verbose=True):
    """Compare per-SNP case-control allele-frequency shifts.

    For each SNP, computes mean genotype/2 in cases minus controls for real and
    synthetic data. The correlation captures whether phenotype signal is placed
    on the same SNPs in the same direction; the through-origin slope captures
    whether effect magnitudes are calibrated.
    """
    labels_syn = np.asarray(labels_syn).reshape(-1)
    labels_train = np.asarray(labels_train).reshape(-1)

    real_case = labels_train == 1
    real_control = labels_train == 0
    syn_case = labels_syn == 1
    syn_control = labels_syn == 0

    if not (real_case.any() and real_control.any() and syn_case.any() and syn_control.any()):
        if verbose:
            print("  Case-control delta: skipped (requires labels 0 and 1 in real and synthetic data)")
        real_delta = np.full(real_train.shape[1], np.nan)
        syn_delta = np.full(synthetic.shape[1], np.nan)
        return CaseControlDeltaResult(
            real_delta=real_delta,
            syn_delta=syn_delta,
            pearson_r=np.nan,
            slope=np.nan,
            mean_abs_delta_error=np.nan,
            max_abs_delta_error=np.nan,
        )

    real_af = real_train.astype(np.float64) / 2.0
    syn_af = synthetic.astype(np.float64) / 2.0

    real_delta = real_af[real_case].mean(axis=0) - real_af[real_control].mean(axis=0)
    syn_delta = syn_af[syn_case].mean(axis=0) - syn_af[syn_control].mean(axis=0)
    delta_error = np.abs(syn_delta - real_delta)

    if np.allclose(real_delta, real_delta[0]) or np.allclose(syn_delta, syn_delta[0]):
        r = np.nan
    else:
        r, _p = scipy_stats.pearsonr(real_delta, syn_delta)

    denom = float(np.dot(real_delta, real_delta))
    slope = np.nan if denom == 0.0 else float(np.dot(real_delta, syn_delta) / denom)

    result = CaseControlDeltaResult(
        real_delta=real_delta,
        syn_delta=syn_delta,
        pearson_r=float(r),
        slope=float(slope),
        mean_abs_delta_error=float(np.mean(delta_error)),
        max_abs_delta_error=float(np.max(delta_error)),
    )

    if verbose:
        print(f"  Case-control delta: Pearson r={result.pearson_r:.6f}, "
              f"slope={result.slope:.6f}, "
              f"mean_abs_error={result.mean_abs_delta_error:.6f}")

    return result


# ---------------------------------------------------------------------------
# PrivacyEvaluator orchestrator
# ---------------------------------------------------------------------------

class PrivacyEvaluator:
    """Orchestrate privacy metrics with split-aware in-place result upgrades."""

    METRIC_NAMES = [
        'imr', 'nndr', 'nnaa', 'dcr', 'mi', 'yelmen_privacy_loss',
        'nearest_neighbor_chain', 'maf', 'case_control_delta',
    ]
    SYNTHETIC_PROTOCOL_VERSION = 'synthetic_privacy_v2_fit_train_yelmen_chains'
    RECONSTRUCTION_PROTOCOL_VERSION = 'reconstruction_privacy_v1_source_train_val'
    RECORD_METRIC_PREFIXES = (
        'imr__', 'nndr__', 'nnaa__', 'dcr__', 'mi__',
        'yelmen_privacy_loss__', 'nearest_neighbor_chain__',
    )

    def __init__(
        self,
        distance='hamming',
        per_class=True,
        device='auto',
        nnaa_n_samples=None,
        verbose=True,
        cache_knn=True,
        yelmen_distance='manhattan',
        matched_n_samples=50000,
        run_yelmen=True,
        yelmen_per_class=False,
        run_chains=True,
        chains_per_class=False,
        chain_min_length=2,
        chain_max_length=5,
        chain_neighbor_k=32,
        seed=42,
        force_recompute=False,
        **knn_kwargs,
    ):
        self.distance = distance
        self.per_class = per_class
        self.device = device
        self.nnaa_n_samples = nnaa_n_samples
        self.verbose = verbose
        self.cache_knn = cache_knn
        self.yelmen_distance = yelmen_distance
        self.matched_n_samples = matched_n_samples
        self.run_yelmen = run_yelmen
        self.yelmen_per_class = yelmen_per_class
        self.run_chains = run_chains
        self.chains_per_class = chains_per_class
        self.chain_min_length = chain_min_length
        self.chain_max_length = chain_max_length
        self.chain_neighbor_k = chain_neighbor_k
        self.seed = int(seed)
        self.force_recompute = force_recompute
        self.knn_kwargs = knn_kwargs

    def _synthetic_manifest(self, bundle):
        def file_identity(path):
            if not path:
                return None
            absolute = os.path.abspath(path)
            try:
                stat = os.stat(absolute)
            except OSError:
                return {'path': absolute, 'exists': False}
            return {
                'path': absolute,
                'exists': True,
                'size_bytes': int(stat.st_size),
                'mtime_ns': int(stat.st_mtime_ns),
            }

        return {
            'protocol_version': self.SYNTHETIC_PROTOCOL_VERSION,
            'evaluation_target': 'synthetic',
            'real_training_split': 'train',
            'real_validation_split': 'val',
            'real_holdout_split': 'test',
            'fidelity_reference_split': 'train_val',
            'dataset_path': bundle.dataset_path,
            'synthetic_path': bundle.synthetic_path,
            'dataset_file_identity': file_identity(bundle.dataset_path),
            'synthetic_file_identity': file_identity(bundle.synthetic_path),
            'split_seed': int(bundle.split_seed),
            'evaluation_seed': self.seed,
            'val_ratio': float(bundle.val_ratio),
            'test_ratio': float(bundle.test_ratio),
            'split_sizes': {
                'train': int(len(bundle.real_train)),
                'validation': int(len(bundle.real_validation)) if bundle.real_validation is not None else 0,
                'test': int(len(bundle.real_holdout)),
                'train_val': int(len(bundle.real_train_val)) if bundle.real_train_val is not None else 0,
                'synthetic': int(len(bundle.synthetic)),
            },
            'standard_distance': self.distance,
            'nnaa_n_samples': self.nnaa_n_samples,
            'yelmen': {
                'enabled': bool(self.run_yelmen),
                'distance': self.yelmen_distance,
                'matched_n_samples': self.matched_n_samples,
                'class_matching': True,
                'per_class': bool(self.yelmen_per_class),
                'definition': 'AA_TS_test_minus_AA_TS_train',
            },
            'nearest_neighbor_chains': {
                'enabled': bool(self.run_chains),
                'distance': 'hamming',
                'matched_n_samples': self.matched_n_samples,
                'per_class': bool(self.chains_per_class),
                'min_length': int(self.chain_min_length),
                'max_length': int(self.chain_max_length),
                'neighbor_k': int(self.chain_neighbor_k),
            },
            'preserved_legacy_metrics': ['maf__*', 'case_control_delta__*'],
        }

    @staticmethod
    def _same_manifest_field(previous, current, key):
        return previous is not None and previous.get(key) == current.get(key)

    def _prepare_synthetic_results(self, bundle, output_dir):
        """Drop stale record metrics while preserving train-val fidelity results."""
        results = load_privacy_results(output_dir)
        previous = load_privacy_manifest(output_dir)
        current = self._synthetic_manifest(bundle)
        core_fields = (
            'protocol_version', 'evaluation_target', 'dataset_path', 'synthetic_path',
            'dataset_file_identity', 'synthetic_file_identity',
            'split_seed', 'val_ratio', 'test_ratio', 'split_sizes',
            'standard_distance', 'nnaa_n_samples',
            'evaluation_seed',
        )
        core_matches = all(
            self._same_manifest_field(previous, current, key) for key in core_fields
        )
        removed = []
        if self.force_recompute or not core_matches:
            for key in list(results):
                if key.startswith(self.RECORD_METRIC_PREFIXES):
                    removed.append(key)
                    del results[key]
        else:
            if previous.get('yelmen') != current['yelmen']:
                for key in list(results):
                    if key.startswith('yelmen_privacy_loss__'):
                        removed.append(key)
                        del results[key]
            if previous.get('nearest_neighbor_chains') != current['nearest_neighbor_chains']:
                for key in list(results):
                    if key.startswith('nearest_neighbor_chain__'):
                        removed.append(key)
                        del results[key]

        if removed and self.verbose:
            print(
                'Protocol upgrade: removed stale split-dependent metrics in place: '
                + ', '.join(sorted(removed))
            )
        # Persist pruning immediately: a later failure cannot leave stale metrics.
        save_privacy_results(results, output_dir)
        save_privacy_manifest(current, output_dir)
        return results, current

    def evaluate(self, bundle, output_dir, eval_target='synthetic'):
        if eval_target == 'synthetic':
            target_data = bundle.synthetic
            target_labels = bundle.labels_syn
            privacy_train = bundle.real_train
            privacy_train_labels = bundle.labels_train
            fidelity_reference = (
                bundle.real_train_val if bundle.real_train_val is not None else bundle.real_train
            )
            fidelity_labels = (
                bundle.labels_train_val if bundle.labels_train_val is not None else bundle.labels_train
            )
            results, manifest = self._prepare_synthetic_results(bundle, output_dir)
            enable_yelmen = self.run_yelmen
            enable_chains = self.run_chains
        elif eval_target == 'reconstructed':
            if bundle.reconstructed is None:
                raise ValueError('No reconstructed data in bundle')
            target_data = bundle.reconstructed
            target_labels = bundle.labels_recon
            # Reconstruction targets derive from train+validation subjects.
            privacy_train = (
                bundle.real_train_val if bundle.real_train_val is not None else bundle.real_train
            )
            privacy_train_labels = (
                bundle.labels_train_val if bundle.labels_train_val is not None else bundle.labels_train
            )
            fidelity_reference = privacy_train
            fidelity_labels = privacy_train_labels
            results = load_privacy_results(output_dir)
            manifest = {
                'protocol_version': self.RECONSTRUCTION_PROTOCOL_VERSION,
                'evaluation_target': 'reconstructed',
                'source_reference_split': 'train_val',
                'real_holdout_split': 'test',
                'dataset_path': bundle.dataset_path,
                'split_seed': int(bundle.split_seed),
            }
            save_privacy_manifest(manifest, output_dir)
            enable_yelmen = False
            enable_chains = False
        else:
            raise ValueError(f'Unknown eval_target: {eval_target}')

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f' Privacy Evaluation: {bundle.model_name} ({eval_target})')
            print(f' Distance: {self.distance}')
            print(
                f' Target: {target_data.shape}, Train reference: {privacy_train.shape}, '
                f'Holdout: {bundle.real_holdout.shape}'
            )
            print(f"{'=' * 60}\n")

        self._run_metrics(
            results,
            output_dir,
            target_data,
            privacy_train,
            bundle.real_holdout,
            suffix='overall',
            target_labels=target_labels,
            real_train_labels=privacy_train_labels,
            real_holdout_labels=bundle.labels_holdout,
            fidelity_reference=fidelity_reference,
            fidelity_labels=fidelity_labels,
            enable_yelmen=enable_yelmen,
            enable_chains=enable_chains,
        )

        if self.per_class and target_labels is not None:
            for label_value in np.unique(privacy_train_labels):
                suffix = f'class_{int(label_value)}'
                train_sub = subsample_by_label(privacy_train, privacy_train_labels, label_value)
                holdout_sub = subsample_by_label(
                    bundle.real_holdout, bundle.labels_holdout, label_value
                )
                target_sub = subsample_by_label(target_data, target_labels, label_value)
                fidelity_sub = subsample_by_label(
                    fidelity_reference, fidelity_labels, label_value
                )
                if min(len(train_sub), len(holdout_sub), len(target_sub)) == 0:
                    if self.verbose:
                        print(f'\n  Skipping class {label_value}: insufficient samples')
                    continue
                if self.verbose:
                    print(f'\n--- Per-class: label={label_value} ---')
                    print(
                        f'  Train: {train_sub.shape}, Holdout: {holdout_sub.shape}, '
                        f'Target: {target_sub.shape}'
                    )
                self._run_metrics(
                    results,
                    output_dir,
                    target_sub,
                    train_sub,
                    holdout_sub,
                    suffix=suffix,
                    fidelity_reference=fidelity_sub,
                    enable_yelmen=enable_yelmen and self.yelmen_per_class,
                    enable_chains=enable_chains and self.chains_per_class,
                )

        manifest['completed_metric_keys'] = sorted(results)
        save_privacy_manifest(manifest, output_dir)
        return results

    def _run_metrics(
        self,
        results,
        output_dir,
        target,
        real_train,
        real_holdout,
        suffix,
        target_labels=None,
        real_train_labels=None,
        real_holdout_labels=None,
        fidelity_reference=None,
        fidelity_labels=None,
        enable_yelmen=False,
        enable_chains=False,
    ):
        cache = KNNCache(enabled=self.cache_knn)
        kw = dict(
            metric=self.distance,
            device=self.device,
            verbose=self.verbose,
            knn_fn=cache,
            **self.knn_kwargs,
        )
        fidelity_reference = real_train if fidelity_reference is None else fidelity_reference
        fidelity_labels = real_train_labels if fidelity_labels is None else fidelity_labels

        key = f'imr__{suffix}'
        if key not in results:
            if self.verbose:
                print(f'\n[{suffix}] Computing Identical Match Rate...')
            results[key] = identical_match_rate(target, real_train, verbose=self.verbose)
            save_privacy_results(results, output_dir)
        elif self.verbose:
            print(f'\n[{suffix}] Skipping IMR (already computed)')

        key = f'nndr__{suffix}'
        if key not in results:
            if self.verbose:
                print(f'\n[{suffix}] Computing NNDR...')
            results[key] = nndr_analysis(target, real_train, **kw)
            save_privacy_results(results, output_dir)
        elif self.verbose:
            print(f'\n[{suffix}] Skipping NNDR (already computed)')

        key = f'nnaa__{suffix}'
        if key not in results:
            if self.verbose:
                print(f'\n[{suffix}] Computing NNAA...')
            results[key] = nnaa(
                target, real_train, n_samples=self.nnaa_n_samples, seed=self.seed, **kw
            )
            save_privacy_results(results, output_dir)
        elif self.verbose:
            print(f'\n[{suffix}] Skipping NNAA (already computed)')

        key = f'dcr__{suffix}'
        if key not in results:
            if self.verbose:
                print(f'\n[{suffix}] Computing Distance to Closest Record...')
            results[key] = dcr_analysis(target, real_train, real_holdout, **kw)
            save_privacy_results(results, output_dir)
        elif self.verbose:
            print(f'\n[{suffix}] Skipping DCR (already computed)')

        key = f'mi__{suffix}'
        if key not in results:
            if self.verbose:
                print(f'\n[{suffix}] Computing Membership Inference...')
            results[key] = membership_inference_distance(
                target, real_train, real_holdout, **kw
            )
            save_privacy_results(results, output_dir)
        elif self.verbose:
            print(f'\n[{suffix}] Skipping MI (already computed)')

        matched = None
        if enable_yelmen or enable_chains:
            matched = matched_privacy_cohorts(
                real_train,
                real_holdout,
                target,
                labels_train=real_train_labels,
                labels_test=real_holdout_labels,
                labels_synthetic=target_labels,
                max_samples=self.matched_n_samples,
                seed=self.seed,
            )

        key = f'yelmen_privacy_loss__{suffix}'
        if enable_yelmen and key not in results:
            if self.verbose:
                print(f'\n[{suffix}] Computing Yelmen-compatible AA_TS privacy loss...')
            results[key] = yelmen_privacy_loss(
                matched,
                metric=self.yelmen_distance,
                device=self.device,
                verbose=self.verbose,
                **self.knn_kwargs,
            )
            save_privacy_results(results, output_dir)
        elif enable_yelmen and self.verbose:
            print(f'\n[{suffix}] Skipping Yelmen privacy loss (already computed)')

        key = f'nearest_neighbor_chain__{suffix}'
        if enable_chains and key not in results:
            if self.verbose:
                print(f'\n[{suffix}] Computing nearest-neighbour chains...')
            results[key] = nearest_neighbor_chain_analysis(
                matched.train,
                matched.synthetic,
                min_length=self.chain_min_length,
                max_length=self.chain_max_length,
                metric='hamming',
                neighbor_k=self.chain_neighbor_k,
                device=self.device,
                verbose=self.verbose,
                **self.knn_kwargs,
            )
            save_privacy_results(results, output_dir)
        elif enable_chains and self.verbose:
            print(f'\n[{suffix}] Skipping nearest-neighbour chains (already computed)')

        key = f'maf__{suffix}'
        if key not in results:
            if self.verbose:
                print(f'\n[{suffix}] Computing MAF Drift...')
            results[key] = allele_frequency_comparison(
                target, fidelity_reference, verbose=self.verbose
            )
            save_privacy_results(results, output_dir)
        elif self.verbose:
            print(f'\n[{suffix}] Skipping MAF (already computed)')

        key = f'case_control_delta__{suffix}'
        if target_labels is not None and fidelity_labels is not None:
            if key not in results:
                if self.verbose:
                    print(f'\n[{suffix}] Computing Case-Control Delta...')
                results[key] = case_control_delta_comparison(
                    target,
                    target_labels,
                    fidelity_reference,
                    fidelity_labels,
                    verbose=self.verbose,
                )
                save_privacy_results(results, output_dir)
            elif self.verbose:
                print(f'\n[{suffix}] Skipping Case-Control Delta (already computed)')

        if self.verbose and self.cache_knn:
            stats = cache.stats
            print(
                f"\n[{suffix}] kNN cache: {stats['hits']} hits, "
                f"{stats['misses']} misses, {stats['upgrades']} upgrades"
            )
