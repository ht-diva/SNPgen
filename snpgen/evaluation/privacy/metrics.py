"""
Privacy metrics for evaluating synthetic SNP data.

Implements 6 metrics:
1. Identical Match Rate (IMR) — exact copy detection
2. Distance to Closest Record (DCR) — min distance distribution analysis
3. NNAA (Nearest Neighbor Adversarial Accuracy) — adapted from GeneDiffusion
4. Distance-based Membership Inference (MI) — re-identification risk
5. NNDR (Nearest Neighbor Distance Ratio) — copying detection
6. Allele Frequency Comparison (MAF Drift) — fidelity sanity check

Plus PrivacyEvaluator class for orchestrating all metrics with incremental saving.
"""

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
    subsample_by_label,
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
    mean_abs_drift: float
    max_abs_drift: float

    def to_summary_dict(self):
        return {
            'pearson_r': float(self.pearson_r),
            'mean_abs_drift': float(self.mean_abs_drift),
            'max_abs_drift': float(self.max_abs_drift),
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

    result = MAFResult(
        maf_real=maf_real,
        maf_syn=maf_syn,
        pearson_r=float(r),
        pearson_pvalue=float(p),
        mean_abs_drift=float(np.mean(drift)),
        max_abs_drift=float(np.max(drift)),
    )

    if verbose:
        print(f"  MAF: Pearson r={result.pearson_r:.6f}, "
              f"mean_drift={result.mean_abs_drift:.6f}, max_drift={result.max_abs_drift:.6f}")

    return result


# ---------------------------------------------------------------------------
# PrivacyEvaluator orchestrator
# ---------------------------------------------------------------------------

class PrivacyEvaluator:
    """Orchestrates all privacy metrics with incremental saving.

    Usage:
        evaluator = PrivacyEvaluator(distance='hamming', per_class=True)
        results = evaluator.evaluate(bundle, output_dir, eval_target='synthetic')
    """

    # Ordered list of metrics to run (higher-k metrics first for cache efficiency)
    METRIC_NAMES = ['imr', 'nndr', 'nnaa', 'dcr', 'mi', 'maf']

    def __init__(self, distance='hamming', per_class=True, device='auto',
                 nnaa_n_samples=None, verbose=True, cache_knn=True, **knn_kwargs):
        """
        Args:
            distance: 'hamming' or 'manhattan'.
            per_class: If True, also compute metrics per label class.
            device: 'auto', 'gpu', or 'cpu'.
            nnaa_n_samples: Number of samples for NNAA (None=auto).
            verbose: Print progress.
            cache_knn: If True, cache kNN results and reuse across metrics.
            **knn_kwargs: Extra args passed to batched_knn.
        """
        self.distance = distance
        self.per_class = per_class
        self.device = device
        self.nnaa_n_samples = nnaa_n_samples
        self.verbose = verbose
        self.cache_knn = cache_knn
        self.knn_kwargs = knn_kwargs

    def evaluate(self, bundle, output_dir, eval_target='synthetic'):
        """Run all privacy metrics with incremental saving.

        Args:
            bundle: PrivacyDataBundle.
            output_dir: Directory to save results.
            eval_target: 'synthetic' or 'reconstructed'.

        Returns:
            Dict mapping metric keys to result dataclass instances.
        """
        results = load_privacy_results(output_dir)

        # Select the target data
        if eval_target == 'synthetic':
            target_data = bundle.synthetic
            target_labels = bundle.labels_syn
        elif eval_target == 'reconstructed':
            if bundle.reconstructed is None:
                raise ValueError("No reconstructed data in bundle")
            target_data = bundle.reconstructed
            target_labels = bundle.labels_recon
        else:
            raise ValueError(f"Unknown eval_target: {eval_target}")

        if self.verbose:
            print(f"\n{'='*60}")
            print(f" Privacy Evaluation: {bundle.model_name} ({eval_target})")
            print(f" Distance: {self.distance}")
            print(f" Target: {target_data.shape}, Train: {bundle.real_train.shape}, "
                  f"Holdout: {bundle.real_holdout.shape}")
            print(f"{'='*60}\n")

        # Run overall metrics
        self._run_metrics(
            results, output_dir,
            target_data, bundle.real_train, bundle.real_holdout,
            suffix='overall'
        )

        # Run per-class metrics
        if self.per_class and target_labels is not None:
            unique_labels = np.unique(bundle.labels_train)
            for label_val in unique_labels:
                suffix = f'class_{int(label_val)}'
                train_sub = subsample_by_label(bundle.real_train, bundle.labels_train, label_val)
                holdout_sub = subsample_by_label(bundle.real_holdout, bundle.labels_holdout, label_val)
                target_sub = subsample_by_label(target_data, target_labels, label_val)

                if len(train_sub) == 0 or len(holdout_sub) == 0 or len(target_sub) == 0:
                    if self.verbose:
                        print(f"\n  Skipping class {label_val}: insufficient samples")
                    continue

                if self.verbose:
                    print(f"\n--- Per-class: label={label_val} ---")
                    print(f"  Train: {train_sub.shape}, Holdout: {holdout_sub.shape}, "
                          f"Target: {target_sub.shape}")

                self._run_metrics(
                    results, output_dir,
                    target_sub, train_sub, holdout_sub,
                    suffix=suffix
                )

        return results

    def _run_metrics(self, results, output_dir, target, real_train, real_holdout, suffix):
        """Run all metrics for a given data subset."""
        cache = KNNCache(enabled=self.cache_knn)
        kw = dict(metric=self.distance, device=self.device, verbose=self.verbose,
                  knn_fn=cache, **self.knn_kwargs)

        # 1. IMR (no kNN)
        key = f'imr__{suffix}'
        if key not in results:
            if self.verbose:
                print(f"\n[{suffix}] Computing Identical Match Rate...")
            results[key] = identical_match_rate(target, real_train, verbose=self.verbose)
            save_privacy_results(results, output_dir)
        elif self.verbose:
            print(f"\n[{suffix}] Skipping IMR (already computed)")

        # 2. NNDR (k=2 — populates cache for later k=1 metrics)
        key = f'nndr__{suffix}'
        if key not in results:
            if self.verbose:
                print(f"\n[{suffix}] Computing NNDR...")
            results[key] = nndr_analysis(target, real_train, **kw)
            save_privacy_results(results, output_dir)
        elif self.verbose:
            print(f"\n[{suffix}] Skipping NNDR (already computed)")

        # 3. NNAA (k=1 and k=2)
        key = f'nnaa__{suffix}'
        if key not in results:
            if self.verbose:
                print(f"\n[{suffix}] Computing NNAA...")
            results[key] = nnaa(target, real_train, n_samples=self.nnaa_n_samples, **kw)
            save_privacy_results(results, output_dir)
        elif self.verbose:
            print(f"\n[{suffix}] Skipping NNAA (already computed)")

        # 4. DCR (k=1 — may hit cache from NNDR's k=2)
        key = f'dcr__{suffix}'
        if key not in results:
            if self.verbose:
                print(f"\n[{suffix}] Computing Distance to Closest Record...")
            results[key] = dcr_analysis(target, real_train, real_holdout, **kw)
            save_privacy_results(results, output_dir)
        elif self.verbose:
            print(f"\n[{suffix}] Skipping DCR (already computed)")

        # 5. MI (k=1)
        key = f'mi__{suffix}'
        if key not in results:
            if self.verbose:
                print(f"\n[{suffix}] Computing Membership Inference...")
            results[key] = membership_inference_distance(target, real_train, real_holdout, **kw)
            save_privacy_results(results, output_dir)
        elif self.verbose:
            print(f"\n[{suffix}] Skipping MI (already computed)")

        # 6. MAF (no kNN)
        key = f'maf__{suffix}'
        if key not in results:
            if self.verbose:
                print(f"\n[{suffix}] Computing MAF Drift...")
            results[key] = allele_frequency_comparison(target, real_train, verbose=self.verbose)
            save_privacy_results(results, output_dir)
        elif self.verbose:
            print(f"\n[{suffix}] Skipping MAF (already computed)")

        if self.verbose and self.cache_knn:
            s = cache.stats
            print(f"\n[{suffix}] kNN cache: {s['hits']} hits, "
                  f"{s['misses']} misses, {s['upgrades']} upgrades")
