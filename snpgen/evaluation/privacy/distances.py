"""
GPU-accelerated and CPU-fallback batched k-NN distance computation for SNP data.

Supports:
- Hamming distance via one-hot matmul trick (GPU) or sklearn (CPU)
- Manhattan distance via cuML/torch (GPU) or sklearn (CPU)
- Automatic GPU/CPU selection and batch size calculation
"""

import warnings
import numpy as np
from tqdm.auto import tqdm

import torch

try:
    from cuml.neighbors import NearestNeighbors as cumlNearestNeighbors
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

from sklearn.neighbors import NearestNeighbors as sklearnNearestNeighbors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gpu_available():
    """Check if CUDA GPU is available."""
    return torch.cuda.is_available()


def _get_gpu_free_memory_gb(device=0):
    """Get free GPU memory in GB."""
    if not _gpu_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info(device)
    return free / (1024 ** 3)


def auto_batch_size(n_query, n_ref, n_features, gpu_mem_gb=None, metric='hamming'):
    """Estimate a safe batch size for batched kNN on GPU.

    Args:
        n_query: Number of query samples.
        n_ref: Number of reference samples.
        n_features: Number of SNP features.
        gpu_mem_gb: Available GPU memory in GB. If None, auto-detected.
        metric: 'hamming' or 'manhattan'.

    Returns:
        Batch size (int).
    """
    if gpu_mem_gb is None:
        gpu_mem_gb = _get_gpu_free_memory_gb()

    if gpu_mem_gb <= 0:
        # CPU mode: use larger batches since we use sklearn
        return min(n_query, 10000)

    # Reserve ~2 GB for overhead
    usable_gb = max(gpu_mem_gb - 2.0, 1.0)

    if metric == 'hamming':
        # Ref one-hot on GPU: n_ref * n_features * 3 * 4 bytes (float32)
        ref_mem_gb = n_ref * n_features * 3 * 4 / (1024 ** 3)
        remaining_gb = usable_gb - ref_mem_gb
        if remaining_gb <= 0.5:
            return min(n_query, 500)
        # Per-sample cost:
        #   - Distance matrix row: n_ref * 4 bytes (float32)
        #   - Query one-hot row:   n_features * 3 * 4 bytes (float32)
        #   - Encoding temporaries: ~n_features * 16 bytes (int64 data + indices)
        bytes_per_sample = n_ref * 4 + n_features * 3 * 4 + n_features * 16
        # 0.8 safety factor for PyTorch allocator overhead and fragmentation
        batch_size = int(remaining_gb * 0.8 * (1024 ** 3) / bytes_per_sample)
    else:
        # Manhattan: ref data on GPU as float32: n_ref * n_features * 4
        ref_mem_gb = n_ref * n_features * 4 / (1024 ** 3)
        remaining_gb = usable_gb - ref_mem_gb
        if remaining_gb <= 0.5:
            return min(n_query, 500)
        bytes_per_sample = n_ref * 4 + n_features * 4
        batch_size = int(remaining_gb * (1024 ** 3) / bytes_per_sample)

    return max(min(batch_size, n_query), 1)


# ---------------------------------------------------------------------------
# One-hot encoding for Hamming distance
# ---------------------------------------------------------------------------

def onehot_encode_snps(data, dtype=torch.float32):
    """One-hot encode SNP data {0,1,2} → (N, 3*n_features) for matmul trick.

    For each SNP position, maps value v to a 3-element one-hot vector at
    columns [3*j, 3*j+1, 3*j+2] where j is the feature index.
    Agreement between two one-hot vectors = 1 if same allele, 0 otherwise.
    So Hamming(x, y) = n_features - dot(onehot(x), onehot(y)).

    Args:
        data: np.ndarray of shape (N, n_features) with values in {0, 1, 2}.
        dtype: Torch dtype for the output (default float32 for precision).

    Returns:
        torch.Tensor of shape (N, 3 * n_features).
    """
    data_t = torch.from_numpy(np.asarray(data, dtype=np.int64))
    n_samples, n_features = data_t.shape
    onehot = torch.zeros(n_samples, n_features * 3, dtype=dtype)
    # indices[i, j] = j * 3 + data[i, j]  → column to set to 1
    indices = torch.arange(n_features).unsqueeze(0) * 3 + data_t  # (N, n_features)
    onehot.scatter_(1, indices, torch.ones_like(indices, dtype=dtype))
    return onehot


# ---------------------------------------------------------------------------
# GPU implementations
# ---------------------------------------------------------------------------

def _knn_gpu_hamming(query, ref, k=1, batch_size=None, verbose=True):
    """GPU-accelerated k-NN using Hamming distance (one-hot matmul trick).

    Args:
        query: np.ndarray (N_q, D) int, values in {0,1,2}.
        ref: np.ndarray (N_r, D) int, values in {0,1,2}.
        k: Number of nearest neighbors.
        batch_size: Batch size for queries. Auto-computed if None.
        verbose: Show progress bar.

    Returns:
        distances: np.ndarray (N_q, k) Hamming distances (int counts of mismatches).
        indices: np.ndarray (N_q, k) indices into ref.
    """
    n_query, n_features = query.shape
    n_ref = ref.shape[0]

    if batch_size is None:
        batch_size = auto_batch_size(n_query, n_ref, n_features, metric='hamming')
        if verbose:
            print(f"Auto batch size for Hamming kNN on GPU: {batch_size}")

    device = torch.device('cuda')

    # One-hot encode reference and move to GPU
    ref_onehot = onehot_encode_snps(ref, dtype=torch.float32).to(device)

    all_distances = np.empty((n_query, k), dtype=np.int32)
    all_indices = np.empty((n_query, k), dtype=np.int64)

    n_batches = (n_query + batch_size - 1) // batch_size
    iterator = range(0, n_query, batch_size)
    if verbose:
        iterator = tqdm(iterator, total=n_batches, desc="kNN Hamming (GPU)")

    for start in iterator:
        end = min(start + batch_size, n_query)
        q_batch = query[start:end]

        # One-hot encode query batch
        q_onehot = onehot_encode_snps(q_batch, dtype=torch.float32).to(device)

        # Agreement = q_onehot @ ref_onehot.T → (batch, n_ref)
        hamming = torch.mm(q_onehot, ref_onehot.t())
        del q_onehot  # free before in-place conversion

        # Hamming = n_features - agreement  (in-place to avoid double allocation)
        hamming.neg_().add_(n_features)

        # Get top-k smallest
        topk_dists, topk_idxs = torch.topk(hamming, k, dim=1, largest=False)

        all_distances[start:end] = topk_dists.cpu().to(torch.int32).numpy()
        all_indices[start:end] = topk_idxs.cpu().numpy()

        del hamming, topk_dists, topk_idxs

    del ref_onehot
    torch.cuda.empty_cache()

    return all_distances, all_indices


def _knn_gpu_manhattan(query, ref, k=1, batch_size=None, verbose=True):
    """GPU-accelerated k-NN using Manhattan distance.

    Uses cuML if available, otherwise batched torch.cdist.

    Args:
        query: np.ndarray (N_q, D) int/float.
        ref: np.ndarray (N_r, D) int/float.
        k: Number of nearest neighbors.
        batch_size: Batch size for queries (used for torch fallback).
        verbose: Show progress bar.

    Returns:
        distances: np.ndarray (N_q, k) Manhattan distances.
        indices: np.ndarray (N_q, k) indices into ref.
    """
    if CUML_AVAILABLE:
        return _knn_cuml_manhattan(query, ref, k, verbose)
    else:
        return _knn_torch_manhattan(query, ref, k, batch_size, verbose)


def _knn_cuml_manhattan(query, ref, k=1, verbose=True):
    """k-NN using cuML NearestNeighbors with Manhattan distance."""
    import cupy as cp

    ref_f32 = ref.astype(np.float32)
    query_f32 = query.astype(np.float32)

    nn = cumlNearestNeighbors(n_neighbors=k, metric='manhattan', algorithm='brute')
    nn.fit(ref_f32)
    distances, indices = nn.kneighbors(query_f32)

    # cuML may return cupy arrays
    if hasattr(distances, 'get'):
        distances = distances.get()
    if hasattr(indices, 'get'):
        indices = indices.get()

    return distances.astype(np.float32), indices.astype(np.int64)


def _knn_torch_manhattan(query, ref, k=1, batch_size=None, verbose=True):
    """Batched k-NN using torch.cdist with Manhattan distance."""
    n_query, n_features = query.shape
    n_ref = ref.shape[0]

    if batch_size is None:
        batch_size = auto_batch_size(n_query, n_ref, n_features, metric='manhattan')
        if verbose:
            print(f"Auto batch size for Manhattan kNN on GPU: {batch_size}")

    device = torch.device('cuda')
    ref_t = torch.from_numpy(ref.astype(np.float32)).to(device)

    all_distances = np.empty((n_query, k), dtype=np.float32)
    all_indices = np.empty((n_query, k), dtype=np.int64)

    n_batches = (n_query + batch_size - 1) // batch_size
    iterator = range(0, n_query, batch_size)
    if verbose:
        iterator = tqdm(iterator, total=n_batches, desc="kNN Manhattan (GPU)")

    for start in iterator:
        end = min(start + batch_size, n_query)
        q_batch = torch.from_numpy(query[start:end].astype(np.float32)).to(device)

        dists = torch.cdist(q_batch, ref_t, p=1)  # (batch, n_ref)
        topk_dists, topk_idxs = torch.topk(dists, k, dim=1, largest=False)

        all_distances[start:end] = topk_dists.cpu().numpy()
        all_indices[start:end] = topk_idxs.cpu().numpy()

        del q_batch, dists, topk_dists, topk_idxs

    del ref_t
    torch.cuda.empty_cache()

    return all_distances, all_indices


# ---------------------------------------------------------------------------
# CPU implementations
# ---------------------------------------------------------------------------

def _knn_cpu(query, ref, k=1, metric='hamming', verbose=True, max_samples=None):
    """CPU-based k-NN using sklearn NearestNeighbors.

    Args:
        query: np.ndarray (N_q, D).
        ref: np.ndarray (N_r, D).
        k: Number of nearest neighbors.
        metric: 'hamming' or 'manhattan'.
        verbose: Print info messages.
        max_samples: If set and query/ref exceed this, subsample with a warning.

    Returns:
        distances: np.ndarray (N_q, k).
        indices: np.ndarray (N_q, k).
    """
    n_query = query.shape[0]
    n_ref = ref.shape[0]

    if max_samples is not None and (n_query > max_samples or n_ref > max_samples):
        warnings.warn(
            f"CPU mode: dataset too large ({n_query}×{n_ref}). "
            f"Subsampling to {max_samples} for feasibility."
        )
        if n_ref > max_samples:
            idx = np.random.choice(n_ref, max_samples, replace=False)
            ref = ref[idx]
        if n_query > max_samples:
            idx = np.random.choice(n_query, max_samples, replace=False)
            query = query[idx]

    if verbose:
        print(f"kNN CPU ({metric}): query={query.shape}, ref={ref.shape}, k={k}")

    if metric == 'hamming':
        # sklearn hamming is normalized (0-1), we want integer counts
        sklearn_metric = 'hamming'
    else:
        sklearn_metric = 'manhattan'

    ref_f = ref.astype(np.float32) if metric == 'manhattan' else ref
    query_f = query.astype(np.float32) if metric == 'manhattan' else query

    nn = sklearnNearestNeighbors(n_neighbors=k, metric=sklearn_metric, algorithm='auto', n_jobs=-1)
    nn.fit(ref_f)
    distances, indices = nn.kneighbors(query_f)

    if metric == 'hamming':
        # sklearn returns normalized hamming (fraction), convert to int counts
        n_features = query.shape[1]
        distances = np.rint(distances * n_features).astype(np.int32)

    return distances.astype(np.float32 if metric == 'manhattan' else np.int32), indices.astype(np.int64)


# ---------------------------------------------------------------------------
# kNN cache
# ---------------------------------------------------------------------------

class KNNCache:
    """Drop-in replacement for ``batched_knn`` with result caching.

    Caches kNN results keyed by ``(id(query), id(ref), metric)``.
    If a cached result has ``k_cached >= k_requested``, returns sliced
    results without recomputation.  Otherwise recomputes with the
    requested *k* and updates the cache entry (k-upgrade).

    Usage::

        cache = KNNCache(enabled=True)
        dists, idxs = cache(query, ref, k=2, metric='hamming', device='auto')
        # later, same pair with k=1 → served from cache
        dists1, idxs1 = cache(query, ref, k=1, metric='hamming', device='auto')

    Pass ``enabled=False`` to make every call fall through to
    ``batched_knn`` (zero overhead).
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._cache = {}          # key → (dists, idxs)
        self.stats = {'hits': 0, 'misses': 0, 'upgrades': 0}

    # same signature as batched_knn
    def __call__(self, query, ref, k=1, metric='hamming', **kwargs):
        if not self.enabled:
            return batched_knn(query, ref, k=k, metric=metric, **kwargs)

        key = (id(query), id(ref), metric)
        verbose = kwargs.get('verbose', True)

        if key in self._cache:
            dists, idxs = self._cache[key]
            if dists.shape[1] >= k:
                self.stats['hits'] += 1
                if verbose:
                    print(f"  [KNNCache] HIT: k={k} (cached k={dists.shape[1]})")
                return dists[:, :k].copy(), idxs[:, :k].copy()
            # need more neighbours than cached → recompute
            self.stats['upgrades'] += 1
            if verbose:
                print(f"  [KNNCache] k-UPGRADE: {dists.shape[1]} → {k}")

        self.stats['misses'] += 1
        dists, idxs = batched_knn(query, ref, k=k, metric=metric, **kwargs)
        self._cache[key] = (dists.copy(), idxs.copy())
        return dists, idxs

    def clear(self):
        """Drop all cached results and reset statistics."""
        self._cache.clear()
        self.stats = {'hits': 0, 'misses': 0, 'upgrades': 0}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def batched_knn(query, ref, k=1, metric='hamming', device='auto',
                batch_size=None, verbose=True, cpu_max_samples=50000):
    """Compute k-NN distances from query to ref with automatic GPU/CPU selection.

    Args:
        query: np.ndarray (N_q, D) with SNP values in {0, 1, 2}.
        ref: np.ndarray (N_r, D) with SNP values in {0, 1, 2}.
        k: Number of nearest neighbors.
        metric: 'hamming' or 'manhattan'.
        device: 'auto' (GPU if available, else CPU), 'gpu', or 'cpu'.
        batch_size: Override batch size (auto-computed if None).
        verbose: Show progress bar / info.
        cpu_max_samples: Max samples per set on CPU before subsampling.

    Returns:
        distances: np.ndarray (N_q, k).
        indices: np.ndarray (N_q, k) indices into ref.
    """
    assert metric in ('hamming', 'manhattan'), f"Unknown metric: {metric}"
    assert query.shape[1] == ref.shape[1], "query and ref must have same number of features"

    use_gpu = (device == 'gpu') or (device == 'auto' and _gpu_available())

    if use_gpu and not _gpu_available():
        warnings.warn("GPU requested but CUDA not available. Falling back to CPU.")
        use_gpu = False

    if use_gpu:
        if metric == 'hamming':
            return _knn_gpu_hamming(query, ref, k, batch_size, verbose)
        else:
            return _knn_gpu_manhattan(query, ref, k, batch_size, verbose)
    else:
        return _knn_cpu(query, ref, k, metric, verbose, max_samples=cpu_max_samples)
