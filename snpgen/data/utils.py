import torch
import numpy as np

import numba

def compute_bce_pos_weight(binary_labels):
    binary_labels = torch.from_numpy(binary_labels)
    num_positives = torch.sum(binary_labels, dim=0)
    num_negatives = len(binary_labels) - num_positives
    pos_weight  = num_negatives / num_positives
    return pos_weight


def create_pad_mask(block_ids, patch_size):
    pad_mask = []
    padded_block_ids = []
    
    for block_id in np.unique(block_ids):
        block_mask = block_ids == block_id
        block_length = np.sum(block_mask)
        padding_needed = (patch_size - (block_length % patch_size)) % patch_size
        
        if padding_needed > 0:
            block_pad_mask = np.append(np.zeros(block_length, dtype=bool), np.ones(padding_needed, dtype=bool))
        else:
            block_pad_mask = np.zeros(block_length, dtype=bool)
            
        padded_block_ids.extend([block_id] * len(block_pad_mask))
        pad_mask.extend(block_pad_mask)
    pad_mask = np.stack(pad_mask)#.astype(bool)
    padded_block_ids = np.array(padded_block_ids)

    return pad_mask, padded_block_ids


@numba.jit(nopython=True, parallel=True, cache=True)
def _fast_assign_2d_scalar(x, y, idx):
    batch_size = x.shape[0]
    
    for i in numba.prange(batch_size):
        x[i, idx] = y
    return x

@numba.jit(nopython=True, parallel=True, cache=True)
def _fast_assign_2d_non_scalar(x, y, idx):
    batch_size = x.shape[0]
    
    for i in numba.prange(batch_size):
        x[i, idx] = y[i]
    return x

@numba.jit(nopython=True, parallel=True, cache=True)
def _fast_assign_3d_scalar(x, y, idx, third_axis_idx=None):
    batch_size = x.shape[0]
    
    if third_axis_idx is None:
        for i in numba.prange(batch_size):
            x[i, idx, :] = y
    else:
        for i in numba.prange(batch_size):
            # We need to loop over the third axis indices to assign values because right
            # now using more than one non-scalar array index is unsupported in Numba.
            # (idx is a non-scalar, and third_axis_idx is also a non-scalar)
            
            # Note: This is not thread-safe.
            # Multiple threads could attempt to write to overlapping regions of x if:
            # - idx contains duplicate values
            # - third_axis_idx contains duplicate values
            for j in third_axis_idx:
                x[i, idx, j] = y
    return x

@numba.jit(nopython=True, parallel=True, cache=True)
def _fast_assign_3d_non_scalar(x, y, idx, third_axis_idx=None):
    batch_size = x.shape[0]
    
    if third_axis_idx is None:
        for i in numba.prange(batch_size):
            x[i, idx, :] = y[i]
    else:
        for i in numba.prange(batch_size):
            # Same considerations as in _fast_assign_3d_scalar()
            for j in third_axis_idx:
                x[i, idx, j] = y[i, :, j]
                
    return x

def fast_assign(x, y, idx, third_axis_idx=None):
    """
    Assigns values from array `y` to array `x` at specified indices `idx`.
    
    Parameters:
    x (numpy.ndarray): The target array to which values will be assigned. Can be 2D or 3D.
    y (int or numpy.ndarray): The integer value or array of values to assign to `x`.
    idx (int or array-like): The indices along the second axis where values from `y` will be assigned.
    third_axis_idx (int or list-like, optional): The indces along the third axis for 3D arrays. If None, values are assigned across all indices of the third axis.
    
    Returns:
    numpy.ndarray: The modified array `x` with values from `y` assigned at the specified indices.
    """
    # Convert boolean mask to indices if necessary
    if isinstance(idx, np.ndarray) and idx.dtype == np.bool_:
        idx = np.where(idx)[0]
    
    if x.ndim == 2:
        if np.isscalar(y) or y.ndim == 0:
            return _fast_assign_2d_scalar(x, y, idx)
        else:
            return _fast_assign_2d_non_scalar(x, y, idx)
        
    elif x.ndim == 3:
        # Convert third_axis_idx to array if necessary
        if third_axis_idx is not None and \
            (np.isscalar(third_axis_idx) or (isinstance(third_axis_idx, np.ndarray) and third_axis_idx.ndim == 0)):
            third_axis_idx = np.array([third_axis_idx])
        
        if np.isscalar(y) or y.ndim == 0:
            return _fast_assign_3d_scalar(x, y, idx, third_axis_idx)
        else:
            return _fast_assign_3d_non_scalar(x, y, idx, third_axis_idx)
    else:
        raise ValueError("Input array must be 2D or 3D.")

# @numba.jit(nopython=True, parallel=True, cache=True)
# def fast_assign(x, y, idx, third_axis_idx=None):
#     """
#     Assigns values from array `y` to array `x` at specified indices `idx`.
    
#     Parameters:
#     x (numpy.ndarray): The target array to which values will be assigned. Can be 2D or 3D.
#     y (int or numpy.ndarray): The integer value or array of values to assign to `x`.
#     idx (int or array-like): The indices along the second axis where values from `y` will be assigned.
#     third_axis_idx (int or list-like, optional): The indces along the third axis for 3D arrays. If None, values are assigned across all indices of the third axis.
    
#     Returns:
#     numpy.ndarray: The modified array `x` with values from `y` assigned at the specified indices.
#     """
    
#     if x.ndim == 2:
#         x[:, idx] = y
        
#     elif x.ndim == 3:
#         if third_axis_idx is None:
#             x[:, idx, :] = y
#         else:
#             if isinstance(third_axis_idx, int):
#                 x[:, idx, third_axis_idx] = y
#             else:
#                 # We need to loop over the third axis indices to assign values because right
#                 # now using more than one non-scalar array index is unsupported in Numba.
#                 # (idx is a non-scalar, and third_axis_idx is also a non-scalar)
                
#                 # Note: This is not thread-safe.
#                 # Multiple threads could attempt to write to overlapping regions of x if:
#                 # - idx contains duplicate values
#                 # - third_axis_idx contains duplicate values
#                 for i in numba.prange(len(third_axis_idx)):
#                     x[:, idx, third_axis_idx[i]] = y[:, :, third_axis_idx[i]]
                
#     else:
#         raise ValueError("Input array must be 2D or 3D.")
        
#     return x


@numba.njit(nogil=True)
def _any_nans(a):
    for x in a:
        if np.isnan(x): return True
    return False

@numba.jit
def any_nans(a):
    if not a.dtype.kind=='f': return False
    return _any_nans(a.flat)


def fast_onehot(data, num_classes=None, dtype=np.int8):
    """
    Convert integer array to one-hot encoding using numba-optimized implementation.
    
    Parameters:
    -----------
    data : numpy.ndarray
        1D or 2D array of class indices to convert to one-hot representation
    num_classes : int, optional
        Number of classes in one-hot encoding. If None, inferred as max(data) + 1
    dtype : numpy.dtype
        Data type for the output array (default: np.int8)
    
    Returns:
    --------
    numpy.ndarray
        One-hot encoded array. For 1D input of shape (features,), returns (n, num_classes).
        For 2D input of shape (batch_size, features), returns (batch_size, features, num_classes).
    
    Raises:
    -------
    ValueError
        If input array is not 1D or 2D.
    """
    # Ensure dtype is a numpy dtype otherwise dtype(1) will fail
    # It must be done outside the @numba decorated function due to nopython=True
    dtype = np.dtype(dtype).type
    
    # Compute num_classes outside the jitted function to avoid OptionalType issue
    if num_classes is None:
        num_classes = np.max(data) + 1
    
    if data.ndim == 1:
       return _fast_onehot_1d(data, num_classes, dtype)
    elif data.ndim == 2:
        return _fast_onehot_2d(data, num_classes, dtype)
    else:
        raise ValueError("Input array must be 1D (features, ) or 2D (batch_size, features).")   
    
@numba.jit(nopython=True, parallel=True, cache=True)
def _fast_onehot_1d(data, num_classes=None, dtype=np.int8):
    features = data.shape[0]
    
    # Create output array with specified dtype directly
    output = np.zeros((features, num_classes), dtype=dtype)
    
    # Using parallel=True allows Numba to parallelize this loop
    for i in numba.prange(features):
        # Cast the 1 to the correct dtype to avoid type conversion
        output[i, data[i]] = dtype(1)
            
    return output
    
@numba.jit(nopython=True, parallel=True, cache=True)
def _fast_onehot_2d(data, num_classes=None, dtype=np.int8):
    batch_size, features = data.shape
    
    # Create output array with specified dtype directly
    output = np.zeros((batch_size, features, num_classes), dtype=dtype)
    
    # Using parallel=True allows Numba to parallelize this loop
    for i in numba.prange(batch_size):
        for j in range(features):
            # Cast the 1 to the correct dtype to avoid type conversion
            output[i, j, data[i, j]] = dtype(1)
            
    return output

# Alternative version that might be slightly faster
@numba.jit(nopython=True, parallel=True, cache=True)
def fast_onehot_flat(data):
    """
    Convert 2D integer array to one-hot encoding using flattened indexing
    
    Parameters:
    data: 2D array of shape (batch_size, features) containing class indices
    
    Returns:
    3D array of shape (batch_size, features, num_classes) containing one-hot vectors
    """
    batch_size, features = data.shape
    total_elements = batch_size * features
    num_classes = np.max(data) + 1
    output = np.zeros((total_elements, num_classes), dtype=np.int8)
    
    # Using parallel=True allows Numba to parallelize this loop
    flat_data = data.ravel()
    for i in numba.prange(total_elements):
        output[i, flat_data[i]] = 1
        
    return output.reshape(batch_size, features, num_classes)


@numba.jit(nopython=True, parallel=True, cache=True)
def _fast_fancy_index_copy_2d(data, indices):
    """
    Numba implementation for parallel fancy indexing (copying) on a 2D array.

    Args:
        data (np.ndarray): The 2D source numpy array.
        indices (np.ndarray): A 1D numpy array of integer indices (rows to select).

    Returns:
        np.ndarray: A new 2D numpy array containing the copied rows.
    """
    # Get dimensions
    num_selected_rows = indices.shape[0]
    # data.shape[1] is safe even if data has 0 rows (but indices should also be empty)
    # If data has columns, num_cols > 0. If data has 0 cols, num_cols = 0.
    num_cols = data.shape[1] 
    output_dtype = data.dtype

    # Allocate the output array
    # Use np.empty as we will fill every element. Slightly faster than np.zeros.
    output = np.empty((num_selected_rows, num_cols), dtype=output_dtype)

    # Parallel loop over the *indices* array
    for i in numba.prange(num_selected_rows):
        # Get the actual row index from the input 'data' array
        row_idx_to_copy = indices[i]
        # Copy the entire row from data to output
        # This slicing operation within the loop is efficient in Numba
        output[i, :] = data[row_idx_to_copy, :]

    return output

def fast_fancy_index_copy(data, indices):
    """
    Performs a faster, parallelized copy of rows specified by indices
    from a 2D data array using Numba.

    This mimics the behavior of `data[indices]` but aims for better performance
    on large arrays by parallelizing the copy operation.

    Args:
        data (np.ndarray): The 2D source numpy array.
        indices (np.ndarray): A 1D numpy array of integer indices specifying the
                              rows to copy. Should have an integer dtype.

    Returns:
        np.ndarray: A new 2D numpy array containing the copied rows.

    Raises:
        ValueError: If inputs are not NumPy arrays or have incorrect dimensions/dtypes.
        IndexError: If indices are out of bounds for the data array (optional check).
    """
    # --- Input Validation ---
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Input 'data' must be a 2D NumPy array.")
    if not isinstance(indices, np.ndarray) or indices.ndim != 1:
         raise ValueError("Input 'indices' must be a 1D NumPy array.")
    if not np.issubdtype(indices.dtype, np.integer):
         raise ValueError(f"Input 'indices' must be of an integer dtype, got {indices.dtype}.")

    # Handle empty indices case explicitly
    if indices.shape[0] == 0:
        return np.empty((0, data.shape[1]), dtype=data.dtype)

    # --- Optional: Bounds Checking (can add overhead) ---
    min_idx = np.min(indices)
    max_idx = np.max(indices)
    if min_idx < 0 or max_idx >= data.shape[0]:
       raise IndexError(f"Indices out of bounds for data array with shape {data.shape}")
       
    return _fast_fancy_index_copy_2d(data, indices)


@numba.jit(nopython=True, parallel=True, cache=True)
def _fast_copy_2d(data):
    """
    Create a copy of a 2D array using Numba parallelization.
    
    Args:
        data (np.ndarray): The 2D numpy array to copy.
        
    Returns:
        np.ndarray: A new copy of the input array.
    """
    rows, cols = data.shape
    result = np.empty_like(data)
    
    for i in numba.prange(rows):
        for j in range(cols):
            result[i, j] = data[i, j]
            
    return result

@numba.jit(nopython=True, parallel=True, cache=True)
def _fast_copy_3d(data):
    """
    Create a copy of a 3D array using Numba parallelization.
    
    Args:
        data (np.ndarray): The 3D numpy array to copy.
        
    Returns:
        np.ndarray: A new copy of the input array.
    """
    dim1, dim2, dim3 = data.shape
    result = np.empty_like(data)
    
    for i in numba.prange(dim1):
        for j in range(dim2):
            for k in range(dim3):
                result[i, j, k] = data[i, j, k]
                
    return result

def fast_copy(data):
    if data.ndim == 2:
        return _fast_copy_2d(data)
    elif data.ndim == 3:
        return _fast_copy_3d(data)
    else:
        raise ValueError("Input array must be 2D or 3D.")
    
    
@numba.jit(nopython=True, cache=True)
def calculate_variant_af(variant, missing_value=-127):
    """
    Calculate the Alternate Allele Frequency for a single variant.
    
    Parameters:
    -----------
    variant : numpy.ndarray
        1D array of genotype values across samples.
        Values: missing_value (missing), 0 (homozygous ref), 1 (heterozygous), 2 (homozygous alt)
    missing_value : int
        Value representing missing data (default: -127)
    
    Returns:
    --------
    float
        The alternate allele frequency.
    """
    alt_allele_count = 0
    total_alleles = 0
    
    for j in range(len(variant)):
        if variant[j] != missing_value:
            alt_allele_count += variant[j]  # 0 adds 0, 1 adds 1, 2 adds 2 alt alleles
            total_alleles += 2  # Each genotype has 2 alleles
    
    if total_alleles == 0:
        return np.nan  # All values are missing
    
    # Return alternate allele frequency directly (no minimum calculation)
    return alt_allele_count / total_alleles

@numba.jit(nopython=True, parallel=True, cache=True)
def calculate_af(genotype_matrix, missing_value=-127, variant_axis=0):
    """
    Calculate the Alternate Allele Frequency (AF) for each variant using Numba's parallel capabilities.
    
    Parameters:
    -----------
    genotype_matrix : numpy.ndarray
        2D array with genotype data. Values can be missing_value (missing), 0, 1, or 2.
    missing_value : int
        Value representing missing data (default: -127)
    variant_axis : int
        Axis along which variants are arranged. 0 means each row is a variant (default),
        1 means each column is a variant.
    
    Returns:
    --------
    numpy.ndarray
        Array containing the AF for each variant.
    """
    if variant_axis == 0:
        # Each row is a variant
        n_variants = genotype_matrix.shape[0]
    else:
        # Each column is a variant
        n_variants = genotype_matrix.shape[1]
    
    # Use float32 to reduce memory usage
    afs = np.zeros(n_variants, dtype=np.float32)
    
    for i in numba.prange(n_variants):
        if variant_axis == 0:
            # Extract the i-th variant (row)
            variant = genotype_matrix[i]
        else:
            # Extract the i-th variant (column)
            variant = genotype_matrix[:, i]
        
        afs[i] = calculate_variant_af(variant, missing_value)
    
    return afs

@numba.jit(nopython=True, cache=True)
def impute_variant_inplace(variant, af, missing_value=-127, seed=42, variant_id=0):
    """
    Impute missing values in a variant in-place using binomial sampling based on AF.
    
    Parameters:
    -----------
    variant : numpy.ndarray
        1D array of a single variant with possible missing values.
    af : float
        Alternate allele frequency for this variant.
    missing_value : int
        Value representing missing data (default: -127)
    seed : int
        Random seed for reproducibility.
    variant_id : int
        Identifier for the variant (used to modify the seed).
    """
    # Handle NaN AF
    if np.isnan(af):
        return
    
    if seed is not None:
        np.random.seed(seed + variant_id)  # Seed for reproducibility
    
    # Find missing values and impute them in-place
    for i in range(len(variant)):
        if variant[i] == missing_value:
            variant[i] = np.random.binomial(2, p=af)

@numba.jit(nopython=True, parallel=True, cache=True)
def _impute_genotypes(genotype_matrix, seed=42, missing_value=-127, default_af=0.2, variant_axis=1):
    """
    Impute missing values in the genotype matrix using binomial sampling based on AF.
    
    Parameters:
    -----------
    genotype_matrix : numpy.ndarray
        2D array with genotype data. Values can be missing_value (missing), 0, 1, or 2.
    seed : int
        Random seed for reproducibility.
    missing_value : int
        Value representing missing data (default: -127)
    default_af : float
        Default alternate allele frequency to use when a variant is completely missing (default: 0.2 or 20%)
    variant_axis : int
        Axis along which variants are arranged. 0 means each row is a variant,
        1 means each column is a variant (default).
    
    Returns:
    --------
    numpy.ndarray
        Genotype matrix with missing values imputed.
    """
    
    if seed is not None:
        np.random.seed(seed)
   
    if variant_axis == 0:
        # Each row is a variant
        n_variants = genotype_matrix.shape[0]
    else:
        # Each column is a variant
        n_variants = genotype_matrix.shape[1]
      
    # First calculate AF for each variant
    afs = calculate_af(genotype_matrix, missing_value, variant_axis)
    
    # Now impute missing values
    for i in numba.prange(n_variants):
       
        # Use calculated AF or default if the variant is completely missing (NaN)
        variant_af = default_af if np.isnan(afs[i]) else afs[i]
        
        if variant_axis == 0:
            # Extract and impute the i-th variant (row) in-place
            variant = genotype_matrix[i]
            impute_variant_inplace(variant, variant_af, missing_value, seed=seed, variant_id=i)
        else:
            # For column variants, we need to be more careful with in-place operations
            # Extract the variant as a view
            variant_view = genotype_matrix[:, i]
            impute_variant_inplace(variant_view, variant_af, missing_value, seed=seed, variant_id=i)
    
    return genotype_matrix

def impute_genotypes(genotype_matrix, seed=42, missing_value=-127, default_af=0.2, variant_axis=1, copy=False):
    """
    Impute missing values in the genotype matrix using binomial sampling based on AF.
    
    Parameters:
    -----------
    genotype_matrix : numpy.ndarray
        2D array with genotype data. Values can be missing_value (missing), 0, 1, or 2.
    seed : int
        Random seed for reproducibility.
    missing_value : int
        Value representing missing data (default: -127)
    default_af : float
        Default alternate allele frequency to use when a variant is completely missing (default: 0.2 or 20%)
    variant_axis : int
        Axis along which variants are arranged. 0 means each row is a variant,
        1 means each column is a variant (default).
    copy : bool
        If True, a copy of the genotype matrix is made before imputation (default: False).
    
    Returns:
    --------
    numpy.ndarray
        Genotype matrix with missing values imputed.
    """
    
    # Create a copy of the input array with the same dtype to avoid modifying input
    if copy:
        imputed_matrix = fast_copy(genotype_matrix)
    else:
        imputed_matrix = genotype_matrix
        
    return _impute_genotypes(
        imputed_matrix,
        seed=seed,
        missing_value=missing_value,
        default_af=default_af,
        variant_axis=variant_axis
    )
    
