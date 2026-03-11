################### Python module to preprocess genotype data #########################

# The code below relies on the python library scikit-allel (https://scikit-allel.readthedocs.io/en/stable/index.html)

import sys
import os
import allel
import numpy as np
import h5py
import pandas as pd
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from snpgen.data.utils import fast_copy, any_nans


def _map_elements(arr, nan_value=None):
    """
    Maps a 2D array of genotype data to a 3D array representing alleles.
    This function takes a 2D numpy array where each element represents a genotype:
    - 0 for homozygous for the reference allele
    - 1 for heterozygous (one ref allele, one alt allele)
    - 2 for homozygous for the alternate allele
    - NaN for missing data

    It returns a 3D numpy array where the last dimension represents the two alleles:
    - [0, 0] for homozygous reference
    - [0, 1] for heterozygous
    - [1, 1] for homozygous alternate
    - [-1, -1] for missing data
    
    Parameters:
    arr (numpy.ndarray): A 2D numpy array of genotype data.
    nan_value (float or int, optional): The value representing missing data in the input array. Defaults to np.nan.
    
    Returns:
    numpy.ndarray: A 3D numpy array with shape (arr.shape[0], arr.shape[1], 2) representing the alleles.
    """
    
    print("Mapping genotype data to alleles...")
    
    # Preallocate the final 3D array directly
    print("  Preallocating output array with zeros...")
    out = np.zeros((arr.shape[0], arr.shape[1], 2), dtype=np.int8)
    print("  Done creating output array with zeros.")
    
    # Much slower and memory intensive
    # out[np.isnan(arr)] = [-1, -1]
    # #out[arr == 0] = [0, 0] # this is not necessary since the array is already initialized with [0,0]
    # out[arr == 1] = [0, 1]
    # out[arr == 2] = [1, 1]
    
    # Vectorized assignments for first and second alleles
    out[..., 1][arr == 1] = 1  # Set second allele for heterozygous
    homozygous_alt_mask = arr == 2
    out[..., 0][homozygous_alt_mask] = 1  # Set first allele for homozygous alt
    out[..., 1][homozygous_alt_mask] = 1  # Set second allele for homozygous alt
    del homozygous_alt_mask
    
    # Handle NaN values - set both alleles to -1
    if nan_value is None or nan_value is np.nan:
        nan_mask = np.isnan(arr)
    else:
        if any_nans(arr):
            print(f"Warning: Found NaN values in the array when checking for nan_value={nan_value}. This may cause unexpected behavior (i.e. NaN values will become 0).")
        nan_mask = arr == nan_value
    out[nan_mask] = [-1, -1]
    
    print("Done mapping elements.")
    
    return out

def dataframe_to_genotype_array(df, sample_axis=0, map_elements=True, verbose=False, nan_value=None):
    """
    Convert a pandas dataframe of genotypes to a GenotypeArray.

    Args:
        df (pandas.DataFrame): Genotype data. 2D dataframe with samples and variants as axes.
        sample_axis (int): Axis of the dataframe that corresponds to the samples.
        map_elements (bool): Whether to map the elements of the dataframe to the values expected by scikit-allel.

    Returns:
        GenotypeArray: Genotype data.
    """

    # Convert to numpy array
    arr = df.to_numpy()

    # Map elements of the dataframe to the values expected by scikit-allel,
    # which is needed if dataframe is in the form of 0, 1, 2, or NaN and not
    # in the form of [0,0], [0,1], [1,1], [-1,-1] (i.e. 0/0, 0/1, 1/1, or ./.)
    if map_elements:
        arr = _map_elements(arr, nan_value=nan_value)

    # Transpose array if necessary
    if sample_axis == 0:
        print("Transposing array...")
        arr = np.transpose(arr, (1,0,2)) # scikit-allel expects input in the form (variants, samples)
        print("Done transposing array.")

    # Convert to list of lists (required by scikit-allel)
    # gen = [list(row) for row in arr] # no idea why this was here

    # Convert to GenotypeArray
    print("Converting to GenotypeArray...")
    gen = allel.GenotypeArray(arr, dtype='i1')
    print("Done converting to GenotypeArray.")

    return gen


def import_unpack_genotype(infile, region=None, samples=None, snp_df=None):
    """
    Import and unpack VCF files.

    Args:
        infile (str): Path to the input VCF file.
        region (str, optional): Genomic region to extract variants for.
            If provided, should be a tabix-style region string, which can be either
            just a chromosome name (e.g., '2L'), or a chromosome name followed by
            1-based beginning and end coordinates (e.g., '2L:100000-200000').
            Note that only variants whose start position (POS) is within the requested
            range will be included.
        samples (list of str, optional): Selection of samples to extract calldata for.
            If provided, should be a list of strings giving sample identifiers.
            May also be a list of integers giving indices of selected samples.
        snp_df (pd.DataFrame, optional): A DataFrame containing SNP data, with columns 'markername', 'effect_allele', and 'noneffect_allele'.
            If provided, the function will remove multiallelic variants from the VCF dictionary based on the SNP DataFrame.

    Returns:
        GenotypeArray: Genotype data (numpy array-like structure). 
        Array of discrete genotype calls for a matrix of variants and samples.
        
        This class represents data on discrete genotype calls as a 3-dimensional numpy array of integers. 
        By convention, the first dimension corresponds to the variants genotyped, the second dimension 
        corresponds to the samples genotyped, and the third dimension corresponds to the ploidy of the samples.
        Each integer within the array corresponds to an allele index, where 0 is the reference allele, 1 is the 
        first alternate allele, 2 is the second alternate allele, … and -1 (or any other negative integer) 
        is a missing allele call.
        
        Samples: Array of samples IDs
        
    """

    print("Importing and unpacking VCF file...")
    vcf = allel.read_vcf(infile, region=region, samples=samples, log=sys.stderr)

    if snp_df is not None:
        print("Cleaning multiallelic variants...")
        vcf = _clean_multiallelic(vcf, snp_df)

    gen = allel.GenotypeArray(vcf['calldata/GT'])

    samples = vcf['samples']
    variants = [rsid.split(';')[0] for rsid in vcf['variants/ID']]

    return gen, samples, variants

def _clean_multiallelic(vcf_dict, snp_df):
    """
    Clean multiallelic variants from a VCF dictionary based on a SNP DataFrame.
    When we extract SNPs based on their rsid from a .bgen using bgenix, we may get duplicated variants if the variant is multiallelic.
    For example, in the resulting VCF, we may have the following:
    #CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO    FORMAT  0
    9	22049130	rs10738605;9:22049130_C_A   C    A	.	.	.	GT	0/0
    9	22049130	rs10738605;9:22049130_C_G   C    G	.	.	.	GT	0/1
    where the same variant rs10738605 is duplicated because it is multiallelic.

    Using the SNP DataFrame, we can remove the duplicated variants by keeping only the rows where the alleles match with those in which
    we are interested. For example, if we are interested in the variant rs10738605 with alleles C and A, we can remove the duplicated variant
    rs10738605 with alleles C and G.

    Args:
        vcf_dict (dict): A dictionary containing VCF data obtained from allel.read_vcf().
        snp_df (pd.DataFrame): A DataFrame containing SNP data, with columns 'markername', 'effect_allele', and 'noneffect_allele'.

    Returns:
        dict: A cleaned VCF dictionary with multiallelic duplicated variants removed based on the SNP DataFrame.
    """
    variants_alt = [alt[0] for alt in vcf_dict['variants/ALT']] 
    cleaned_variants_id = [rsid.split(';')[0] for rsid in vcf_dict['variants/ID']]
    vcf_df = pd.DataFrame({'clean_rsid': cleaned_variants_id, 'rsid': vcf_dict['variants/ID'], 'ref': vcf_dict['variants/REF'], 'alt': variants_alt})

    duplicated_rsid = list(set([rsid for rsid in cleaned_variants_id if cleaned_variants_id.count(rsid) > 1])) # rsid of duplicated variants
    
    if len(duplicated_rsid)  == 0:
        return vcf_dict
    
    print(f"Found {len(duplicated_rsid)} duplicated variants. Cleaning...")

    idx_to_delete = []

    for rsid in duplicated_rsid:
        a_df = snp_df[snp_df.markername == rsid] # SNP data for the specific duplicated variant
        df_rsid = vcf_df[vcf_df.clean_rsid == rsid] # VCF data for the specific duplicated variant
        for idx, row in a_df.iterrows():
            # keep only the rows where the alleles match
            allel_condition =    (df_rsid.ref == row.effect_allele) & (df_rsid.alt == row.noneffect_allele) \
                               | (df_rsid.ref == row.noneffect_allele) & (df_rsid.alt == row.effect_allele)
            # to_keep = df_rsid[allel_condition]
            to_delete = df_rsid[~allel_condition]
            idx_to_delete += to_delete.index.values.tolist()

    cleaned_vcf_df = vcf_df[~vcf_df.index.isin(idx_to_delete)] # remove unwanted duplicated variants

    good_indices = cleaned_vcf_df.index.values.tolist()

    cleaned_vcf = {k: v[good_indices] for k, v in vcf_dict.items() if k != 'samples'} # rebuild the cleaned vcf dictionary
    cleaned_vcf['samples'] = vcf_dict['samples']

    return cleaned_vcf


def base_preprocessing(gen, output_filename=None, output_path=None,
                       prune_ld=True, ld_iter=3, prune_size=200, rescale=False,
                       save_allele_counts=False, seed=None):
    """
    Base preprocessing for SNP data.

    Args:
        gen (GenotypeArray): Genotype data.
        output_filename (str): Name of the file to be saved.
        output_path (str): Path for saving results.
        prune_ld (bool): Whether to prune for LD.
        ld_iter (int): Number of LD pruning iterations.
        prune_size (int): Size for LD pruning.
        rescale (bool): Scale genotype data in [0,1]. 
                        0 = homozygous reference, 
                        0.5 = heterozygous, 
                        1 = homozygous alternate
        save_allele_counts (bool): Whether to save allele counts.
        seed (int): Random seed.

    Returns:
        GenotypeArray: Processed genotype data.
        numpy.ndarray: Indices of the kept variants.
    """
    if seed:
        np.random.seed(seed) # set random seed for reproducibility of missing data imputation

    # Initialize an array to keep track of original indices
    kept_indices = np.arange(gen.shape[0])
    original_indices = np.arange(gen.shape[0])

    # SNP filters
    print("Counting alleles")
    # Count the number of calls of each allele per variant.
    # e.g. if for a given variant we have 3 samples 0/0, 1/1, 0/1, then the count will be 3 | 3,
    # because we have three times '0' and three times '1'.
    ac_all = gen.count_alleles() 
    # Count of alleles per snp per individual
    # e.g. if for a given variant we have 3 samples 0/0, 1/1, 0/1, then the allele counts will 2:0, 0:2, 1:1
    # because we have two times '0', two times '1', and one time '0' and '1'.
    ac = gen.to_allele_counts() 

    print("Dropping non-biallelic sites")
    # a biallelic variant is a genetic variant with two possible alleles (variants) at a specific genomic position.
    biallel = ac_all.is_biallelic()
    dc_all = ac_all[biallel, 1]
    dc = np.array(ac[biallel, :, 1], dtype="int_")
    missingness = gen[biallel, :, :].is_missing()
    kept_indices = kept_indices[biallel]
    print(f"  Dropped {np.sum(~biallel)} non-biallelic sites")

    print("Dropping singletons")
    # filtering out variants where the derived allele count is less than or equal to 2, 
    # which implies that these variants are observed in only one or two individuals. 
    # In population genetics, "singletons" often refer to variants that are observed in only one individual in a sample.
    ninds = np.array([np.sum(x) for x in ~missingness])
    singletons = np.array([x <= 2 for x in dc_all])
    dc_all = dc_all[~singletons]
    dc = dc[~singletons, :]
    ninds = ninds[~singletons]
    missingness = missingness[~singletons, :]
    kept_indices = kept_indices[~singletons]
    print(f"  Dropped {np.sum(singletons)} singletons")

    print("Filling missing data with rbinom(2,derived_allele_frequency)")
    # Compute allele frequency for each SNP
    af = np.array([dc_all[x] / (ninds[x] * 2) for x in range(dc_all.shape[0])])
    # Impute missing values with Binomial distribution
    for i in tqdm(range(np.shape(dc)[1])):
        indmiss = missingness[:, i]
        dc[indmiss, i] = np.random.binomial(2, af[indmiss])

    if prune_ld:
        print("Pruning genotypes for linkage disequilibrium")
        dc, ld_indices = ld_prune(dc, ld_iter, prune_size, step=200, threshold=0.1)
        kept_indices = kept_indices[ld_indices]

    dc = np.transpose(dc)

    if rescale:
        dc = dc * 0.5  

    print("Genotype preprocessing done.")

    # Save hdf5 for reanalysis
    if save_allele_counts:
        print("Saving derived counts for reanalysis")
        with h5py.File(os.path.join(output_path, output_filename + ".hdf5"), "w") as hf:
            hf.create_dataset("derived_counts", data=dc, compression="gzip")
        print("  Done saving derived counts.")

    # Calculate removed indices
    #removed_indices = np.setdiff1d(original_indices, kept_indices)

    return dc, kept_indices


def ld_prune(gen, ld_iter, size, step, threshold):
    """
    Prune genotypes for linkage disequilibrium (i.e., the function removes variants that are highly correlated)

    Args:
        gen (GenotypeArray): Genotype data.
        ld_iter (int): Number of LD pruning iterations.
        size (int): Window size (number of variants).
        step (int): Number of variants to advance to the next window.
        threshold (float): Maximum value of r**2 to include variants.
    Returns:
        GenotypeArray: Pruned genotype data.
        numpy.ndarray: Indices of the variants that were kept.
    
        
    Note: The value of r**2 between each pair of variants is calculated using the method of Rogers and Huff (2008).
    """

    kept_indices = np.arange(gen.shape[0])
    
    for i in range(ld_iter):
        loc_unlinked = allel.locate_unlinked(gen, size=size, step=step, threshold=threshold)
        n = np.count_nonzero(loc_unlinked)
        n_remove = gen.shape[0] - n
        print('Iteration', i + 1, 'Retaining', n, 'Removing', n_remove, 'variants')
        gen = gen.compress(loc_unlinked, axis=0)
        kept_indices = kept_indices[loc_unlinked]
    
    return gen, kept_indices


def perform_pca(dc, n_components, pca_type='standard', save_pca=True, output_path=None):
    """
    Perform PCA on the genotype data.

    Args:
        dc (numpy.ndarray): Genotype data.
        n_components (int): Number of principal components.
        pca_type (str): Type of PCA ('standard' or 'fast').
        save_pca (bool): Whether to save PCA results.
        output_path (str): Path for saving PCA results.

    Returns:
        Tuple: PCA components, PCA model.
    """

    if pca_type:
        print("Performing %s PCA and keeping %i Principal Components" % (pca_type, n_components))

        if pca_type == "fast":
            pca_model = PCA(n_components, svd_solver='randomized')
            pca_components = pca_model.fit_transform(dc)
            variance_explained = sum(pca_model.explained_variance_ratio_)
            print("This set of PCs explains %d percent of the total variance" % (variance_explained * 100))

        elif pca_type == "standard":
            pca_model = PCA(n_components, svd_solver='full')
            pca_components = pca_model.fit_transform(dc)
            variance_explained = sum(pca_model.explained_variance_ratio_)
            print("This set of PCs explains %d percent of the total variance" % (variance_explained * 100))
        else:
            raise ValueError("Solver not implemented")


    if save_pca:
        print("Saving PCA results.")
        cols = ['PC{}'.format(x) for x in range(1, n_components + 1)]
        pd.DataFrame(pca_components, columns=cols).to_hdf(
            output_path + "/%sPCA_%icomponents.h5" % (pca_type, n_components), key='pca'
        )

    print("PCA done.")

    return pca_components, pca_model