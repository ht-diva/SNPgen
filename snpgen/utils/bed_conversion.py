"""
Utility functions for converting BED files to HDF5 format with proper allele alignment.

This module provides functions to:
- Load genotype data from BED files
- Handle phenotype data loading and filtering by ethnicity
- Load GWAS summary statistics
- Align alleles between genotype data and GWAS and flip betas if needed
- Save processed data to HDF5 format
"""

import os
import re
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from bed_reader import open_bed


def extract_output_filename(bed_file: str) -> tuple[str, str]:
    """
    Extract output filename and extra info from BED filename.

    Extracts parameters like kb, r, p1, hm3 from the filename to create
    a standardized output filename.

    Args:
        bed_file: Path to the BED file.

    Returns:
        Tuple of (output_filename, extra_info) where extra_info contains
        the extracted parameters as a string suffix.
    """
    filename = os.path.basename(bed_file).replace('.bed', '')

    kb_match = re.search(r'kb(\d+)', filename)
    r_match = re.search(r'r(\d+\.?\d*)', filename)
    p_match = re.search(r'p1(\d+e-?\d+|\d+\.\d+)', filename)
    hm3_match = re.search(r'hm3', filename)

    kb_match = kb_match.group(1) if kb_match else None
    r_match = r_match.group(1) if r_match else None
    p_match = p_match.group(1) if p_match else None
    hm3_match = hm3_match.group(0) if hm3_match else None

    output_filename = 'snp_dataset'
    extra_info = ''

    if kb_match:
        extra_info += f'_kb{kb_match}'
    if r_match:
        extra_info += f'_r{r_match}'
    if p_match:
        extra_info += f'_p1{p_match}'
    if hm3_match:
        extra_info += '_hm3'

    output_filename += extra_info

    return output_filename, extra_info


def load_bed_file(bed_file: str, count_A1: bool = False) -> tuple:
    """
    Load genotype data from a BED file.

    Args:
        bed_file: Path to the BED file.
        count_A1: Whether to count A1 alleles (default: False counts A2).

    Returns:
        Tuple of (G, data, snp_ids, eids, chrom_pos_df) where:
        - G: The bed_reader object
        - data: Genotype data as numpy array (samples x SNPs)
        - snp_ids: Array of SNP IDs
        - eids: Array of sample IDs
        - chrom_pos_df: DataFrame with chromosome and position for each SNP
    """
    bim_file = bed_file.replace('.bed', '.bim')
    fam_file = bed_file.replace('.bed', '.fam')

    G = open_bed(
        bed_file,
        fam_location=fam_file,
        bim_location=bim_file,
        count_A1=count_A1
    )

    data = G.read(dtype='int8')
    snp_ids = G.sid
    eids = G.iid.astype('int64')

    # Build chromosome/position DataFrame
    df_data = []
    for chrom, pos in zip(G.chromosome, G.bp_position):
        df_data.append([int(chrom), int(pos)])

    chrom_pos_df = pd.DataFrame(df_data, columns=['chrom', 'pos'])
    chrom_pos_df.index = snp_ids.tolist()

    return G, data, snp_ids, eids, chrom_pos_df


def verify_snp_ordering(snp_ids: np.ndarray, chrom_pos_df: pd.DataFrame, verbose: bool = True) -> bool:
    """
    Verify that SNPs are correctly ordered by chromosome and position.

    Args:
        snp_ids: Array of SNP IDs.
        chrom_pos_df: DataFrame with chromosome and position for each SNP.
        verbose: Whether to show progress bar.

    Returns:
        True if SNPs are correctly ordered.

    Raises:
        AssertionError: If SNPs are not correctly ordered.
    """
    last_chrom = 1
    last_pos = 0

    iterator = tqdm(snp_ids, desc="Verifying SNP order") if verbose else snp_ids

    for snp in iterator:
        sel_snp = chrom_pos_df.loc[snp]
        chrom = sel_snp['chrom']
        pos = sel_snp['pos']

        if chrom > last_chrom:
            last_pos = 0

        assert (last_chrom <= chrom and last_pos < pos), \
            f"Error on SNP {snp}.\n\tchrom: {chrom}\n\tpos: {pos}\n\tlast_chrom: {last_chrom}\n\tlast_pos: {last_pos}"

        last_pos = pos
        last_chrom = chrom

    return True


def load_phenotype(pheno_file: str) -> pd.DataFrame:
    """
    Load phenotype data from CSV file.

    Args:
        pheno_file: Path to the binary phenotype CSV file.

    Returns:
        Series with phenotype data indexed by f.eid.
    """
    print(f"Loading phenotype from {pheno_file}")
    pheno_df = pd.read_csv(pheno_file)
    pheno_df.set_index('f.eid', inplace=True)
    pheno_df = pheno_df['phenotype'].astype('i1')
    pheno_df.name = 'phenotype'

    return pheno_df


def filter_by_ethnicity(
    pheno_df: pd.DataFrame,
    desired_ethnicity: list,
    ancestry_file: str,
    ethnicity_coding_file: str,
    ethnicity_field: str = 'Ethnic_background.0.0'
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter phenotype data by ethnicity.

    Args:
        pheno_df: Phenotype DataFrame indexed by f.eid.
        desired_ethnicity: List of ethnicity codes to keep (empty for all).
        ancestry_file: Path to ancestry TSV file.
        ethnicity_coding_file: Path to ethnicity coding TSV file.
        ethnicity_field: Column name for ethnicity in ancestry file.

    Returns:
        Tuple of (filtered_pheno_df, ancestry_df).
    """
    ancestry_df = pd.read_csv(
        ancestry_file,
        sep='\t',
        index_col='f.eid',
        usecols=['f.eid', ethnicity_field]
    )
    ethnicity_coding = pd.read_csv(ethnicity_coding_file, sep='\t')

    ancestry_df[ethnicity_field] = ancestry_df[ethnicity_field].fillna(-1)
    ancestry_df[ethnicity_field] = ancestry_df[ethnicity_field].astype(np.int32)

    # Map sub-ethnicity to main ethnicity
    conditions = ethnicity_coding['parent_id'] != 0
    mapping_dict = dict(zip(
        ethnicity_coding.loc[conditions, 'coding'],
        ethnicity_coding.loc[conditions, 'parent_id']
    ))

    # Build list of actual desired ethnicities (including sub-ethnicities)
    actual_desired_ethnicities = []
    for eth in desired_ethnicity:
        if eth in mapping_dict.values():
            sub_ethnicities = ethnicity_coding[ethnicity_coding['parent_id'] == eth]['coding'].tolist()
            sub_ethnicities.append(eth)  # also include the main ethnicity because some samples may be labeled with it directly
            actual_desired_ethnicities.extend(sub_ethnicities)
        else:
            actual_desired_ethnicities.append(eth)

    actual_desired_ethnicities = list(set(actual_desired_ethnicities))

    if len(actual_desired_ethnicities) > 0:
        print(f"Filtering by ethnicity codes: {actual_desired_ethnicities}")
        eids_to_keep = ancestry_df[ancestry_df[ethnicity_field].isin(actual_desired_ethnicities)].index.values

        print(f"Samples before ethnicity filter: {len(pheno_df)}")
        pheno_df = pheno_df.loc[pheno_df.index.isin(eids_to_keep)]
        print(f"Samples after ethnicity filter: {len(pheno_df)}")
    else:
        print("Keeping all ethnicities")
        eid_is_in = pheno_df.index.isin(ancestry_df.index)
        if sum(eid_is_in) < len(eid_is_in):
            print(f"{len(eid_is_in) - sum(eid_is_in)} samples removed (not in ancestry file)")
        pheno_df = pheno_df.loc[eid_is_in]
        print(f"Total samples: {len(pheno_df)}")

    return pheno_df, ancestry_df


def load_gwas(
    gwas_file: str,
    rsid_col: str = 'rsid',
    beta_col: str = 'Effect',
    p_value_col: str = 'P-value',
    effect_allele_col: str = 'Allele1'
) -> pd.DataFrame:
    """
    Load GWAS summary statistics from a file.

    Args:
        gwas_file: Path to the GWAS file.
        rsid_col: Column name for rsIDs.
        beta_col: Column name for effect sizes.
        p_value_col: Column name for p-values.
        effect_allele_col: Column name for effect alleles.

    Returns:
        DataFrame with GWAS data indexed by rsID.
    """
    effects_df = pd.read_csv(
        gwas_file,
        sep='\t',
        index_col=rsid_col,
        low_memory=False
    )
    return effects_df


def deduplicate_gwas_by_alleles(
    gwas_df: pd.DataFrame,
    snp_ids: np.ndarray,
    bed_alleles: dict,
    effect_allele_col: str,
    p_value_col: str,
    baseline_allele_col: str = 'Baseline.Meta',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Deduplicate GWAS entries for SNPs, preferring allele-matching entries.

    When multiple GWAS entries exist for the same SNP ID, this function:
    1. Preferentially keeps the entry whose alleles match the BED file
    2. Falls back to keeping the entry with the lowest p-value if no match

    Args:
        gwas_df: GWAS DataFrame indexed by rsID.
        snp_ids: Array of SNP IDs to deduplicate.
        bed_alleles: Dictionary mapping SNP IDs to (allele_1, allele_2) tuples.
        effect_allele_col: Column name for effect allele in GWAS.
        p_value_col: Column name for p-values in GWAS.
        baseline_allele_col: Column name for baseline/reference allele.
        verbose: Whether to print statistics.

    Returns:
        Deduplicated DataFrame containing only the SNPs from snp_ids.
    """
    # Extract only the relevant SNPs
    relevant_gwas = gwas_df.loc[snp_ids].copy()
    
    # Check for duplicates
    n_duplicates = relevant_gwas.index.duplicated().sum()
    if n_duplicates == 0:
        if verbose:
            print("No duplicate SNP entries found in GWAS")
        return relevant_gwas
    
    if verbose:
        print(f"Found {n_duplicates} duplicate SNP entries in GWAS")
    
    # For each row, check if alleles match BED (in either orientation)
    def alleles_match_bed(row):
        rsid = row.name
        if rsid not in bed_alleles:
            return False
        bed_a1, bed_a2 = bed_alleles[rsid]
        gwas_effect = row[effect_allele_col]
        gwas_baseline = row.get(baseline_allele_col, row.get('Baseline', None))
        if gwas_baseline is None:
            return True  # Can't check, assume it matches
        # Check if alleles match (either orientation)
        return ({gwas_effect, gwas_baseline} == {bed_a1, bed_a2})
    
    # Add match indicator
    relevant_gwas['_alleles_match'] = relevant_gwas.apply(alleles_match_bed, axis=1)
    
    # Track statistics for duplicates only
    duplicate_mask = relevant_gwas.index.duplicated(keep=False)
    duplicated_snps = relevant_gwas[duplicate_mask]
    
    # Count how many duplicates have at least one allele match
    dup_with_match = duplicated_snps.groupby(level=0)['_alleles_match'].any()
    dup_without_match = duplicated_snps.groupby(level=0)['_alleles_match'].any() == False
    total_dup_snps = duplicated_snps.index.nunique()
    
    # Sort by: allele match (True first), then p-value (lowest first)
    relevant_gwas = relevant_gwas.sort_values(['_alleles_match', p_value_col], ascending=[False, True])
    
    # Keep first occurrence of each SNP
    kept = relevant_gwas.loc[~relevant_gwas.index.duplicated(keep='first')]
    
    # Drop the temporary column
    kept = kept.drop(columns=['_alleles_match'])
    
    if verbose:
        print(f"Deduplication results:")
        print(f"  Total duplicated SNPs: {total_dup_snps}")
        print(f"  Kept by allele match: {dup_with_match.sum()}")
        print(f"    e.g.: {dup_with_match[dup_with_match].index[:3].values}")
        print(f"  Kept by lowest p-value: {dup_without_match.sum()}")
        print(f"    e.g.: {dup_without_match[dup_without_match].index[:3].values}")
        print(f"After deduplication: {len(kept)} unique SNPs")
    
    return kept


def align_alleles_and_flip_betas(
    snp_ids: np.ndarray,
    bed_allele_1: np.ndarray,
    bed_allele_2: np.ndarray,
    effects_df: pd.DataFrame,
    effect_allele_col: str,
    beta_col: str,
    verbose: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align alleles between BED file and GWAS and flip betas where needed.

    When the GWAS effect allele matches the BED A2 allele (instead of A1),
    the beta is flipped (negated) to ensure consistency.

    Args:
        snp_ids: Array of SNP IDs from the BED file.
        bed_allele_1: Array of A1 alleles from the BED file.
        bed_allele_2: Array of A2 alleles from the BED file.
        effects_df: GWAS DataFrame indexed by rsID.
        effect_allele_col: Column name for effect alleles in GWAS.
        beta_col: Column name for betas in GWAS.
        verbose: Whether to print alignment statistics.

    Returns:
        Tuple of (aligned_betas, flip_mask) where:
        - aligned_betas: Betas with signs flipped where needed
        - flip_mask: Boolean array indicating which SNPs were flipped
    """
    # Get effect alleles from GWAS aligned to SNP order
    gwas_effect_alleles = effects_df.loc[snp_ids, effect_allele_col].values
    betas = effects_df.loc[snp_ids, beta_col].astype(float).values

    # Decode bytes if needed
    if isinstance(bed_allele_1[0], bytes):
        bed_allele_1 = np.array([x.decode() for x in bed_allele_1])
    if isinstance(bed_allele_2[0], bytes):
        bed_allele_2 = np.array([x.decode() for x in bed_allele_2])

    # Convert to uppercase for comparison
    bed_a1_upper = np.char.upper(bed_allele_1.astype(str))
    bed_a2_upper = np.char.upper(bed_allele_2.astype(str))
    gwas_upper = np.char.upper(gwas_effect_alleles.astype(str))

    # Check allele matching
    match_a1 = (gwas_upper == bed_a1_upper)  # GWAS effect allele matches BED A1 → keep beta
    match_a2 = (gwas_upper == bed_a2_upper)  # GWAS effect allele matches BED A2 → flip beta
    neither = ~match_a1 & ~match_a2  # Neither matches (possible strand flip)

    if verbose:
        print(f"Allele alignment statistics:")
        print(f"  GWAS effect allele matches BED A1 (no flip): {match_a1.sum()}")
        print(f"  GWAS effect allele matches BED A2 (flip beta): {match_a2.sum()}")
        print(f"  Neither matches (strand flip or error): {neither.sum()}")

    # Handle strand flips (A↔T, C↔G)
    if neither.sum() > 0:
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

        def get_complement(allele):
            return complement.get(allele, allele)

        gwas_complement = np.array([get_complement(a) for a in gwas_upper])

        # Check if complement matches
        complement_match_a1 = neither & (gwas_complement == bed_a1_upper)
        complement_match_a2 = neither & (gwas_complement == bed_a2_upper)
        unresolved = neither & ~complement_match_a1 & ~complement_match_a2

        if verbose:
            print(f"  Strand flip resolved (matches A1): {complement_match_a1.sum()}")
            print(f"  Strand flip resolved (matches A2, flip beta): {complement_match_a2.sum()}")
            if unresolved.sum() > 0:
                print(f"  WARNING: {unresolved.sum()} SNPs could not be aligned!")
                unresolved_idx = np.where(unresolved)[0][:5]
                for idx in unresolved_idx:
                    print(f"    SNP {snp_ids[idx]}: BED A1/A2={bed_allele_1[idx]}/{bed_allele_2[idx]}, GWAS={gwas_effect_alleles[idx]}")

        # Update match arrays with strand flip results
        match_a1 = match_a1 | complement_match_a1
        match_a2 = match_a2 | complement_match_a2

    # Flip betas where GWAS effect allele matches BED A2
    flip_mask = match_a2
    aligned_betas = betas.copy()
    aligned_betas[flip_mask] = -aligned_betas[flip_mask]

    if verbose:
        print(f"  Total betas flipped: {flip_mask.sum()}")

    return aligned_betas, flip_mask


def save_to_hdf5(
    filepath: str,
    data: np.ndarray,
    labels: np.ndarray,
    snp_ids: np.ndarray,
    betas: np.ndarray,
    p_values: np.ndarray,
    eids: np.ndarray,
    ancestry: np.ndarray,
    chrom: np.ndarray,
    pos: np.ndarray,
    allele_1: np.ndarray,
    allele_2: np.ndarray,
    beta_flipped: np.ndarray = None,
    compression: str = 'gzip'
):
    """
    Save processed genotype data to HDF5 file.

    Args:
        filepath: Output HDF5 file path.
        data: Genotype data (samples x SNPs).
        labels: Binary phenotype labels.
        snp_ids: SNP identifiers.
        betas: Effect sizes (aligned to genotype allele coding).
        p_values: GWAS p-values.
        eids: Sample identifiers.
        ancestry: Ancestry codes for each sample.
        chrom: Chromosome for each SNP.
        pos: Position for each SNP.
        allele_1: A1 alleles from BED file.
        allele_2: A2 alleles from BED file.
        beta_flipped: Optional boolean array indicating which betas were flipped.
        compression: Compression algorithm (default: 'gzip').
    """
    # Convert string arrays to bytes for HDF5 compatibility
    def to_bytes_array(arr):
        """Convert string/object array to bytes array for HDF5."""
        if arr.dtype == object or arr.dtype.kind == 'U':
            return np.array([x.encode('utf-8') if isinstance(x, str) else x for x in arr])
        return arr

    with h5py.File(filepath, "w") as hf:
        hf.create_dataset("data", data=data, compression=compression)
        hf.create_dataset("labels", data=labels, compression=compression)
        hf.create_dataset("snp_ids", data=to_bytes_array(snp_ids), compression=compression)
        hf.create_dataset("betas", data=betas, compression=compression)
        hf.create_dataset("p_values", data=p_values, compression=compression)
        hf.create_dataset("eids", data=eids, compression=compression)
        hf.create_dataset("ancestry", data=ancestry, compression=compression)
        hf.create_dataset("chrom", data=chrom, compression=compression)
        hf.create_dataset("pos", data=pos, compression=compression)
        hf.create_dataset("allele_1", data=to_bytes_array(allele_1), compression=compression)
        hf.create_dataset("allele_2", data=to_bytes_array(allele_2), compression=compression)
        if beta_flipped is not None:
            hf.create_dataset("beta_flipped", data=beta_flipped, compression=compression)

    print(f"Saved to {filepath}")


def select_top_k_snps(
    top_k: int,
    data: np.ndarray,
    snp_ids: np.ndarray,
    betas: np.ndarray,
    p_values: np.ndarray,
    chrom: np.ndarray,
    pos: np.ndarray,
    allele_1: np.ndarray,
    allele_2: np.ndarray,
    beta_flipped: np.ndarray = None
) -> tuple:
    """
    Select top K SNPs based on p-values while preserving genomic order.

    Args:
        top_k: Number of top SNPs to select.
        data: Genotype data (samples x SNPs).
        snp_ids: SNP identifiers.
        betas: Effect sizes.
        p_values: GWAS p-values.
        chrom: Chromosome for each SNP.
        pos: Position for each SNP.
        allele_1: A1 alleles.
        allele_2: A2 alleles.
        beta_flipped: Optional boolean array indicating which betas were flipped.

    Returns:
        Tuple of filtered arrays maintaining genomic order.
    """
    # Get indices of top K SNPs sorted by p-value, then sort by original position
    top_k_indices = np.sort(np.argsort(p_values)[:top_k])

    result = (
        data[:, top_k_indices],
        snp_ids[top_k_indices],
        betas[top_k_indices],
        p_values[top_k_indices],
        chrom[top_k_indices],
        pos[top_k_indices],
        allele_1[top_k_indices],
        allele_2[top_k_indices],
        beta_flipped[top_k_indices] if beta_flipped is not None else None,
        top_k_indices
    )
    return result
