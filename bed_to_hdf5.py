#!/usr/bin/env python3
"""Prepare genotypes using the PLINK BED, phenotype and GWAS inputs.

Configuration values come from command-line arguments.
"""

import argparse
import os
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description="Convert PLINK BED + phenotype + GWAS to HDF5 using the genotype-processing pipeline.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--bed-file', help='PLINK .bed input, with matching .bim/.fam alongside; already LD-clumped.')
    source.add_argument('--bed-prefix', help='PLINK prefix without .bed/.bim/.fam.')
    parser.add_argument('--phenotype-file', '--phenotype', required=True, help="CSV with f.eid and phenotype (binary 0/1); see input examples in Run Settings.")
    parser.add_argument('--gwas-file', '--gwas', required=True, help='Tab-delimited GWAS summary statistics; column names are configurable below.')
    parser.add_argument('--ancestry-file', help='Optional TSV with f.eid and the ethnicity field.')
    parser.add_argument('--ethnicity-coding-file', help='Hierarchical ethnicity coding TSV: coding, meaning, parent_id.')
    parser.add_argument('--ethnicity-field', default='Ethnic_background.0.0')
    parser.add_argument('--desired-ethnicity', nargs='*', type=int, default=[], help='Codes retained by the ethnicity helper, including immediate subgroups; empty keeps all.')
    parser.add_argument('--gwas-rsid-col', default='markername')
    parser.add_argument('--beta-column', default='beta')
    parser.add_argument('--p-value-column', default='p_dgc')
    parser.add_argument('--effect-allele-column', default='effect_allele')
    parser.add_argument('--other-allele-column', default='noneffect_allele')
    parser.add_argument('--top-k', type=int, default=2048, help='Top SNP count by GWAS p-value; 0 saves all only.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--count-a1', action='store_true', help='Count A1; default A2. Betas follow the counted allele.')
    parser.add_argument('--skip-order-check', action='store_true', help='Skip the chromosome/position ordering assertion.')
    parser.add_argument('--output-dir', help='Default: BED directory.')
    parser.add_argument('--output-name', help='Default: the BED filename extraction.')
    parser.add_argument('--output-suffix', default='WHITE', help='filename suffix; pass an empty string to omit it.')
    parser.add_argument('--output', help='Explicit all-SNP HDF5 filename; top-K defaults to NAME_topK.hdf5.')
    parser.add_argument('--top-output', help='Explicit top-K HDF5 filename.')
    parser.add_argument('--snp-ids-output', help='Explicit top-K SNP-ID text filename.')
    parser.add_argument('--overwrite', action='store_true')
    return parser



def run(args):
    if args.top_k < 0:
        raise ValueError('--top-k must be non-negative')
    import os
    import pandas as pd
    import numpy as np

    from sklearn.metrics import roc_auc_score

    from snpgen.utils import genotype_handler as g_handler
    from snpgen.utils import bed_conversion as bed_conv

    # =============================================================================
    # Run Settings
    # =============================================================================

    # Input BED file: PLINK .bed file (with .bim and .fam in the same directory).
    # The BED should already contain LD-clumped SNPs.
    BED_FILE = str(Path(args.bed_file).expanduser()) if args.bed_file else str(Path(args.bed_prefix).expanduser()) + '.bed'

    # Phenotype file: CSV with columns 'f.eid' (sample ID) and 'phenotype' (0/1 binary label).
    # Example content:
    #   f.eid,phenotype
    #   1000015,0
    #   1000027,0
    PHENO_FILE = args.phenotype_file

    # GWAS summary statistics: tab-delimited file with SNP-level association results.
    # Expected columns include: markername, effect_allele, noneffect_allele, beta, p_dgc (or equivalent).
    # Example content:
    #   markername	chr	bp_hg19	effect_allele	noneffect_allele	beta	p_dgc
    #   rs143225517	1	751756	C	T	.013006	.4528019
    #   rs3094315	1	752566	A	G	-.005243	.7394597
    GWAS_FILE = args.gwas_file

    # Ancestry file: TSV with 'f.eid' and an ethnicity field column (e.g., 'Ethnic_background.0.0').
    # Example content:
    #   f.eid	Sex.0.0	Year_of_birth.0.0	Ethnic_background.0.0	...
    #   1000015	0	1944	1001
    #   1000027	0	1954	1001
    ANCESTRY_FILE = args.ancestry_file

    # Ethnicity coding file: TSV with hierarchical ethnicity codes.
    # Required columns: 'coding' (int), 'meaning' (str), 'parent_id' (int).
    # Top-level groups have parent_id=0; sub-groups reference their parent's coding.
    # Used to expand DESIRED_ETHNICITY to include all sub-groups (e.g., ethnicity=1 "White"
    # also selects coding=1001 "British", 1002 "Irish", 1003 "Any other white background").
    # For UK Biobank, it can be downloaded from https://biobank.ndph.ox.ac.uk/ukb/coding.cgi?id=1001
    # Example content:
    #   coding	meaning	node_id	parent_id	selectable
    #   1	White	1	0	Y
    #   2	Mixed	2	0	Y
    #   3	Asian or Asian British	3	0	Y
    #   1001	British	1001	1	Y
    #   1002	Irish	1002	1	Y
    ETHNICITY_CODING_FILE = args.ethnicity_coding_file
    ETHNICITY_FIELD = args.ethnicity_field

    # GWAS column names (adjust to match your GWAS file header)
    GWAS_RSID_COL = args.gwas_rsid_col
    BETA_COLUMN = args.beta_column
    P_VALUE_COLUMN = args.p_value_column
    EFFECT_ALLELE_COLUMN = args.effect_allele_column
    OTHER_ALLELE_COLUM = args.other_allele_column

    # Filtering options
    DESIRED_ETHNICITY = args.desired_ethnicity  # List of ethnicity codes to keep (empty for all)
    TOP_K = args.top_k or None  # Number of top SNPs to select by p-value (None for all)

    # Output directory
    OUTPUT_DIR = args.output_dir or os.path.dirname(os.path.abspath(BED_FILE))

    # Set seed for reproducibility
    RANDOM_SEED = args.seed
    np.random.seed(RANDOM_SEED)

    # 1. Read Genotype Data from BED File

    # Extract output filename from BED filename
    output_filename, extra_info = bed_conv.extract_output_filename(BED_FILE)
    if args.output_name:
        output_filename = args.output_name
    suffix = '_' + args.output_suffix if args.output_suffix else ''
    all_output = args.output or os.path.join(OUTPUT_DIR, f'{output_filename}_ALLSNPS{suffix}.hdf5')
    top_output = args.top_output or (str(Path(args.output).with_suffix('')) + f'_top{TOP_K}.hdf5' if args.output else os.path.join(OUTPUT_DIR, f'{output_filename}{suffix}.hdf5'))
    snp_ids_file = args.snp_ids_output or os.path.join(OUTPUT_DIR, f'snp_ids{extra_info}_TOP{TOP_K}.txt')
    for path in [all_output] + ([top_output, snp_ids_file] if TOP_K is not None else []):
        if os.path.exists(path) and not args.overwrite:
            raise FileExistsError(f'Output exists: {path}; pass --overwrite to replace it')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Output filename: {output_filename}")

    # Load BED file
    G, data, snp_ids, eids, chrom_pos_df = bed_conv.load_bed_file(BED_FILE, count_A1=args.count_a1)
    print(f"Loaded genotype data: {data.shape[0]} samples x {data.shape[1]} SNPs")

    # Verify SNP ordering (optional but recommended)
    if not args.skip_order_check:
        bed_conv.verify_snp_ordering(snp_ids, chrom_pos_df, verbose=True)

    # 2. Convert to DataFrame and Preprocess

    # Convert to pandas DataFrame
    b_df = pd.DataFrame(data, index=eids, columns=snp_ids)
    b_df.index.name = 'sample'
    b_df.columns.name = 'snp'
    print(f"DataFrame shape: {b_df.shape}")

    # Preprocess with genotype_handler
    b_allel = g_handler.dataframe_to_genotype_array(b_df, sample_axis=0, map_elements=True, verbose=True, nan_value=-127)
    b_preproc, kept_variants = g_handler.base_preprocessing(b_allel, save_allele_counts=False, prune_ld=False, seed=RANDOM_SEED)
    b_df_preproc = pd.DataFrame(b_preproc, index=b_df.index, columns=b_df.columns[kept_variants]).astype('i1')
    print(f"Preprocessed DataFrame shape: {b_df_preproc.shape}")

    # 3. Load Phenotype Data

    pheno_df = bed_conv.load_phenotype(PHENO_FILE)
    print(f"Loaded phenotype data for {len(pheno_df)} samples")
    print(pheno_df.head())

    # 4. Filter by Ethnicity

    if ANCESTRY_FILE:
        if not ETHNICITY_CODING_FILE:
            raise ValueError('--ethnicity-coding-file is required with --ancestry-file')
        pheno_df, ancestry_df = bed_conv.filter_by_ethnicity(
            pheno_df,
            DESIRED_ETHNICITY,
            ANCESTRY_FILE,
            ETHNICITY_CODING_FILE,
            ETHNICITY_FIELD
        )
    else:
        if DESIRED_ETHNICITY:
            raise ValueError('--desired-ethnicity requires ancestry and coding files')
        ancestry_df = pd.DataFrame({ETHNICITY_FIELD: np.full(len(pheno_df), -1, dtype=np.int32)}, index=pheno_df.index)


    # Join genotype and phenotype data
    _prev_n = len(b_df_preproc)
    print(f"Samples before merging with phenotype: {_prev_n}")

    out_df = b_df_preproc.join(pheno_df, how='inner')
    print(f"Samples after merging with phenotype: {len(out_df)}")
    print(f"Samples removed: {_prev_n - len(out_df)}")

    # 5. Load GWAS Summary Statistics

    effects_df = bed_conv.load_gwas(
        GWAS_FILE,
        rsid_col=GWAS_RSID_COL,
        beta_col=BETA_COLUMN,
        p_value_col=P_VALUE_COLUMN,
        effect_allele_col=EFFECT_ALLELE_COLUMN
    )
    print(f"Loaded GWAS data for {len(effects_df)} variants")
    print(effects_df.head())

    # 5.1. Remove missing SNPs

    # Get SNP columns (exclude phenotype columns)
    snp_cols_mask = ~out_df.columns.str.contains('pheno')
    all_snp_ids = out_df.columns[snp_cols_mask]

    # Check which SNPs are present in the GWAS dataframe using pandas Index (hash-based lookup)
    snps_in_gwas = all_snp_ids.isin(effects_df.index)
    n_removed = (~snps_in_gwas).sum()
    n_kept = snps_in_gwas.sum()

    print(f"SNPs before filtering: {len(all_snp_ids)}")
    print(f"SNPs removed (not in GWAS): {n_removed}")
    print(f"SNPs kept: {n_kept}")

    # Filter to keep only SNPs in GWAS
    final_snp_ids = all_snp_ids[snps_in_gwas].to_numpy()

    # Handle duplicate SNP IDs in GWAS - keep the one whose alleles match the BED file
    # Create a lookup for BED alleles
    bed_alleles = {sid: (a1, a2) for sid, a1, a2 in zip(G.sid, G.allele_1, G.allele_2)}

    # Deduplicate GWAS entries using the utility function
    relevant_gwas = bed_conv.deduplicate_gwas_by_alleles(
        effects_df,
        final_snp_ids,
        bed_alleles,
        EFFECT_ALLELE_COLUMN,
        P_VALUE_COLUMN,
        baseline_allele_col=OTHER_ALLELE_COLUM,
        verbose=True
    )

    # Re-impose BED order after deduplication
    final_snp_ids = all_snp_ids[all_snp_ids.isin(relevant_gwas.index)].to_numpy()
    relevant_gwas = relevant_gwas.reindex(final_snp_ids)
    assert relevant_gwas.index.is_unique, "GWAS index still has duplicates"

    # Update the column mask to only include SNPs that are in GWAS
    snp_cols_mask = out_df.columns.isin(final_snp_ids)

    # Get alleles from the BED file for the final SNPs in the same order
    snp_id_to_idx = {snp: i for i, snp in enumerate(snp_ids)}
    final_snp_indices = np.array([snp_id_to_idx[snp] for snp in final_snp_ids], dtype=int)

    # 6. Align Alleles and Flip Betas

    bed_allele_1 = np.array(G.allele_1)[final_snp_indices]
    bed_allele_2 = np.array(G.allele_2)[final_snp_indices]

    # Align alleles and flip betas where needed
    aligned_betas, flip_mask = bed_conv.align_alleles_and_flip_betas(
        final_snp_ids,
        bed_allele_1,
        bed_allele_2,
        relevant_gwas,
        EFFECT_ALLELE_COLUMN,
        BETA_COLUMN,
        verbose=True
    )

    # 7. Save Data to HDF5

    # The alignment helper returns A1-oriented betas.
    # Negate its result for A2-counted dosages, so effects
    # refer to the allele encoded in the genotype matrix (see docs/cli.md).
    if not args.count_a1:
        aligned_betas = -aligned_betas
        flip_mask = ~flip_mask

    # Prepare data for saving
    genotype_data = out_df.loc[:, final_snp_ids].to_numpy()
    labels = out_df['phenotype'].to_numpy()

    # Get p-values for final SNPs
    p_values = relevant_gwas[P_VALUE_COLUMN].astype(float).to_numpy()

    # Get sample info
    final_eids = out_df.index.to_numpy()
    final_ancestry = ancestry_df.loc[final_eids, ETHNICITY_FIELD].to_numpy()

    # Get chromosome and position for final SNPs
    final_chrom = chrom_pos_df.loc[final_snp_ids, 'chrom'].to_numpy()
    final_pos = chrom_pos_df.loc[final_snp_ids, 'pos'].to_numpy()

    assert genotype_data.shape[1] == len(final_snp_ids), "Number of SNPs in genotype data does not match number of final SNP IDs"

    auc = roc_auc_score(labels, np.dot(genotype_data, aligned_betas))
    if auc < 0.5:
        auc = 1 - auc

    print(f"""
    Data shape: {genotype_data.shape}
    Labels shape: {labels.shape}
    SNP IDs shape: {final_snp_ids.shape}
    Betas shape: {aligned_betas.shape}
    EIDs shape: {final_eids.shape}
    Ancestry shape: {final_ancestry.shape}
    Chrom shape: {final_chrom.shape}
    Pos shape: {final_pos.shape}
    Allele 1 shape: {bed_allele_1.shape}
    Allele 2 shape: {bed_allele_2.shape}
    AUC: {auc}
    """)

    # Save ALL SNPs version
    bed_conv.save_to_hdf5(
        filepath=all_output,
        data=genotype_data,
        labels=labels,
        snp_ids=final_snp_ids,
        betas=aligned_betas,
        p_values=p_values,
        eids=final_eids,
        ancestry=final_ancestry,
        chrom=final_chrom,
        pos=final_pos,
        allele_1=bed_allele_1,
        allele_2=bed_allele_2,
        beta_flipped=flip_mask
    )

    # Select top K SNPs by p-value (if specified)
    if TOP_K is not None and TOP_K <= len(final_snp_ids):
        (
            data_topk, snp_ids_topk, betas_topk, p_values_topk,
            chrom_topk, pos_topk, allele_1_topk, allele_2_topk,
            flip_mask_topk, _
        ) = bed_conv.select_top_k_snps(
            TOP_K, genotype_data, final_snp_ids, aligned_betas, p_values,
            final_chrom, final_pos, bed_allele_1, bed_allele_2, flip_mask
        )

        auc_topk = roc_auc_score(labels, np.dot(data_topk, betas_topk))
        if auc_topk < 0.5:
            auc_topk = 1 - auc_topk

        print(f"""
        TOP K = {TOP_K}
        Data shape: {data_topk.shape}
        SNP IDs shape: {snp_ids_topk.shape}
        Betas shape: {betas_topk.shape}
        AUC: {auc_topk}
        """)

        # Save top K version
        bed_conv.save_to_hdf5(
            filepath=top_output,
            data=data_topk,
            labels=labels,
            snp_ids=snp_ids_topk,
            betas=betas_topk,
            p_values=p_values_topk,
            eids=final_eids,
            ancestry=final_ancestry,
            chrom=chrom_topk,
            pos=pos_topk,
            allele_1=allele_1_topk,
            allele_2=allele_2_topk,
            beta_flipped=flip_mask_topk
        )

        # Save SNP IDs to text file
        with open(snp_ids_file, 'w') as f:
            for snp_id in snp_ids_topk.tolist():
                f.write(f"{snp_id}\n")
        print(f"SNP IDs saved to {snp_ids_file}")

    else:
        print("TOP_K not specified or greater than total SNPs; skipping top K selection.")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
