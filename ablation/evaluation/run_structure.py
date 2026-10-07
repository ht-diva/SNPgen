"""Run matched cross-model genotype-structure diagnostics on saved cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ablation.evaluation.cohort_protocol import load_matched_cohorts, parse_cohort_specs
from snpgen.evaluation.genotype_structure import (
    allele_frequencies,
    compare_ld,
    fit_reference_pca,
    hwe_statistics,
    ld_decay_table,
    pairwise_ld_r2,
    pca_comparison,
    transform_reference_pca,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--reference-config", required=True)
    parser.add_argument("--cohort", action="append", default=[], metavar="NAME=HDF5")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-split", choices=("train_val", "test", "full"), default="test")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Optional common sample cap; by default use every class-matched "
            "sample available in the held-out reference split"
        ),
    )
    parser.add_argument("--pca-components", type=int, default=10)
    parser.add_argument("--ld-bins", type=int, default=50)
    parser.add_argument("--rare-thresholds", nargs="+", type=float, default=(0.005, 0.01, 0.05))
    return parser


def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return np.nan
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cohort_paths = parse_cohort_specs(args.cohort)
    cohorts, provenance = load_matched_cohorts(
        args.reference_config,
        cohort_paths,
        reference_split=args.reference_split,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = provenance["metadata"]
    n_snps = cohorts["real"].samples.shape[1]

    frequencies: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    hwe_tables: dict[tuple[str, str], pd.DataFrame] = {}
    per_snp_rows = []
    for name, cohort in cohorts.items():
        af, maf = allele_frequencies(cohort.samples)
        frequencies[name] = (af, maf)
        for scope, mask in (("pooled", np.ones(cohort.labels.size, dtype=bool)), ("controls", cohort.labels == 0)):
            hwe = hwe_statistics(cohort.samples[mask])
            hwe_tables[(name, scope)] = hwe
            table = hwe.copy()
            table.insert(0, "hwe_scope", scope)
            table.insert(0, "cohort", name)
            table.insert(0, "trait", args.trait)
            table["snp_index"] = np.arange(n_snps)
            table["snp_id"] = np.asarray(metadata["snp_id"]).astype(str)
            table["chrom"] = np.asarray(metadata["chrom"])
            table["pos"] = np.asarray(metadata["pos"])
            table["frequency_scope"] = "pooled"
            table["pooled_af"] = af
            table["pooled_maf"] = maf
            table["real_pooled_af"] = frequencies["real"][0] if "real" in frequencies else af
            table["real_pooled_maf"] = frequencies["real"][1] if "real" in frequencies else maf
            table["delta_pooled_af"] = table.pooled_af - table.real_pooled_af
            table["delta_pooled_maf"] = table.pooled_maf - table.real_pooled_maf
            per_snp_rows.append(table)

    pca_state, real_scores = fit_reference_pca(cohorts["real"].samples, args.pca_components, provenance["manifest"]["seed"])
    scores = {"real": real_scores}
    for name, cohort in cohorts.items():
        if name != "real":
            scores[name] = transform_reference_pca(cohort.samples, pca_state)

    ld_matrices = {name: pairwise_ld_r2(cohort.samples) for name, cohort in cohorts.items()}
    decay = ld_decay_table(ld_matrices, metadata["chrom"], metadata["pos"], n_bins=args.ld_bins)
    if not decay.empty:
        decay.insert(0, "trait", args.trait)

    real_af, real_maf = frequencies["real"]
    summaries = []
    pca_rows = []
    for name, cohort in cohorts.items():
        af, maf = frequencies[name]
        row = {
            "trait": args.trait,
            "cohort": name,
            "n_samples": cohort.samples.shape[0],
            "n_controls": int(np.sum(cohort.labels == 0)),
            "n_cases": int(np.sum(cohort.labels == 1)),
            "n_snps": n_snps,
            "af_mae": float(np.mean(np.abs(af - real_af))),
            "af_bias": float(np.mean(af - real_af)),
            "af_pearson_r": _safe_correlation(real_af, af),
            "maf_mae": float(np.mean(np.abs(maf - real_maf))),
            "monomorphic_count": int(np.sum(maf == 0)),
            "singleton_snp_count": int(np.sum(np.isclose(2.0 * cohort.samples.shape[0] * maf, 1.0))),
            "doubleton_snp_count": int(np.sum(np.isclose(2.0 * cohort.samples.shape[0] * maf, 2.0))),
        }
        for threshold in args.rare_thresholds:
            key = str(threshold).replace(".", "p")
            real_rare = (real_maf > 0) & (real_maf <= threshold)
            row[f"real_rare_count_le_{key}"] = int(real_rare.sum())
            row[f"rare_retained_polymorphic_fraction_le_{key}"] = float(np.mean(maf[real_rare] > 0)) if real_rare.any() else np.nan
            row[f"rare_remains_rare_fraction_le_{key}"] = float(np.mean((maf[real_rare] > 0) & (maf[real_rare] <= threshold))) if real_rare.any() else np.nan
            row[f"af_mae_real_rare_le_{key}"] = float(np.mean(np.abs(af[real_rare] - real_af[real_rare]))) if real_rare.any() else np.nan

        hwe = hwe_tables[(name, "controls")]
        real_hwe = hwe_tables[("real", "controls")]
        valid_f = np.isfinite(hwe.hwe_f) & np.isfinite(real_hwe.hwe_f)
        row["hwe_controls_f_median"] = float(np.nanmedian(hwe.hwe_f))
        row["hwe_controls_f_rmse_vs_real"] = float(np.sqrt(np.mean((hwe.loc[valid_f, "hwe_f"] - real_hwe.loc[valid_f, "hwe_f"]) ** 2))) if valid_f.any() else np.nan
        row["hwe_controls_p_lt_1e_minus_6_fraction"] = float(np.mean(hwe.hwe_exact_p < 1e-6))
        row.update(compare_ld(ld_matrices["real"], ld_matrices[name], metadata["chrom"], metadata["pos"]))
        pca_table, pca_summary = pca_comparison(real_scores, scores[name])
        pca_table.insert(0, "cohort", name)
        pca_table.insert(0, "trait", args.trait)
        pca_rows.append(pca_table)
        row.update(pca_summary)
        for pc in (1, 2):
            if pc <= pca_table.shape[0]:
                pc_row = pca_table.iloc[pc - 1]
                row[f"pc{pc}_wasserstein_over_real_sd"] = pc_row.wasserstein_over_real_sd
                row[f"pc{pc}_outside_real_1_99_fraction"] = pc_row.outside_real_1_99_fraction
        summaries.append(row)

    # Compare distance-binned means after they are calculated with common bins.
    if not decay.empty:
        real_decay = decay.loc[decay.cohort == "real", ["distance_left", "mean_r2"]].rename(columns={"mean_r2": "real_mean_r2"})
        for row in summaries:
            model_decay = decay.loc[decay.cohort == row["cohort"], ["distance_left", "mean_r2"]]
            merged = real_decay.merge(model_decay, on="distance_left")
            valid = np.isfinite(merged.real_mean_r2) & np.isfinite(merged.mean_r2)
            row["ld_decay_mae_vs_real"] = float(np.mean(np.abs(merged.loc[valid, "mean_r2"] - merged.loc[valid, "real_mean_r2"]))) if valid.any() else np.nan
    else:
        for row in summaries:
            row["ld_decay_mae_vs_real"] = np.nan

    pd.concat(per_snp_rows, ignore_index=True).to_csv(output_dir / "per_snp.csv.gz", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "summary.csv", index=False)
    pd.concat(pca_rows, ignore_index=True).to_csv(output_dir / "pca_summary.csv", index=False)
    decay.to_csv(output_dir / "ld_decay.csv", index=False)
    np.savez_compressed(output_dir / "pca_scores.npz", **scores)
    np.savez_compressed(output_dir / "pairwise_ld_r2.npz", **ld_matrices)

    manifest = provenance["manifest"]
    manifest.update(
        {
            "protocol_version": "structure_v1",
            "trait": args.trait,
            "pca_fit": "real reference only; centered and HWE-scaled with real allele frequency",
            "hwe_primary_scope": "controls; pooled values are also present in per_snp.csv.gz",
            "hwe_method": "exact Wigginton recurrence",
            "per_snp_frequency_scope": "pooled cohort frequencies are repeated for each hwe_scope row",
            "ld_metric": "squared Pearson correlation of additive dosages; undefined pairs retained as NaN",
            "ld_distance_bins": args.ld_bins,
            "rare_thresholds": list(args.rare_thresholds),
            "rare_scope_note": "Rare-allele retention applies only to the selected evaluated SNP panel, not genome-wide rare variation.",
        }
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Structure results written to {output_dir}")


if __name__ == "__main__":
    main()
