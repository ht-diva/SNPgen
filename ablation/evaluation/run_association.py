"""Fit and compare per-SNP association effects for saved synthetic cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ablation.evaluation.cohort_protocol import (
    iter_selected_sample_chunks,
    parse_cohort_specs,
    prepare_matched_cohort_selections,
)
from snpgen.evaluation.association import (
    LogisticFitOptions,
    compare_association_effects,
    fit_binary_associations_from_counts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--reference-config", required=True, help="Training config that identifies the real HDF5 and split seed")
    parser.add_argument("--cohort", action="append", default=[], metavar="NAME=HDF5", help="Repeat for every CFG/model cohort")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-split", choices=("train_val", "test", "full"), default="train_val")
    parser.add_argument("--seed", type=int, default=None, help="Defaults to the seed in the reference config")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional common total sample cap")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--chunk-size", type=int, default=8192, help="HDF5 rows aggregated at a time")
    parser.add_argument("--metadata-beta-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cohort_paths = parse_cohort_specs(args.cohort)
    selections, provenance = prepare_matched_cohort_selections(
        args.reference_config,
        cohort_paths,
        reference_split=args.reference_split,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    options = LogisticFitOptions(max_iter=args.max_iter, tolerance=args.tolerance)
    metadata = provenance["metadata"]

    fitted: dict[str, pd.DataFrame] = {}
    all_rows = []
    cohort_summary = []
    for name, selection in selections.items():
        totals = None
        cases = None
        offset = 0
        for samples in iter_selected_sample_chunks(selection.path, selection.indices, args.chunk_size):
            labels = selection.labels[offset : offset + samples.shape[0]]
            case_samples = samples[labels == 1]
            chunk_totals = np.stack([(samples == dosage).sum(axis=0) for dosage in (0, 1, 2)], axis=1)
            chunk_cases = np.stack([(case_samples == dosage).sum(axis=0) for dosage in (0, 1, 2)], axis=1)
            totals = chunk_totals if totals is None else totals + chunk_totals
            cases = chunk_cases if cases is None else cases + chunk_cases
            offset += samples.shape[0]
        if offset != selection.labels.size:
            raise RuntimeError(f"Read {offset} rows for {name}, expected {selection.labels.size}")
        table = fit_binary_associations_from_counts(totals, cases, options)
        table.insert(0, "cohort", name)
        table.insert(0, "trait", args.trait)
        table["snp_id"] = np.asarray(metadata["snp_id"]).astype(str)
        table["chrom"] = np.asarray(metadata["chrom"])
        table["pos"] = np.asarray(metadata["pos"])
        if "metadata_beta" in metadata:
            table["metadata_beta"] = args.metadata_beta_sign * np.asarray(metadata["metadata_beta"], dtype=float)
        fitted[name] = table
        all_rows.append(table)
        cohort_summary.append(
            {
                "trait": args.trait,
                "cohort": name,
                "n_samples": selection.labels.size,
                "n_controls": int(np.sum(selection.labels == 0)),
                "n_cases": int(np.sum(selection.labels == 1)),
                "n_snps": totals.shape[0],
                "n_converged": int(np.sum(table.fit_status == "ok")),
                "n_monomorphic": int(np.sum(table.fit_status == "monomorphic")),
                "n_separated_or_failed": int(np.sum(~table.fit_status.isin(["ok", "monomorphic"]))),
                "discoveries_bh": int(np.sum(table.q_value <= args.alpha)),
            }
        )

    comparisons = []
    for name in cohort_paths:
        row = compare_association_effects(fitted["real"], fitted[name], alpha=args.alpha)
        row.update({"trait": args.trait, "reference": "real", "cohort": name, "alpha": args.alpha})
        comparisons.append(row)

    pd.concat(all_rows, ignore_index=True).to_csv(output_dir / "per_variant.csv.gz", index=False)
    pd.DataFrame(cohort_summary).to_csv(output_dir / "cohort_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output_dir / "comparison_summary.csv", index=False)
    manifest = provenance["manifest"]
    manifest.update(
        {
            "protocol_version": "association_v1",
            "trait": args.trait,
            "model": "unadjusted per-SNP additive logistic regression with intercept",
            "multiple_testing": "Benjamini-Hochberg",
            "hdf5_chunk_size": args.chunk_size,
            "alpha": args.alpha,
            "metadata_beta_sign": args.metadata_beta_sign,
            "discovery_metric_note": "power_vs_reference and fdr_vs_reference are empirical replication/discordance proxies against fitted real-cohort discoveries, not formal power or false-discovery rates against causal truth.",
            "covariate_note": "No age, sex, ancestry-PC, or other sample covariates are available in the configured HDF5 files; these fits are deliberately unadjusted.",
        }
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Association results written to {output_dir}")


if __name__ == "__main__":
    main()
