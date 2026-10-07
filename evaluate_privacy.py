#!/usr/bin/env python3
"""Run privacy evaluation using the genotype-processing pipeline.

Configuration values come from command-line arguments.
"""

import argparse
import os
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description='Evaluate privacy with the genotype privacy metrics and diagnostic plots.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint-dir', help='DDPM run directory with config.yaml and generated HDF5 files.')
    parser.add_argument('--config', help='Explicit saved config; use --dataset-path to relocate the input.')
    parser.add_argument('--traits-config', help='YAML mapping of trait names to the trait settings (optional multi-trait mode).')
    parser.add_argument('--checkpoint-base', help='Base directory for relative paths in --traits-config.')
    parser.add_argument('--model-name', default='SNPgen')
    parser.add_argument('--dataset-path')
    parser.add_argument('--syn-path', help='Explicit synthetic HDF5 path.')
    parser.add_argument('--recon-path', help='Explicit reconstruction HDF5 path.')
    parser.add_argument('--vae-checkpoint-dir')
    parser.add_argument('--split-seed', type=int, help='Manual override; normally the saved training split seed.')
    parser.add_argument('--val-ratio', type=float)
    parser.add_argument('--test-ratio', type=float)
    parser.add_argument('--distance', choices=['hamming','manhattan'], default='hamming')
    parser.add_argument('--device', choices=['auto','gpu','cpu'], default='auto')
    parser.add_argument('--no-per-class', action='store_true')
    parser.add_argument('--no-synthetic', action='store_true')
    parser.add_argument('--evaluate-reconstructed', action='store_true', default=True)
    parser.add_argument('--no-reconstructed', dest='evaluate_reconstructed', action='store_false')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nnaa-n-samples', type=int)
    parser.add_argument('--matched-n-samples', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=1024, help='Nearest-neighbour query batch size.')
    parser.add_argument('--no-yelmen', action='store_true')
    parser.add_argument('--yelmen-per-class', action='store_true')
    parser.add_argument('--no-chains', action='store_true')
    parser.add_argument('--chains-per-class', action='store_true')
    parser.add_argument('--chain-min-length', type=int, default=2)
    parser.add_argument('--chain-max-length', type=int, default=5)
    parser.add_argument('--chain-neighbor-k', type=int, default=32)
    parser.add_argument('--force-recompute', action='store_true')
    parser.add_argument('--output-dir', help='Synthetic privacy output; default checkpoint privacy_results directory.')
    parser.add_argument('--recon-output-dir')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--plot-dir')
    return parser

def load_trait_settings(args):
    """Populate the trait settings from explicit CLI paths or YAML."""
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    if args.traits_config:
        traits = OmegaConf.to_container(OmegaConf.load(args.traits_config), resolve=True)
        for entry in traits.values():
            for key in ['ddpm_checkpoint', 'vae_checkpoint', 'dataset_path']:
                if entry.get(key) and not os.path.isabs(entry[key]):
                    entry[key] = os.path.join(args.checkpoint_base or os.path.dirname(args.traits_config), entry[key])
        return traits
    if not args.checkpoint_dir and not args.config:
        raise ValueError('Provide --checkpoint-dir, --config, or --traits-config')
    checkpoint_dir = args.checkpoint_dir or os.path.dirname(os.path.abspath(args.config))
    # Use the automatic loader when no manual settings are needed.
    if not any([args.config, args.dataset_path, args.vae_checkpoint_dir, args.recon_path,
                args.split_seed is not None, args.val_ratio is not None, args.test_ratio is not None]):
        return {args.model_name: {'ddpm_checkpoint': checkpoint_dir}}
    config = OmegaConf.load(args.config or os.path.join(checkpoint_dir, 'config.yaml'))
    raw = config.data.raw_dataset.params
    dataset = args.dataset_path or config.get('dataset_path')
    split_seed = args.split_seed if args.split_seed is not None else int(config.get('dataset_split_seed', raw.get('seed', config.get('seed', 42))))
    vae_path = OmegaConf.select(config, 'model.params.first_stage_config.params.ckpt_path') or ''
    vae_dir = args.vae_checkpoint_dir or (os.path.dirname(os.path.abspath(args.recon_path)) if args.recon_path else os.path.dirname(vae_path))
    return {args.model_name: {
        'manual': True, 'dataset_path': dataset, 'ddpm_checkpoint': checkpoint_dir,
        'vae_checkpoint': vae_dir, 'seed': split_seed,
        'val_ratio': args.val_ratio if args.val_ratio is not None else raw.get('val_ratio', 0.2),
        'test_ratio': args.test_ratio if args.test_ratio is not None else raw.get('test_ratio', 0.1),
    }}


def run(args):
    # Privacy Analysis
    #
    # Evaluates privacy of synthetic and reconstructed SNP data against real training data.
    #
    # **Metrics computed:**
    # 1. **IMR** — Identical Match Rate (exact copy detection)
    # 2. **DCR** — Distance to Closest Record (min distance distribution)
    # 3. **NNAA** — Nearest Neighbor Adversarial Accuracy
    # 4. **MI** — Distance-based Membership Inference (ROC-AUC)
    # 5. **NNDR** — Nearest Neighbor Distance Ratio (copying detection)
    # 6. **MAF** — Allele Frequency Comparison (fidelity sanity check)
    #
    # Results are saved incrementally to checkpoint directories to survive crashes.

    # 1. Setup & Configuration

    import os
    import sys

    # Add project root to path
    PROJECT_ROOT = str(Path(__file__).resolve().parent)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from snpgen.evaluation.privacy import (
        PrivacyEvaluator,
        load_privacy_data_from_checkpoint,
        load_privacy_data_manual,
        plot_dcr_distributions,
        plot_nnaa_summary,
        plot_mi_roc,
        plot_nndr_histogram,
        plot_maf_scatter,
        plot_privacy_summary_table,
    )

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ============================================================
    # RUN SETTINGS — supplied by CLI arguments
    # ============================================================

    # Base directory containing all SNPgen checkpoints.
    # All TRAITS entries below are resolved relative to this path.
    CKPT_BASE = args.checkpoint_base or ''

    # ============================================================
    # EVALUATION SETTINGS
    # ============================================================
    DISTANCE = args.distance
    PER_CLASS = not args.no_per_class  # Compute metrics per label class
    DEVICE = args.device
    EVALUATE_SYNTHETIC = not args.no_synthetic
    EVALUATE_RECONSTRUCTED = args.evaluate_reconstructed

    # ============================================================
    # TRAIT CONFIGURATION
    # ============================================================
    # Define all traits to evaluate. Each entry needs:
    #   - ddpm_checkpoint: path to DDPM checkpoint directory
    # ============================================================

    TRAITS = load_trait_settings(args)

    # 2. Data Loading

    bundles = {}

    for trait_name, cfg in TRAITS.items():
        print(f"\n{'='*60}")
        print(f" Loading: {trait_name}")
        print(f"{'='*60}")

        try:
            if cfg.get("manual", False):
                bundle = load_privacy_data_manual(
                    dataset_path=cfg["dataset_path"],
                    ddpm_checkpoint_dir=cfg["ddpm_checkpoint"],
                    vae_checkpoint_dir=cfg.get("vae_checkpoint", ""),
                    model_name=trait_name,
                    seed=cfg.get("seed", 42),
                    val_ratio=cfg.get("val_ratio", 0.2),
                    test_ratio=cfg.get("test_ratio", 0.1),
                    syn_filename=cfg.get('syn_filename', args.syn_path or 'syn_complete_dataset.hdf5'),
                    recon_filename=cfg.get('recon_filename', args.recon_path or 'vae_reconstruction_dataset_train_val.hdf5'),
                )
            else:
                bundle = load_privacy_data_from_checkpoint(
                    ddpm_checkpoint_dir=cfg["ddpm_checkpoint"],
                    model_name=trait_name,
                    syn_filename=cfg.get('syn_filename', args.syn_path or 'syn_complete_dataset.hdf5'),
                    recon_filename=cfg.get('recon_filename', args.recon_path or 'vae_reconstruction_dataset_train_val.hdf5'),
                )

            bundle.summary()
            bundles[trait_name] = bundle

        except Exception as e:
            print(f"  ERROR loading {trait_name}: {e}")
            continue

    print(f"\nLoaded {len(bundles)}/{len(TRAITS)} traits successfully.")

    # 3. Privacy Evaluation

    if len(bundles) != len(TRAITS):
        raise ValueError('Some requested privacy inputs failed to load; see errors above')

    evaluator = PrivacyEvaluator(
        distance=DISTANCE,
        per_class=PER_CLASS,
        device=DEVICE,
        seed=args.seed,
        matched_n_samples=args.matched_n_samples,
        nnaa_n_samples=args.nnaa_n_samples,
        run_yelmen=not args.no_yelmen,
        yelmen_per_class=args.yelmen_per_class,
        run_chains=not args.no_chains,
        chains_per_class=args.chains_per_class,
        chain_min_length=args.chain_min_length,
        chain_max_length=args.chain_max_length,
        chain_neighbor_k=args.chain_neighbor_k,
        batch_size=args.batch_size,
        force_recompute=args.force_recompute,
    )

    all_results = {}

    for trait_name, bundle in bundles.items():

        # ── Synthetic privacy (saved to DDPM checkpoint dir) ──
        if EVALUATE_SYNTHETIC:
            syn_output_dir = (os.path.join(args.output_dir, trait_name) if args.output_dir and len(TRAITS) > 1 else args.output_dir) or os.path.join(bundle.ddpm_checkpoint_dir, "privacy_results")
            print(f"\n{'#'*60}")
            print(f"# {trait_name} — SYNTHETIC")
            print(f"# Output: {syn_output_dir}")
            print(f"{'#'*60}")

            syn_results = evaluator.evaluate(
                bundle,
                output_dir=syn_output_dir,
                eval_target='synthetic',
            )
            all_results[f"{trait_name}_syn"] = syn_results

        # ── Reconstruction privacy (saved to VAE checkpoint dir) ──
        if EVALUATE_RECONSTRUCTED and bundle.reconstructed is not None:
            recon_output_dir = args.recon_output_dir or (os.path.join(args.output_dir, trait_name + '_reconstructed') if args.output_dir else os.path.join(bundle.vae_checkpoint_dir, "privacy_results"))
            print(f"\n{'#'*60}")
            print(f"# {trait_name} — RECONSTRUCTED")
            print(f"# Output: {recon_output_dir}")
            print(f"{'#'*60}")

            recon_results = evaluator.evaluate(
                bundle,
                output_dir=recon_output_dir,
                eval_target='reconstructed',
            )
            all_results[f"{trait_name}_recon"] = recon_results
        elif EVALUATE_RECONSTRUCTED:
            print(f"\n  {trait_name}: No reconstructed data available, skipping.")

    print(f"\nEvaluation complete. {len(all_results)} result sets computed.")

    # 4. Visualization

    if args.no_plots:
        return
    plot_dir = args.plot_dir or os.path.join(args.output_dir or next(iter(bundles.values())).ddpm_checkpoint_dir, 'privacy_plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_number = 0
    for trait_key, results in all_results.items():
        print(f"\n{'='*60}")
        print(f" {trait_key}")
        print(f"{'='*60}")

        # DCR
        dcr_r = results.get('dcr__overall')
        if dcr_r is not None:
            fig = plot_dcr_distributions(dcr_r, title=f'{trait_key} — DCR Distribution')
            plot_number += 1
            fig.savefig(os.path.join(plot_dir, f'privacy_{plot_number:02d}.png'), dpi=200, bbox_inches='tight')
            plt.close(fig)

        # NNAA
        nnaa_r = results.get('nnaa__overall')
        if nnaa_r is not None:
            fig = plot_nnaa_summary(nnaa_r, title=f'{trait_key} — NNAA')
            plot_number += 1
            fig.savefig(os.path.join(plot_dir, f'privacy_{plot_number:02d}.png'), dpi=200, bbox_inches='tight')
            plt.close(fig)

        # MI
        mi_r = results.get('mi__overall')
        if mi_r is not None:
            fig = plot_mi_roc(mi_r, title=f'{trait_key} — Membership Inference')
            plot_number += 1
            fig.savefig(os.path.join(plot_dir, f'privacy_{plot_number:02d}.png'), dpi=200, bbox_inches='tight')
            plt.close(fig)

        # NNDR
        nndr_r = results.get('nndr__overall')
        if nndr_r is not None:
            fig = plot_nndr_histogram(nndr_r, title=f'{trait_key} — NNDR')
            plot_number += 1
            fig.savefig(os.path.join(plot_dir, f'privacy_{plot_number:02d}.png'), dpi=200, bbox_inches='tight')
            plt.close(fig)

        # MAF
        maf_r = results.get('maf__overall')
        if maf_r is not None:
            fig = plot_maf_scatter(maf_r, title=f'{trait_key} — MAF')
            plot_number += 1
            fig.savefig(os.path.join(plot_dir, f'privacy_{plot_number:02d}.png'), dpi=200, bbox_inches='tight')
            plt.close(fig)

    # Per-class visualizations

    if PER_CLASS:
        for trait_key, results in all_results.items():
            # Dynamically discover per-class suffixes from result keys
            class_keys = sorted([k for k in results if k.startswith('dcr__class_')])
            if not class_keys:
                continue

            for dcr_key in class_keys:
                suffix = dcr_key.replace('dcr__', '')
                label_val = suffix.replace('class_', '')
                label_name = f'Class {label_val}'

                dcr_r = results.get(f'dcr__{suffix}')
                if dcr_r is None:
                    continue

                print(f"\n--- {trait_key} | {label_name} (label={label_val}) ---")

                fig = plot_dcr_distributions(dcr_r, title=f'{trait_key} — DCR ({label_name})')
                plot_number += 1
                fig.savefig(os.path.join(plot_dir, f'privacy_{plot_number:02d}.png'), dpi=200, bbox_inches='tight')
                plt.close(fig)

                nnaa_r = results.get(f'nnaa__{suffix}')
                if nnaa_r is not None:
                    fig = plot_nnaa_summary(nnaa_r, title=f'{trait_key} — NNAA ({label_name})')
                    plot_number += 1
                    fig.savefig(os.path.join(plot_dir, f'privacy_{plot_number:02d}.png'), dpi=200, bbox_inches='tight')
                    plt.close(fig)

    # 5. Cross-Trait Summary

    fig = plot_privacy_summary_table(all_results)
    if fig is not None:
        plot_number += 1
        fig.savefig(os.path.join(plot_dir, f'privacy_{plot_number:02d}.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)


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
