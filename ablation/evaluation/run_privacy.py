"""Run privacy evaluation for one ablation checkpoint directory."""

from __future__ import annotations

import argparse
import os

import torch
from omegaconf import OmegaConf

from ablation.utils import resolve_checkpoint_dir_with_config
from snpgen.evaluation.privacy import PrivacyEvaluator, load_privacy_data_manual

OmegaConf.register_new_resolver("eval", eval, replace=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint-dir")
    source.add_argument(
        "--reference-config",
        help="Config defining the real dataset/splits when evaluating an explicit --syn-path.",
    )
    parser.add_argument("--syn-path", help="Explicit generated HDF5 path (requires --reference-config).")
    parser.add_argument("--output-dir", help="Override synthetic privacy output directory.")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--distance", default="hamming", choices=["hamming", "manhattan"])
    parser.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    parser.add_argument("--per-class", action="store_true", default=True)
    parser.add_argument("--no-per-class", dest="per_class", action="store_false")
    parser.add_argument("--evaluate-synthetic", action="store_true", default=True)
    parser.add_argument("--skip-synthetic", dest="evaluate_synthetic", action="store_false")
    parser.add_argument("--evaluate-reconstructed", action="store_true", default=False)
    parser.add_argument("--syn-filename", default="syn_complete_dataset.hdf5")
    parser.add_argument("--recon-filename", default="vae_reconstruction_dataset_train_val.hdf5")
    parser.add_argument("--knn-batch-size", type=int, default=None)
    parser.add_argument("--nnaa-n-samples", type=int, default=None)
    parser.add_argument("--matched-n-samples", type=int, default=50000)
    parser.add_argument("--yelmen-distance", choices=["hamming", "manhattan"], default="manhattan")
    parser.add_argument("--skip-yelmen", dest="run_yelmen", action="store_false")
    parser.add_argument("--yelmen-per-class", action="store_true")
    parser.add_argument("--skip-chains", dest="run_chains", action="store_false")
    parser.add_argument("--chains-per-class", action="store_true")
    parser.add_argument("--chain-neighbor-k", type=int, default=32)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--no-cache-knn", dest="cache_knn", action="store_false")
    parser.set_defaults(cache_knn=True, run_yelmen=True, run_chains=True)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", dest="verbose", action="store_false")
    args = parser.parse_args()
    if args.device == "gpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device gpu was requested but CUDA is unavailable; refusing a silent "
            "CPU/subsampled privacy evaluation."
        )
    if args.reference_config:
        if not args.syn_path or not args.output_dir:
            parser.error("--reference-config requires both --syn-path and --output-dir")
        if args.evaluate_reconstructed:
            parser.error("--evaluate-reconstructed is only supported with --checkpoint-dir")
        config_path = args.reference_config
        checkpoint_dir = os.path.dirname(os.path.abspath(args.syn_path))
        syn_filename = os.path.basename(args.syn_path)
    else:
        if args.syn_path:
            parser.error("--syn-path requires --reference-config")
        checkpoint_dir = resolve_checkpoint_dir_with_config(args.checkpoint_dir)
        config_path = os.path.join(checkpoint_dir, "config.yaml")
        syn_filename = args.syn_filename
    assert os.path.exists(config_path), f"config.yaml not found at {config_path}"
    config = OmegaConf.load(config_path)

    dataset_path = config["dataset_path"]
    seed = int(config.get("seed", 42))
    raw_params = config.data.raw_dataset.get("params", {})
    val_ratio = float(raw_params.get("val_ratio", 0.2))
    test_ratio = float(raw_params.get("test_ratio", 0.1))
    model_name = args.model_name or os.path.basename(os.path.normpath(checkpoint_dir))

    vae_checkpoint_dir = checkpoint_dir if args.evaluate_reconstructed else ""
    bundle = load_privacy_data_manual(
        dataset_path=dataset_path,
        ddpm_checkpoint_dir=checkpoint_dir,
        vae_checkpoint_dir=vae_checkpoint_dir,
        model_name=model_name,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        syn_filename=syn_filename,
        recon_filename=args.recon_filename,
        verbose=args.verbose,
    )
    if args.verbose:
        bundle.summary()

    evaluator = PrivacyEvaluator(
        distance=args.distance,
        per_class=args.per_class,
        device=args.device,
        nnaa_n_samples=args.nnaa_n_samples,
        verbose=args.verbose,
        cache_knn=args.cache_knn,
        batch_size=args.knn_batch_size,
        yelmen_distance=args.yelmen_distance,
        matched_n_samples=args.matched_n_samples,
        run_yelmen=args.run_yelmen,
        yelmen_per_class=args.yelmen_per_class,
        run_chains=args.run_chains,
        chains_per_class=args.chains_per_class,
        chain_neighbor_k=args.chain_neighbor_k,
        seed=seed,
        force_recompute=args.force_recompute,
    )

    if args.evaluate_synthetic:
        output_dir = args.output_dir or os.path.join(checkpoint_dir, "privacy_results", "synthetic")
        evaluator.evaluate(bundle, output_dir=output_dir, eval_target="synthetic")

    if args.evaluate_reconstructed:
        if bundle.reconstructed is None:
            print("No reconstructed data available, skipping privacy reconstruction target.")
        else:
            output_dir = os.path.join(checkpoint_dir, "privacy_results", "reconstructed")
            evaluator.evaluate(bundle, output_dir=output_dir, eval_target="reconstructed")


if __name__ == "__main__":
    main()
