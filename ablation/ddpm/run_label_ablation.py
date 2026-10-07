"""Evaluate DDPM synthetic utility after in-memory label perturbations."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _checkpoint_dir_from_config_or_dir(path: str) -> str:
    if os.path.basename(path) == "config.yaml":
        return os.path.dirname(path)
    return path


def _run_variant(
    checkpoint_dir: str,
    variant: str,
    label_seed: int,
    models: list[str],
    skip_real_training: bool,
    cpu: bool,
):
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "ablation.evaluation.run_evaluate",
        "--checkpoint-dir",
        checkpoint_dir,
        "--syn-dataset-types",
        "complete",
        "--synthetic-label-strategy",
        variant,
        "--synthetic-label-seed",
        str(label_seed),
        "--models",
        *models,
    ]
    if skip_real_training:
        cmd.append("--skip-real-training")
    if cpu:
        cmd.append("--cpu")
    print(" ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default=None, help="Existing conditional DDPM checkpoint directory.")
    parser.add_argument("--saved-ddpm-config", default=None, help="Path to an existing DDPM run config.yaml.")
    parser.add_argument("--variants", nargs="+", default=["permuted", "prevalence"], choices=["matched", "permuted", "prevalence"])
    parser.add_argument("--label-seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", default=["xgboost", "xgboost_balanced", "catboost", "prs"])
    parser.add_argument("--skip-real-training", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.checkpoint_dir is None and args.saved_ddpm_config is None:
        raise ValueError("Pass --checkpoint-dir or --saved-ddpm-config")

    checkpoint_dir = _checkpoint_dir_from_config_or_dir(args.checkpoint_dir or args.saved_ddpm_config)
    assert os.path.exists(os.path.join(checkpoint_dir, "config.yaml")), f"config.yaml not found in {checkpoint_dir}"
    assert os.path.exists(os.path.join(checkpoint_dir, "syn_complete_dataset.hdf5")), (
        f"syn_complete_dataset.hdf5 not found in {checkpoint_dir}. Run synthetic_analysis.py first."
    )

    for variant in args.variants:
        _run_variant(
            checkpoint_dir=checkpoint_dir,
            variant=variant,
            label_seed=args.label_seed,
            models=args.models,
            skip_real_training=args.skip_real_training,
            cpu=args.cpu,
        )


if __name__ == "__main__":
    main()
