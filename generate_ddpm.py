#!/usr/bin/env python3
"""Generate synthetic data using the genotype-processing pipeline.

Configuration values come from command-line arguments.
"""

import argparse
import os
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description='Generate synthetic cohorts with the phenotype-conditioned latent diffusion.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint', required=True, help='Explicit DDPM checkpoint filename.')
    parser.add_argument('--config', help='Default: config.yaml beside checkpoint.')
    parser.add_argument('--dataset-path', help='Override the saved real-data HDF5 path.')
    parser.add_argument('--modes', '--syn-dataset-type', nargs='+', choices=['complete','matched','augmented','syn_recon'], default=['complete','augmented'])
    parser.add_argument('--batch-size', type=int, default=6144)
    parser.add_argument('--num-workers', '--workers', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda or a CUDA device identifier.')
    scales=parser.add_mutually_exclusive_group()
    scales.add_argument('--cfg-scale', type=float)
    scales.add_argument('--cfg-scales', nargs='+', type=float)
    parser.add_argument('--sampling-steps', type=int, help='Override the saved sampler step count.')
    parser.add_argument('--no-ema', action='store_true')
    parser.add_argument('--output-dir', help='Default: checkpoint directory (CFG sweeps create one subdirectory per scale).')
    existing=parser.add_mutually_exclusive_group()
    existing.add_argument('--overwrite', action='store_true')
    existing.add_argument('--skip-existing', action='store_true')
    return parser



def run(args):
    if args.batch_size < 1 or (args.num_workers is not None and args.num_workers < 0):
        raise ValueError('Batch size must be positive and worker count non-negative')
    import torch

    import numpy as np
    import h5py
    import os
    import glob
    import re
    from tqdm import tqdm
    import pickle
    from omegaconf import OmegaConf

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from lightning.pytorch import seed_everything

    from snpgen.utils import instantiate_from_config
    from snpgen.models.modules.utils import load_ddpm_model


    OmegaConf.register_new_resolver("eval", eval, replace=True)

    plt.rcParams['figure.dpi'] = 200 # increase show resoultion

    # Run Settings

    # ============================================================
    # RUN SETTINGS - configured with CLI arguments
    # ============================================================

    # Path to the DDPM checkpoint to load
    reload_path = args.checkpoint

    syn_dataset_type = ['complete' if mode == 'matched' else mode for mode in args.modes]

    # Sampling batch size for generation
    batch_size = args.batch_size

    seed = args.seed
    seed_everything(seed, workers=True)

    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)

    NUM_WORKERS = args.num_workers if args.num_workers is not None else int(os.environ.get('SLURM_CPUS_PER_TASK', '0'))
    NUM_NODES = int(os.environ.get('SLURM_NNODES', '1'))
    ALLOCATED_GPUS_PER_NODE = int(os.environ.get('SLURM_GPUS_ON_NODE', '0'))
    SLURM_JOBID = os.environ.get('SLURM_JOB_ID', 'local')

    num_gpus = torch.cuda.device_count()
    print(f"{num_gpus} GPU(s) available")
    print(f"Using {NUM_WORKERS} workers for the DataLoader")

    # Load Config

    config_path = args.config or os.path.join(os.path.dirname(reload_path), 'config.yaml')

    if os.path.exists(config_path):
        # Try to load config directly from the checkpoint directory
        print(f"Loading config from checkpoint directory: {config_path}")
        config = OmegaConf.load(config_path)
    else:
        raise FileNotFoundError(
            f"config.yaml not found in checkpoint directory: {os.path.dirname(reload_path)}\n"
            "Please ensure the checkpoint directory contains a config.yaml file."
        )

    if args.dataset_path:
        config.dataset_path = args.dataset_path
    if args.cfg_scale is not None:
        config.model.params.sampler_config.params.guider_config.params.scale = args.cfg_scale
    if args.sampling_steps is not None:
        config.model.params.sampler_config.params.num_steps = args.sampling_steps
    # The sampler is not a torch Module; its device must be configured explicitly.
    config.model.params.sampler_config.params.device = str(device)
    resolved_config_dict = OmegaConf.to_container(config, resolve=True)
    config_orig = config.copy() # keep a backup of the original config prior to any change

    encoder_config = OmegaConf.to_container(config.model.params.first_stage_config.params.encoder_config.params, resolve=True)
    decoder_config = OmegaConf.to_container(config.model.params.first_stage_config.params.decoder_config.params, resolve=True)

    # Build Dataset

    if 'dataset_path' in config:
        h5_path = config['dataset_path']
        print(f"Using dataset path from reloaded config: {h5_path}")
        proj_name = os.path.basename(os.path.dirname(config['dataset_path'])).replace('ukb_', '')
        print(f"Inferred project name: {proj_name}")
    else:
        raise ValueError("dataset_path not found in config. Please ensure the config.yaml contains a dataset_path entry with the appropriate path to the dataset.")

    print(f"Loading Dataset from: {h5_path}")
    if config.get('data', {}).get('raw_dataset', None):
        split_seed = int(config.get('dataset_split_seed', config.data.raw_dataset.params.get('seed', config.seed)))
        if split_seed != seed:
            print(f"Overriding raw_dataset seed from {seed} to {split_seed}")
        raw_dataset = instantiate_from_config(config.data.raw_dataset, file_path=h5_path, seed=split_seed)
    else:
        raise ValueError("raw_dataset config not found. Please ensure the config.yaml contains a data.raw_dataset section with the appropriate dataset configuration.")

    if config.get('data', {}).get('dataset', None):
        train_dataset = instantiate_from_config(config.data.dataset, raw_dataset.get_split('train'), block_ids=raw_dataset.get_metadata('full', 'block_id') if hasattr(raw_dataset, 'get_metadata') else None)
        val_dataset = instantiate_from_config(config.data.dataset, raw_dataset.get_split('val'), block_ids=raw_dataset.get_metadata('full', 'block_id') if hasattr(raw_dataset, 'get_metadata') else None)
        test_dataset = instantiate_from_config(config.data.dataset, raw_dataset.get_split('test'), block_ids=raw_dataset.get_metadata('full', 'block_id') if hasattr(raw_dataset, 'get_metadata') else None)

        complete_dataset = instantiate_from_config(config.data.dataset, raw_dataset.get_split('full'), block_ids=raw_dataset.get_metadata('full', 'block_id') if hasattr(raw_dataset, 'get_metadata') else None)

    else:
        raise ValueError("dataset config not found. Please ensure the config.yaml contains a data.dataset section with the appropriate dataset configuration.")

    # Build Models

    ddpm_use_ema = not args.no_ema  # whether to use the EMA weights at inference

    model_config = OmegaConf.create(OmegaConf.to_container(config.model, resolve=True))
    model_config.params.first_stage_config.params.ckpt_path = None
    ddpm_training_wrapper = load_ddpm_model(
        model_config,
        reload_path,
        device=device,
        ema=ddpm_use_ema,
    )

    # Setup DataLoaders

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=NUM_WORKERS > 0,
        #sampler=ImbalancedDatasetSampler(train_dataset, strategy='inverse_freq'), # balance dataset on labels (which also implicitly performs shuffling)
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=NUM_WORKERS > 0,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=NUM_WORKERS > 0,
    )

    complete_dataloader = torch.utils.data.DataLoader(
        complete_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=NUM_WORKERS > 0,
    )

    # Generate samples, compute reconstructions (only once!)
    # Needed only once, then we can save them for future analysis

    from snpgen.inference import (
        SyntheticDatasetGenerator,
        get_sampler,
        save_synthetic_dataset,
        get_output_filename,
        dataset_exists,
    )

    if not isinstance(syn_dataset_type, list):
        syn_dataset_type = [syn_dataset_type]

    assert all(mode in ['complete', 'syn_recon', 'augmented'] for mode in syn_dataset_type), "Invalid choice in syn_dataset_type"

    # ===================================================================================
    # SETUP GENERATOR
    # ===================================================================================

    # Create generator
    generator = SyntheticDatasetGenerator(
        model=ddpm_training_wrapper,
        config=config,
        decoder_config=decoder_config,
        device=str(device)
    )
    generator.prepare_model()

    # ===================================================================================
    # GENERATE DATASET
    # ===================================================================================

    for mode in syn_dataset_type:

        print(f"\n=== Generating dataset of type: {mode} ===")

        output_dir = args.output_dir or os.path.dirname(reload_path)
        os.makedirs(output_dir, exist_ok=True)
        saved_config = os.path.join(output_dir, 'config.yaml')
        if os.path.abspath(output_dir) != os.path.abspath(os.path.dirname(reload_path)):
            if os.path.exists(saved_config) and not args.overwrite:
                existing = OmegaConf.to_container(OmegaConf.load(saved_config), resolve=True)
                if existing != OmegaConf.to_container(config, resolve=True):
                    raise FileExistsError(f'Output config differs: {saved_config}; pass --overwrite')
            else:
                OmegaConf.save(config, saved_config)
        output_filename = get_output_filename(
            base_name='syn',
            mode=mode,
        )
        output_path = os.path.join(output_dir, output_filename)

        if dataset_exists(output_dir, output_filename) and not (args.overwrite or args.skip_existing):
            raise FileExistsError(f'Output exists: {output_path}; pass --skip-existing or --overwrite')
        if args.overwrite or not dataset_exists(output_dir, output_filename):

            if mode == 'complete':
                # Generate with same labels as original dataset
                print(f"Generating complete dataset...")
                result = generator.generate_complete(
                    complete_dataloader,
                )

                save_synthetic_dataset(
                    result=result,
                    output_path=output_path,
                    mode='complete',
                )

            elif mode == 'syn_recon':
                # Generate with reconstructions and latent space info
                print(f"Generating syn_recon dataset...")
                onehot = config.data.raw_dataset.params.get('onehot', True)
                result = generator.generate_syn_recon(
                    val_dataloader,
                    onehot=onehot,
                )

                save_synthetic_dataset(
                    result=result,
                    output_path=output_path,
                    mode='syn_recon',
                )

            elif mode == 'augmented':
                # Generate with augmented label distribution (binary balanced)
                print(f"Generating augmented dataset...")

                # Get original labels for reference
                original_labels = complete_dataset.targets

                # Binary: generate balanced dataset with 2 * num_controls samples
                num_controls = (original_labels == 0).sum()
                n_samples = 2 * num_controls

                print(f"Original dataset size: {len(original_labels)}")
                print(f"Augmented dataset size: {n_samples}")

                sampler = get_sampler(
                    strategy='binary_balanced',
                    original_labels=original_labels,
                    n_classes=2
                )

                # Generate samples
                result = generator.generate_augmented(
                    label_sampler=sampler,
                    total_samples=n_samples,
                    batch_size=batch_size,
                    seq_len=config.seq_len,
                )

                save_synthetic_dataset(
                    result=result,
                    output_path=output_path,
                    mode='augmented',
                    augmentation_strategy=sampler.name
                )

            print(f"\nSaved dataset to: {output_path}")
            print(f"Generated {len(result.samples)} samples")

        else:
            print(f"Dataset already exists: {output_path}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.cfg_scales:
            if len(set(args.cfg_scales)) != len(args.cfg_scales):
                raise ValueError('CFG scales must be unique')
            for scale in args.cfg_scales:
                sweep_args = argparse.Namespace(**vars(args))
                sweep_args.cfg_scale = scale
                sweep_args.output_dir = str(Path(args.output_dir or Path(args.checkpoint).parent) / f'cfg_scale_{scale:g}'.replace('.', 'p'))
                run(sweep_args)
        else:
            run(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
