#!/usr/bin/env python3
"""Reconstruct genotypes using the VAE inference pipeline.

Configuration values come from command-line arguments.
"""

import argparse
import os
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description='Generate VAE reconstructions using the genotype-processing pipeline.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', help='Default: config.yaml beside checkpoint.')
    parser.add_argument('--dataset-path', help='Override the saved original dataset path.')
    parser.add_argument('--splits', nargs='+', choices=['train','val','test','train_val','full'], default=['train_val','test'])
    parser.add_argument('--posterior-mean', action='store_true', help='Use deterministic posterior mean; default samples the posterior by default.')
    parser.add_argument('--store-latents', action='store_true')
    parser.add_argument('--store-originals', action='store_true')
    parser.add_argument('--batch-size', type=int, default=768)
    parser.add_argument('--num-workers', '--workers', type=int)
    parser.add_argument('--seed', type=int, default=42, help='Inference RNG seed; source splits use the saved training split seed.')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--no-ema', action='store_true')
    parser.add_argument('--output-dir', help='Default: VAE checkpoint directory.')
    existing=parser.add_mutually_exclusive_group()
    existing.add_argument('--overwrite', action='store_true')
    existing.add_argument('--skip-existing', action='store_true')
    return parser



def run(args):
    import torch
    import torchmetrics

    import numpy as np
    import os
    import glob
    import re
    from tqdm import tqdm
    from omegaconf import OmegaConf

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from lightning.pytorch import seed_everything

    from snpgen.utils import instantiate_from_config
    from snpgen.data.loader import SplitDataset

    # Inference module for VAE reconstructions
    from snpgen.inference import (
        ReconstructionGenerator,
        save_synthetic_dataset,
        get_output_filename,
        dataset_exists,
    )

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    # Run Settings


    # ============================================================
    # RUN SETTINGS - supplied by CLI arguments
    # ============================================================

    # Path to the VAE checkpoint to load
    reload_path = args.checkpoint

    # Source splits to reconstruct
    splits_to_reconstruct = args.splits

    # Whether to sample from the posterior (stochastic) or use mean (deterministic)
    sample_posterior = not args.posterior_mean

    # Whether to store latent space info (mu, logvar, z) - increases memory usage
    store_latents = args.store_latents

    # Whether to store original samples in the output file (for analysis)
    store_originals = args.store_originals

    # Sampling batch size for reconstruction generation
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
    assert os.path.exists(config_path), f"config.yaml not found at {config_path}. This command requires a run with saved config.yaml."
    config = OmegaConf.load(config_path)

    if args.dataset_path:
        config.dataset_path = args.dataset_path
    resolved_config_dict = OmegaConf.to_container(config, resolve=True)
    config_orig = config.copy() # keep a backup of the original config prior to any change

    # Build Dataset

    if 'dataset_path' in config:
        h5_path = config['dataset_path']
    else:
        raise ValueError("dataset_path not found in config. Please ensure the config.yaml contains the dataset_path key pointing to the original dataset used for training.")

    print(h5_path)

    print(f"Loading Dataset from: {h5_path}")
    if config.get('data', {}).get('raw_dataset', None):
        split_seed = int(config.get('dataset_split_seed', config.data.raw_dataset.params.get('seed', config.seed)))
        raw_dataset = instantiate_from_config(config.data.raw_dataset, file_path=h5_path, metadata=True, seed=split_seed)
    else:
        raise ValueError("raw_dataset config not found. Please ensure the config.yaml contains a data.raw_dataset section with the appropriate dataset configuration.")

    if config.get('data', {}).get('dataset', None):
        train_dataset = instantiate_from_config(config.data.dataset, raw_dataset.get_split('train'), block_ids=raw_dataset.get_metadata('full', 'block_id') if hasattr(raw_dataset, 'get_metadata') else None)
        val_dataset = instantiate_from_config(config.data.dataset, raw_dataset.get_split('val'), block_ids=raw_dataset.get_metadata('full', 'block_id') if hasattr(raw_dataset, 'get_metadata') else None)
        test_dataset = instantiate_from_config(config.data.dataset, raw_dataset.get_split('test'), block_ids=raw_dataset.get_metadata('full', 'block_id') if hasattr(raw_dataset, 'get_metadata') else None)

        train_val_dataset = instantiate_from_config(config.data.dataset, raw_dataset.get_split('train_val'), block_ids=raw_dataset.get_metadata('full', 'block_id') if hasattr(raw_dataset, 'get_metadata') else None)
        complete_dataset = instantiate_from_config(config.data.dataset, raw_dataset.get_split('full'), block_ids=raw_dataset.get_metadata('full', 'block_id') if hasattr(raw_dataset, 'get_metadata') else None)

    else:
        raise ValueError("dataset config not found. Please ensure the config.yaml contains a data.dataset section with the appropriate dataset configuration.")

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

    train_val_dataloader = torch.utils.data.DataLoader(
        train_val_dataset,
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

    # Build Models

    config.model.params.autoencoder_config.params.ckpt_path = reload_path
    config.model.params.autoencoder_config.params.load_ema_ckpt = not args.no_ema

    if 'seq_len' in config:
        config.seq_len = train_dataset.get_seq_len()

    vae_training_wrapper = instantiate_from_config(config.model)

    # Generate and Save VAE Reconstructions
    #
    # This section generates reconstructions through the VAE encoder-decoder pipeline.
    # The reconstructions are saved in HDF5 format compatible with the ML/PRS training pipeline,
    # allowing us to evaluate whether the VAE preserves the predictive information in the data.
    #
    # **Note:** The VAE is not conditioned on phenotype labels, so we save the real targets
    # alongside the reconstructions. This allows fair comparison with the original data
    # when training downstream ML/PRS models.

    # Output directory (same as VAE checkpoint directory)
    output_dir = args.output_dir or os.path.dirname(reload_path)

    # =============================================================================
    # GENERATE RECONSTRUCTIONS
    # =============================================================================

    os.makedirs(output_dir, exist_ok=True)

    # Create the reconstruction generator
    recon_generator = ReconstructionGenerator(vae_training_wrapper, device=str(device))
    recon_generator.prepare_model()

    # Map split names to dataloaders
    split_dataloaders = {
        'train': train_dataloader,
        'val': val_dataloader,
        'test': test_dataloader,
        'train_val': train_val_dataloader,
        'full': complete_dataloader,
    }

    # Check if data is one-hot encoded
    onehot = config.data.raw_dataset.params.get('onehot', True)

    result = None
    for split in splits_to_reconstruct:
        print(f"\n{'='*60}")
        print(f"Processing split: {split}")
        print(f"{'='*60}")

        # Get output filename
        output_filename = get_output_filename(
            base_name='vae',
            mode='reconstruction',
            split=split
        )
        output_path = os.path.join(output_dir, output_filename)

        # Check if already exists
        if dataset_exists(output_dir, output_filename) and not args.overwrite:
            print(f"Reconstruction dataset already exists: {output_path}")
            if not args.skip_existing:
                raise FileExistsError(f'Output exists: {output_path}; pass --skip-existing or --overwrite')
            print("Skipping generation...")
            continue

        # Get the appropriate dataloader
        if split not in split_dataloaders:
            print(f"Warning: Unknown split '{split}', skipping...")
            continue

        dataloader = split_dataloaders[split]

        # Get pad mask if available
        pad_mask = getattr(dataloader.dataset, 'pad_mask', None)

        # Generate reconstructions using the inference module
        result = recon_generator.generate_reconstructions(
            dataloader,
            sample_posterior=sample_posterior,
            store_latents=store_latents,
            store_originals=store_originals,
            onehot=onehot,
            pad_mask=pad_mask,
            verbose=True
        )

        # ==========================================================================
        # EXTRACT EIDS FROM METADATA
        # ==========================================================================
        # Since we use shuffle=False in dataloaders, the order is preserved and
        # matches the split indices from raw_dataset.

        # Get metadata for this split
        _, original_targets, split_metadata = raw_dataset.get_split(split, metadata=True)

        if split_metadata is not None:
            # Check that dataloader targets match original targets otherwise metadata alignment is off
            assert np.all(original_targets == result.targets), "Mismatch between original targets and dataloader targets! Are dataloaders using shuffle=False?"

            result.eids = split_metadata.get('eids', None)

        # Save the reconstruction dataset
        # Uses 'syn_samples' key for compatibility with SplitDataset loader
        save_synthetic_dataset(
            result=result,
            output_path=output_path,
            mode='reconstruction',
            extra_attrs={
                'sample_posterior': sample_posterior,
                'vae_checkpoint': reload_path,
                'split': split,
            }
        )

        print(f"\nSaved reconstruction dataset to: {output_path}")
        print(f"  - Reconstructions shape: {result.reconstructions.shape}")
        print(f"  - Targets shape: {result.targets.shape}")
        if result.orig_samples is not None:
            print(f"  - Original samples shape: {result.orig_samples.shape}")
        if result.eids is not None:
            print(f"  - EIDs shape: {result.eids.shape}")

    # Keep results from last split for analysis
    # In a CLI resume where all files already exist, no new result was generated.
    if result is None:
        return
    reconstructions = result.reconstructions
    targets = result.targets
    orig_samples = result.orig_samples
    eids = result.eids

    # Create torch tensors for analysis
    reconstructions_torch = torch.from_numpy(reconstructions)
    orig_samples_torch = torch.from_numpy(orig_samples) if orig_samples is not None else None
    targets_torch = torch.from_numpy(targets)

    # Transpose for SNP-wise analysis
    orig_samples_T = np.transpose(orig_samples) if orig_samples is not None else None
    reconstructions_T = np.transpose(reconstructions)


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
