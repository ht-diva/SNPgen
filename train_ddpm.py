#!/usr/bin/env python3
"""Train the latent DDPM with configurable command-line settings.

Configuration values come from command-line arguments.
"""

import argparse
import os
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description='Train a DDPM from YAML configuration and a trained VAE.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--encoder-type', choices=['encoder'], default='encoder')
    parser.add_argument('--emb-size', type=int, help='Embedding size for run naming only; inferred from VAE path/config by default.')
    parser.add_argument('--proj-name', default='trait1')
    parser.add_argument('--dataset-file', default='snp_dataset_kb10_r0.5_WHITE')
    parser.add_argument('--dataset-path', help='Explicit HDF5 input; overrides the data-root convention.')
    parser.add_argument('--base-scratch-dir', '--data-root', default='.', help='Root containing data/ukb_<trait> and checkpoints/<trait>.')
    parser.add_argument('--extra-name', help='Suffix appended to the run name.')
    parser.add_argument('--run-name')
    parser.add_argument('--output-dir', help='Explicit run directory; otherwise use the checkpoints/<trait> convention.')
    parser.add_argument('--config', action='append', help='Replace the auto-selected YAML list; repeat to merge files in order.')
    parser.add_argument('--set', dest='sets', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--val-batch-size', type=int)
    parser.add_argument('--num-workers', '--workers', type=int)
    parser.add_argument('--num-nodes', type=int, default=1)
    parser.add_argument('--accelerator', choices=['auto','cpu','gpu'], default='auto')
    parser.add_argument('--devices', type=int, help='Selected GPU count; default all visible GPUs.')
    parser.add_argument('--precision', choices=['32-true','16-mixed','bf16-mixed'])
    parser.add_argument('--strategy')
    parser.add_argument('--max-epochs', type=int, default=500)
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--accumulate-grad-batches', type=int, default=1)
    parser.add_argument('--limit-train-batches', type=batch_limit, default=1.0)
    parser.add_argument('--limit-val-batches', type=batch_limit, default=1.0)
    parser.add_argument('--monitor', help='Default: config training.monitor, usually val/loss.')
    parser.add_argument('--monitor-mode', choices=['min','max'])
    parser.add_argument('--save-last', action='store_true')
    parser.add_argument('--resume-from', help='Resume trainer/model state from an explicit checkpoint; use the saved config via --config.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--no-progress-bar', action='store_true')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging; disabled by default.')
    parser.add_argument('--wandb-project', default='SNPgen')
    parser.add_argument('--wandb-tags', nargs='*')
    parser.add_argument('--vae-checkpoint', help='Explicit VAE checkpoint; otherwise use the YAML setting.')
    parser.add_argument('--validation-seed', type=int)
    parser.add_argument('--validate-with-ema', action='store_true', default=None)
    parser.add_argument('--no-validate-with-ema', dest='validate_with_ema', action='store_false')
    parser.add_argument('--checkpoint-every-n-epochs', type=int)
    return parser

def batch_limit(value):
    # Lightning distinguishes an integer batch count from a float proportion.
    if str(value).isdigit() and int(value) > 0:
        return int(value)
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError('use a positive batch count or a proportion in (0, 1]')
    return number


def run(args):
    import torch

    import numpy as np
    import os
    import glob
    import re
    import wandb
    from omegaconf import OmegaConf

    from torchinfo import summary

    import lightning.pytorch as pl
    from lightning.pytorch import seed_everything
    from lightning.pytorch.loggers.logger import DummyLogger
    from lightning.pytorch.callbacks import LearningRateMonitor

    from snpgen.training.callbacks.checkpointing import ValidationAndPeriodicCheckpoint
    from snpgen.training.callbacks.progress import SimpleProgressBar
    from snpgen.training.loggers import setup_wandb_logger
    from snpgen.utils import instantiate_from_config, save_config, scale_lr_optimizer_config

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    NUM_WORKERS = args.num_workers if args.num_workers is not None else int(os.environ.get('SLURM_CPUS_PER_TASK', '0'))
    NUM_NODES = args.num_nodes
    ALLOCATED_GPUS_PER_NODE = args.devices
    SLURM_JOBID = os.environ.get('SLURM_JOB_ID', 'local')

    num_gpus = (torch.cuda.device_count() if args.devices is None else args.devices) if args.accelerator != 'cpu' and torch.cuda.is_available() else 0
    print(f"{num_gpus} GPU(s) available")
    print(f"Using {NUM_WORKERS} workers for the DataLoader")

    # ============================================================
    # RUN SETTINGS — supplied by CLI arguments
    # ============================================================

    # Model configuration
    encoder_type = args.encoder_type

    # Dataset fallback (used when the VAE config does not specify a dataset path)
    proj_name = args.proj_name
    h5_filename = args.dataset_file

    # Base directory for checkpoints and data
    base_scratch_dir = args.base_scratch_dir

    # Whether to use Wandb for logging
    use_wandb = args.wandb

    # Load Config

    assert encoder_type in ['encoder'], "Invalid encoder type"

    base = ['./configs/ddpm/base.yaml']
    base.append(f'./configs/ddpm/{encoder_type}/base.yaml')

    if args.config:
        base = args.config
    print(f"Loading config from: {base}")
    configs = [OmegaConf.load(cfg) for cfg in base]
    cli = OmegaConf.from_dotlist(args.sets)
    config = OmegaConf.merge(*configs, cli)

    if args.vae_checkpoint:
        config.vae_ckpt_path = os.path.abspath(args.vae_checkpoint)
        config.model.params.first_stage_config.params.ckpt_path = config.vae_ckpt_path
    if args.seed is not None:
        config.seed = args.seed
    if args.validation_seed is not None:
        config.model.params.validation_seed = args.validation_seed
    if args.validate_with_ema is not None:
        config.model.params.validate_with_ema_weights = args.validate_with_ema
    if args.monitor:
        config.training.monitor = args.monitor
    if args.monitor_mode:
        config.training.monitor_mode = args.monitor_mode
    if args.checkpoint_every_n_epochs is not None:
        config.training.every_n_epochs = args.checkpoint_every_n_epochs or None
    if args.save_last:
        config.training.save_last = True
    config_orig = config.copy() # keep a backup of the original config prior to any change

    n_classes = config.get('n_classes', 2)
    print(f"Using DISCRETE PHENOTYPE CONDITIONING with {n_classes} classes")

    # Try to load VAE config to get the dataset path, model params and data params
    vae_config_path = os.path.join(os.path.dirname(config.vae_ckpt_path), 'config.yaml')

    if os.path.exists(vae_config_path):
        print(f"Loading VAE config from: {vae_config_path}")
        vae_config = OmegaConf.load(vae_config_path)

        # Update dataset path from VAE config if available
        if 'dataset_path' in vae_config:
            h5_path = vae_config['dataset_path']
            print(f"Using dataset path from VAE config: {h5_path}")
            proj_name = os.path.basename(os.path.dirname(vae_config.dataset_path)).replace('ukb_', '')
            print(f"Inferred project name: {proj_name}")
        else:
            print("Dataset path not found in VAE config, will use manual definition")
            h5_path = None

        # Update data config from VAE config if available
        if 'data' in vae_config:
            print("Updating data config from VAE config")
            # Give priority to any manual definition in the DDPM config
            config.data = OmegaConf.merge(
                vae_config.data,
                config.data)

        split_seed = int(vae_config.get('dataset_split_seed', vae_config.data.raw_dataset.params.get('seed', vae_config.seed)))
        config.data.raw_dataset.params.seed = split_seed
        config.dataset_split_seed = split_seed
        # Update first stage (VAE) config using reloaded VAE config
        # Give priority to any manual definition in the DDPM config
        print("Updating first stage (VAE) config from VAE config")
        config.model.params.first_stage_config = OmegaConf.merge(
            vae_config.model.params.autoencoder_config,
            config.model.params.first_stage_config)

    else:
        raise FileNotFoundError(f"VAE config not found at {vae_config_path}, cannot proceed without dataset definition")

    encoder_config = OmegaConf.to_container(config.model.params.first_stage_config.params.encoder_config.params, resolve=True)
    decoder_config = OmegaConf.to_container(config.model.params.first_stage_config.params.decoder_config.params, resolve=True)

    vae_model_size = config.model.params.first_stage_config.model_size
    vae_ckpt_path = config.model.params.first_stage_config.params.ckpt_path
    vae_use_ema = config.model.params.first_stage_config.params.load_ema_ckpt

    resolved_config_dict = OmegaConf.to_container(config, resolve=True)
    config_orig = config.copy() # keep a backup of the original config prior to any change

    # Scale LR
    if not args.resume_from and hasattr(config.model.params, 'optimizer_config'):
        print(f"Scaling learning rate in optimizer config for {num_gpus} GPUs")
        scale_lr_optimizer_config(config.model.params.optimizer_config, num_gpus=max(num_gpus, 1))

    seed = config.get('seed', 42)
    seed_everything(seed, workers=True)

    # Build Dataset

    if h5_path is None:
        # Fallback to manual dataset definition
        print("Using manual dataset definition")

        data_path = os.path.join(base_scratch_dir, f'data/ukb_{proj_name}/')
        h5_path = os.path.join(data_path, h5_filename+'.hdf5')

    if args.dataset_path:
        h5_path = args.dataset_path
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.val_batch_size is not None:
        config.data.val_batch_size = args.val_batch_size
    config_orig['dataset_path'] = h5_path

    print(f"Loading Dataset from: {h5_path}")
    if config.get('data', {}).get('raw_dataset', None):
        if split_seed != seed:
            print(f"Overriding raw_dataset seed from {seed} to {split_seed}")
        raw_dataset = instantiate_from_config(config.data.raw_dataset, file_path=h5_path, seed=split_seed)
    else:
        raise ValueError("Raw dataset config not found in DDPM config, cannot proceed without dataset instantiation")

    if config.get('data', {}).get('dataset', None):
        train_dataset = instantiate_from_config(config.data.dataset, raw_dataset.get_split('train'), block_ids=raw_dataset.get_metadata('full', 'block_id') if hasattr(raw_dataset, 'get_metadata') else None)
        val_dataset = instantiate_from_config(config.data.dataset, raw_dataset.get_split('val'), block_ids=raw_dataset.get_metadata('full', 'block_id') if hasattr(raw_dataset, 'get_metadata') else None)
        test_dataset = instantiate_from_config(config.data.dataset, raw_dataset.get_split('test'), block_ids=raw_dataset.get_metadata('full', 'block_id') if hasattr(raw_dataset, 'get_metadata') else None)
    else:
        raise ValueError("Dataset config not found in DDPM config, cannot proceed without dataset instantiation")

    batch_size = config.data.batch_size
    actual_batch_size = batch_size * max(num_gpus, 1) * NUM_NODES * args.accumulate_grad_batches
    val_batch_size = config.data.val_batch_size

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
        #sampler=ImbalancedDatasetSampler(train_dataset, strategy='inverse_freq'), # balance dataset on labels (which also implicitly performs shuffling)
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=NUM_WORKERS > 0,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=NUM_WORKERS > 0,
    )

    # Build Model

    if 'seq_len' in config:
        config.seq_len = train_dataset.get_seq_len()

    ddpm_training_wrapper = instantiate_from_config(config.model)

    # bs = 3
    # summary(ddpm_training_wrapper.model.diffusion_model, [(bs, ddpm_training_wrapper.model.diffusion_model.in_channels, emb_size), (bs,), (bs, 1, ddpm_training_wrapper.model.diffusion_model.context_dim)], dtypes=[torch.float32, torch.float32, torch.float32], depth=2)

    # Train

    extra_name = args.extra_name if args.extra_name is not None else f'{vae_model_size}_white'

    # Use the run naming convention when available; a checkpoint
    # path need not encode its embedding size in the parent directory name.
    emb_match = re.search(r"emb(\d+)", os.path.basename(os.path.dirname(vae_ckpt_path)))
    emb_size = args.emb_size or (int(emb_match.group(1)) if emb_match else decoder_config['z_dim'])

    actual_emb_size = decoder_config['z_dim']

    run_name = f"{proj_name}_ddpm\
    _emb{emb_size}{f'_actualEmb{actual_emb_size}' if actual_emb_size != emb_size else ''}\
    {f'_{extra_name}' if extra_name != '' else ''}-{SLURM_JOBID}"
    base_run_dir = os.path.join(base_scratch_dir, f"checkpoints/{proj_name}")
    if args.run_name:
        run_name = args.run_name
    run_dir = args.output_dir or f"{base_run_dir}/{run_name}/"
    if os.path.isdir(run_dir) and os.listdir(run_dir) and not (args.overwrite or args.resume_from):
        raise FileExistsError(f'Run directory is not empty: {run_dir}; pass --overwrite or --resume-from')
    os.makedirs(run_dir, exist_ok=True)
    print('run_dir: ', run_dir)

    mixed_precision = config.training.mixed_precision


    if args.precision:
        precision = args.precision
    elif num_gpus == 0:
        precision = '32-true'
    elif mixed_precision:
        if torch.cuda.torch.cuda.is_bf16_supported(including_emulation=False):
            precision = 'bf16-mixed'
        else:
            precision = '16-mixed'
    else:
        precision = '32-true'

    print(precision)

    enable_progress_bar = not args.no_progress_bar

    model_ckpt_cb = ValidationAndPeriodicCheckpoint(
        dirpath=run_dir,
        monitor=config.training.get("monitor", "val/loss"),
        mode=config.training.get("monitor_mode", "min"),
        every_n_epochs=config.training.get("every_n_epochs", None),
        periodic_filename=config.training.get("checkpoint_periodic_filename", "epoch={epoch}-step={step}"),
        best_filename=config.training.get(
            "checkpoint_best_filename",
            "best-epoch={epoch}-step={step}-val_loss={val_loss:.4f}",
        ),
        save_last=bool(config.training.get("save_last", False)),
    )

    lr_monitor_cb = LearningRateMonitor(logging_interval='step')

    callbacks = [
        lr_monitor_cb,
        model_ckpt_cb,
    ]

    if enable_progress_bar:
        callbacks.append(SimpleProgressBar())

    resolved_config_dict = OmegaConf.to_container(config, resolve=True)

    extra_config = {
        "SLURM_JOBID": SLURM_JOBID, "dataset_path": h5_path,
        "encoder_config": encoder_config, "decoder_config": decoder_config,
        "yaml_config": config_orig, "resolved_yaml_config": resolved_config_dict,
        "batch_size": batch_size, "actual_batch_size": actual_batch_size,
        "vae_ckpt_path": vae_ckpt_path, "vae_use_ema": vae_use_ema,
    }

    if use_wandb:
        wandb_logger = setup_wandb_logger(project=args.wandb_project, name=run_name, save_code=True, save_dir=base_scratch_dir,
                                        group='DDPM', tags=args.wandb_tags or [proj_name],
                                        extra_config=extra_config, extra_sync_metric="trainer/samples_seen")

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        num_nodes=NUM_NODES,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        accelerator="gpu" if num_gpus else "cpu",
        default_root_dir=run_dir,
        devices=max(num_gpus, 1), # devices=ALLOCATED_GPUS_PER_NODE
        strategy=args.strategy or ('auto' if max(num_gpus, 1) * NUM_NODES == 1 else 'ddp'),
        logger=wandb_logger if use_wandb else DummyLogger(),
        log_every_n_steps=1,
        enable_checkpointing=True,
        enable_progress_bar=enable_progress_bar,
        callbacks=callbacks,
        precision=precision,
        #limit_train_batches=10, # only for testing
        #limit_val_batches=5, # only for testing
    )

    # Save the effective configuration and the VAE training split.
    config.dataset_path = h5_path
    config.data.raw_dataset.params.seed = split_seed
    config.dataset_split_seed = split_seed
    save_config(OmegaConf.create(OmegaConf.to_container(config, resolve=True)), run_dir)
    trainer.fit(ddpm_training_wrapper, train_dataloader, val_dataloaders=val_dataloader, ckpt_path=args.resume_from)

    if use_wandb:
        wandb.finish()

    # another round of training
    # trainer.fit_loop.max_epochs = trainer.max_epochs + 1000
    # trainer.fit(ddpm_training_wrapper, train_dataloader, val_dataloaders=val_dataloader, ckpt_path=get_latest_ckpt(f'{run_dir}/*.ckpt'))


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
