#!/usr/bin/env python3
"""Train the VAE with configurable command-line settings.

Configuration values come from command-line arguments.
"""

import argparse
import os
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description='Train a VAE from YAML configuration and genotype data.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-size', choices=['tiny','small','medium','big'], default='small')
    parser.add_argument('--encoder-type', choices=['encoder'], default='encoder')
    parser.add_argument('--emb-size', type=int, choices=[32,64,128,256], default=128)
    parser.add_argument('--no-discriminator', action='store_true')
    parser.add_argument('--proj-name', default='trait1')
    parser.add_argument('--dataset-file', default='snp_dataset_kb10_r0.5_WHITE')
    parser.add_argument('--dataset-path', help='Explicit HDF5 input; overrides the data-root convention.')
    parser.add_argument('--base-scratch-dir', '--data-root', default='.', help='Root containing data/ukb_<trait> and checkpoints/<trait>.')
    parser.add_argument('--extra-name', default='', help='Suffix appended to the run name.')
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
    parser.add_argument('--max-epochs', type=int, default=400)
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--accumulate-grad-batches', type=int, default=1)
    parser.add_argument('--limit-train-batches', type=batch_limit, default=1.0)
    parser.add_argument('--limit-val-batches', type=batch_limit, default=1.0)
    parser.add_argument('--monitor', default='val/metrics/recons/accuracy')
    parser.add_argument('--monitor-mode', choices=['min','max'], default='max')
    parser.add_argument('--save-last', action='store_true')
    parser.add_argument('--resume-from', help='Resume trainer/model state from an explicit checkpoint; use the saved config via --config.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--no-progress-bar', action='store_true')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging; disabled by default.')
    parser.add_argument('--wandb-project', default='SNPgen')
    parser.add_argument('--wandb-tags', nargs='*')
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
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError('--num-workers must be non-negative')
    if args.devices is not None and args.devices < 1:
        raise ValueError('--devices must be positive')
    import torch

    import os
    import glob
    import re
    import wandb
    from omegaconf import OmegaConf

    from torchinfo import summary

    import lightning.pytorch as pl
    from lightning.pytorch import seed_everything
    from lightning.pytorch.loggers.logger import DummyLogger
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

    from snpgen.utils import instantiate_from_config, save_config, scale_lr_optimizer_config
    from snpgen.training.callbacks.progress import SimpleProgressBar
    from snpgen.training.loggers import setup_wandb_logger

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
    model_size = args.model_size
    encoder_type = args.encoder_type
    emb_size = args.emb_size
    use_discriminator = not args.no_discriminator

    # Dataset and experiment naming
    proj_name = args.proj_name
    h5_filename = args.dataset_file
    extra_name = args.extra_name or ''  # Suffix appended to the run name

    # Base directory for checkpoints and data
    base_scratch_dir = args.base_scratch_dir

    # Whether to use Wandb for logging
    use_wandb = args.wandb

    # Load Config

    assert model_size in ['tiny', 'small', 'medium', 'big'], "Invalid model size"
    assert emb_size in [32, 64, 128, 256], "Invalid embedding size"
    assert encoder_type in ['encoder',], "Invalid encoder type"

    base = ['./configs/vae/base.yaml']
    base.append(f'./configs/vae/{encoder_type}/base.yaml')

    if model_size == 'tiny':
        base.append(f'./configs/vae/{encoder_type}/tiny_emb{emb_size}.yaml')
    elif model_size == 'small':
        base.append(f'./configs/vae/{encoder_type}/small_emb{emb_size}.yaml')
    elif model_size == 'medium':
        base.append(f'./configs/vae/{encoder_type}/medium_emb{emb_size}.yaml')
    elif model_size == 'big':
        base.append(f'./configs/vae/{encoder_type}/big_emb{emb_size}.yaml')
    else:
        raise NotImplementedError

    if use_discriminator:
        base.append(f'./configs/vae/base_disc.yaml')

    if args.config:
        base = args.config
    print(f"Loading config from: {base}")
    configs = [OmegaConf.load(cfg) for cfg in base]
    cli = OmegaConf.from_dotlist(args.sets)
    config = OmegaConf.merge(*configs, cli)

    if args.seed is not None:
        config.seed = args.seed
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.val_batch_size is not None:
        config.data.val_batch_size = args.val_batch_size
    config_orig = config.copy() # keep a backup of the original config prior to any change

    encoder_config = OmegaConf.to_container(config.model.params.autoencoder_config.params.encoder_config.params, resolve=True)
    decoder_config = OmegaConf.to_container(config.model.params.autoencoder_config.params.decoder_config.params, resolve=True)

    if use_discriminator:
        discriminator_config = OmegaConf.to_container(config.model.params.loss_config.params.discriminator_config.params, resolve=True)

    vae_model_size = config.model.params.autoencoder_config.model_size

    # Scale LR
    if not args.resume_from:
        scale_lr_optimizer_config(config.model.params.optimizer_config, num_gpus=max(num_gpus, 1))

    seed = int(config.get('seed', 42))
    # Dataset loading below uses this seed; keep saved split metadata consistent.
    config.data.raw_dataset.params.seed = seed
    seed_everything(seed, workers=True)

    # Build Dataset

    data_path = os.path.join(base_scratch_dir, f'data/ukb_{proj_name}/')
    h5_path = args.dataset_path or config.get('dataset_path') or os.path.join(data_path, h5_filename if h5_filename.endswith('.hdf5') else h5_filename+'.hdf5')

    config_orig['dataset_path'] = h5_path

    print(f"Loading Dataset from: {h5_path}")
    raw_dataset = instantiate_from_config(config.data.raw_dataset, file_path=h5_path, seed=seed)

    train_dataset = instantiate_from_config(config.data.dataset, raw_dataset.get_split('train'), block_ids=raw_dataset.get_metadata('full', 'block_id'))
    val_dataset = instantiate_from_config(config.data.dataset, raw_dataset.get_split('val'), block_ids=raw_dataset.get_metadata('full', 'block_id'))
    test_dataset = instantiate_from_config(config.data.dataset, raw_dataset.get_split('test'), block_ids=raw_dataset.get_metadata('full', 'block_id'))

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

    # Build Models

    if 'seq_len' in config:
        config.seq_len = train_dataset.get_seq_len()

    vae_training_wrapper = instantiate_from_config(config.model)

    # torch.compile() is quite broken when used with PyTorch Lightning modules, especially for the logging stuff
    #vae_training_wrapper = torch.compile(vae_training_wrapper, mode="reduce-overhead", dynamic=True, fullgraph=True)

    # summary(vae_training_wrapper.autoencoder.encoder, input_size=[(3, 3, train_dataset.get_seq_len())], dtypes=[torch.float32], depth=2)

    # summary(vae_training_wrapper.autoencoder.decoder, input_size=[(3, decoder_config['z_channels'], decoder_config['z_dim'])], dtypes=[torch.float32], depth=2)

    # summary(vae_training_wrapper.autoencoder.decoder, input_size=[(3, decoder_config['z_dim'])], dtypes=[torch.float32], depth=2)

    # if use_discriminator:
    #     print(summary(vae_training_wrapper.loss.discriminator, input_size=[(3, 3, seq_len)], dtypes=[torch.float32], depth=2))

    # Train

    actual_emb_size = decoder_config['z_dim']

    run_name = f"{proj_name}_vae{f'_disc' if use_discriminator else ''}\
    {f'_{encoder_type}' if encoder_type != 'encoder' else ''}\
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

    metric_to_monitor = args.monitor
    filename = f'epoch={{epoch}}-step={{step}}-val_accuracy_recons={{{metric_to_monitor}:.3f}}'

    model_ckpt_cb = ModelCheckpoint(
        dirpath=run_dir,
        monitor=metric_to_monitor,
        mode=args.monitor_mode,
        filename=filename,
        auto_insert_metric_name=False,
        save_last=args.save_last,
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
        "batch_size": batch_size, "actual_batch_size": actual_batch_size
    }
    if use_discriminator:
        extra_config['discriminator_config'] = discriminator_config

    if use_wandb:
        wandb_logger = setup_wandb_logger(project=args.wandb_project, name=run_name, save_code=True, save_dir=base_scratch_dir,
                                        group='VAE', tags=args.wandb_tags or [proj_name],
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
        #limit_train_batches=50, # only for testing
        #limit_val_batches=5, # only for testing
    )

    # Save the effective configuration used by this run, including the dataset and split.
    config.dataset_path = h5_path
    config.dataset_split_seed = seed
    save_config(OmegaConf.create(OmegaConf.to_container(config, resolve=True)), run_dir)
    trainer.fit(vae_training_wrapper, train_dataloader, val_dataloaders=val_dataloader, ckpt_path=args.resume_from)

    if use_wandb:
        wandb.finish()


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
