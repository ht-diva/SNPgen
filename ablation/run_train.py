"""Train one ablation model with the same flow as the ECCB VAE trainer."""

from __future__ import annotations

import argparse
import os
import re
from typing import Any, Iterable

import lightning.pytorch as pl
import torch
import wandb
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers.logger import DummyLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ablation.naming import build_run_dir, build_run_name, experiment_value
from ablation.utils import load_config, load_saved_vae_metadata
from snpgen.training.callbacks.checkpointing import ValidationAndPeriodicCheckpoint
from snpgen.training.callbacks.progress import SimpleProgressBar
from snpgen.training.loggers import setup_wandb_logger
from snpgen.utils import instantiate_from_config, save_config, scale_lr_optimizer_config

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _slurm_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _num_workers(config) -> int:
    if "num_workers" in config.data:
        return int(config.data.num_workers)
    return _slurm_int("SLURM_CPUS_PER_TASK", 4)


def _parse_devices(value: str):
    """Keep Lightning's supported device forms, typing numeric forms for CPU too."""
    if value.lower() == "auto":
        return "auto"
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if "," in value:
        try:
            return [int(device.strip()) for device in value.split(",")]
        except ValueError:
            pass
    return value


def _parse_batch_limit(value: str):
    """Parse integer batch counts as ints and fractional limits as floats."""
    if re.fullmatch(r"\+?\d+", value):
        return int(value)
    number = float(value)
    if number > 1 and number.is_integer():
        return int(number)
    return number


def _set_if_present(config, dotted_key: str, value: Any) -> None:
    if value is None:
        return
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        if part not in node:
            return
        node = node[part]
    if parts[-1] in node:
        node[parts[-1]] = value


def _load_trait_config(path: str, trait_name: str):
    traits = OmegaConf.load(path)
    if "traits" not in traits or trait_name not in traits.traits:
        valid = ", ".join(traits.get("traits", {}).keys())
        raise KeyError(f"Trait {trait_name!r} not found in {path}. Valid traits: {valid}")
    return traits.traits[trait_name]


def _apply_trait_overrides(config, trait_name: str | None, traits_config: str | None):
    if trait_name is None:
        return None
    if traits_config is None:
        raise ValueError("--traits-config is required when --trait is set")

    trait = _load_trait_config(traits_config, trait_name)
    metadata = None
    if trait.get("saved_vae_config"):
        metadata = load_saved_vae_metadata(trait.saved_vae_config, trait_name=trait_name)

    h5_path = trait.get("dataset_path")
    if h5_path is None and metadata is not None:
        h5_path = metadata["dataset_path"]
    if not h5_path:
        raise KeyError(
            f"Trait {trait_name!r} needs `dataset_path`; alternatively provide a `saved_vae_config` "
            "whose config contains dataset_path."
        )

    config.dataset_path = h5_path
    seq_len = trait.get("seq_len", metadata["seq_len"] if metadata is not None else None)
    if seq_len is not None:
        config.seq_len = int(seq_len)
        if "seq_len" in config.get("data", {}).get("dataset", {}).get("params", {}):
            config.data.dataset.params.seq_len = config.seq_len
    if "file_path" in config.data.raw_dataset.params:
        config.data.raw_dataset.params.file_path = h5_path

    if "experiment" not in config:
        config.experiment = {}
    config.experiment["proj_name"] = trait_name
    if metadata is not None:
        if metadata["base_scratch_dir"] is not None:
            config.experiment["base_scratch_dir"] = metadata["base_scratch_dir"]
        inherit_vae_model_metadata = bool(config.experiment.get("inherit_vae_model_metadata", False))
        if inherit_vae_model_metadata:
            config.experiment["model_size"] = metadata["model_size"]
            config.experiment["encoder_type"] = metadata["encoder_type"]
            config.experiment["emb_size"] = metadata["emb_size"]
            config.experiment["extra_name"] = metadata["extra_name"]
            if "use_discriminator" not in config.experiment:
                config.experiment["use_discriminator"] = metadata["use_discriminator"]

    _set_if_present(config, "model.params.seq_len", config.get("seq_len", None))
    return metadata


def _build_datasets_and_loaders(config, h5_path: str, seed: int):
    print(f"Loading Dataset from: {h5_path}")
    raw_dataset = instantiate_from_config(config.data.raw_dataset, file_path=h5_path, seed=seed)
    block_ids = raw_dataset.get_metadata("full", "block_id")

    train_dataset = instantiate_from_config(
        config.data.dataset,
        raw_dataset.get_split("train"),
        block_ids=block_ids,
    )
    val_dataset = instantiate_from_config(
        config.data.dataset,
        raw_dataset.get_split("val"),
        block_ids=block_ids,
    )
    test_dataset = instantiate_from_config(
        config.data.dataset,
        raw_dataset.get_split("test"),
        block_ids=block_ids,
    )

    workers = _num_workers(config)
    print(f"Using {workers} workers for the DataLoader")
    persistent_workers = workers > 0
    pin_memory = True

    batch_size = int(config.data.batch_size)
    val_batch_size = int(config.data.val_batch_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
    )
    return train_dataset, train_dataloader, val_dataloader, test_dataloader


def _precision_from_config(config) -> str:
    if "precision" in config.training:
        return str(config.training.precision)

    if not bool(config.training.get("mixed_precision", False)):
        return "32-true"
    try:
        bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return "bf16-mixed" if bf16_supported else "16-mixed"


def _model_configs(config) -> tuple[dict, dict, dict | None]:
    params = config.model.params
    encoder_config = {}
    decoder_config = {}
    discriminator_config = None

    if "autoencoder_config" in params:
        ae_params = params.autoencoder_config.params
        if "encoder_config" in ae_params:
            encoder_config = OmegaConf.to_container(ae_params.encoder_config.params, resolve=True)
        if "decoder_config" in ae_params:
            decoder_config = OmegaConf.to_container(ae_params.decoder_config.params, resolve=True)
    if "loss_config" in params and "discriminator_config" in params.loss_config.get("params", {}):
        discriminator_config = OmegaConf.to_container(
            params.loss_config.params.discriminator_config.params,
            resolve=True,
        )
    return encoder_config, decoder_config, discriminator_config


def _run_name(config, decoder_config: dict, ablation_name: str | None) -> str:
    return build_run_name(config, decoder_config, ablation_name=ablation_name)


def _run_dir(config, run_name: str, output_dir: str | None, base_scratch_dir: str | None) -> str:
    return build_run_dir(config, run_name, output_dir=output_dir, base_scratch_dir=base_scratch_dir)


def _callbacks(config, run_dir: str):
    enable_progress_bar = bool(config.training.get("enable_progress_bar", True))
    callbacks = [LearningRateMonitor(logging_interval="step")]
    callbacks.append(
        ValidationAndPeriodicCheckpoint(
            dirpath=run_dir,
            monitor=config.training.get("monitor", "val/loss"),
            mode=config.training.get("monitor_mode", "min"),
            every_n_epochs=config.training.get("every_n_epochs", None),
            periodic_filename=config.training.get("checkpoint_periodic_filename", "epoch={epoch}-step={step}"),
            best_filename=config.training.get(
                "checkpoint_best_filename",
                "best-epoch={epoch}-step={step}-val_loss={val_loss:.4f}",
            ),
            metric_filename_key=config.training.get("checkpoint_metric_filename_key", None),
            save_last=bool(config.training.get("save_last", False)),
        )
    )
    if enable_progress_bar:
        callbacks.append(SimpleProgressBar())
    return callbacks, enable_progress_bar


def _setup_logger(
    config,
    run_name: str,
    run_dir: str,
    h5_path: str,
    config_orig,
    encoder_config: dict,
    decoder_config: dict,
    discriminator_config: dict | None,
    batch_size: int,
    actual_batch_size: int,
    use_wandb: bool,
):
    if not use_wandb:
        return DummyLogger()

    resolved_config_dict = OmegaConf.to_container(config, resolve=True)
    extra_config = {
        "SLURM_JOBID": os.environ.get("SLURM_JOB_ID", "local"),
        "dataset_path": h5_path,
        "encoder_config": encoder_config,
        "decoder_config": decoder_config,
        "yaml_config": config_orig,
        "resolved_yaml_config": resolved_config_dict,
        "batch_size": batch_size,
        "actual_batch_size": actual_batch_size,
    }
    if discriminator_config is not None:
        extra_config["discriminator_config"] = discriminator_config

    base_scratch_dir = experiment_value(config, "base_scratch_dir", os.path.dirname(run_dir))
    proj_name = experiment_value(config, "proj_name", "trait")
    tags = list(config.training.get("wandb_tags", ["white_ethnicity", proj_name]))
    group = config.training.get("wandb_group", experiment_value(config, "wandb_group", "Ablation"))
    return setup_wandb_logger(
        project=config.training.get("wandb_project", "SNPgen"),
        name=run_name,
        save_code=True,
        save_dir=base_scratch_dir,
        group=group,
        tags=tags,
        extra_config=extra_config,
        extra_sync_metric="trainer/samples_seen",
    )


def _merge_dotlist(config, overrides: Iterable[str]):
    if not overrides:
        return config
    return OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", required=True, help="One or more OmegaConf YAML files to merge.")
    parser.add_argument("--traits-config", default=None, help="Trait registry used by --trait.")
    parser.add_argument("--trait", default=None, help="Trait key from --traits-config.")
    parser.add_argument("--output-dir", default=None, help="Explicit run directory. Defaults to checkpoint-style path.")
    parser.add_argument("--base-scratch-dir", default=None, help="Override experiment.base_scratch_dir.")
    parser.add_argument("--ablation-name", default=None, help="Run-name suffix identifying the ablation.")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--devices", type=_parse_devices, default=None)
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-wandb", dest="use_wandb", action="store_true", default=None)
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    parser.add_argument("--limit-train-batches", type=_parse_batch_limit, default=None)
    parser.add_argument("--limit-val-batches", type=_parse_batch_limit, default=None)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. training.max_epochs=5")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    _apply_trait_overrides(config, args.trait, args.traits_config)
    if args.base_scratch_dir is not None:
        if "experiment" not in config:
            config.experiment = {}
        config.experiment.base_scratch_dir = args.base_scratch_dir
    if args.ablation_name is not None:
        if "experiment" not in config:
            config.experiment = {}
        config.experiment.ablation_name = args.ablation_name
    config = _merge_dotlist(config, args.overrides)
    config_orig = config.copy()

    accelerator = args.accelerator or ("gpu" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if accelerator in {"gpu", "cuda"} else 0
    print(f"{num_gpus} GPU(s) available")
    if bool(config.training.get("scale_lr", True)) and "optimizer_config" in config.model.params:
        scale_lr_optimizer_config(config.model.params.optimizer_config, num_gpus=max(num_gpus, 1))

    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    seed_everything(seed, workers=True)

    h5_path = config.get("dataset_path", config.data.raw_dataset.params.get("file_path", None))
    if h5_path is None:
        raise ValueError("Set dataset_path or data.raw_dataset.params.file_path, or pass --trait.")
    config_orig["dataset_path"] = h5_path

    train_dataset, train_dataloader, val_dataloader, _test_dataloader = _build_datasets_and_loaders(config, h5_path, seed)
    if "seq_len" in config:
        config.seq_len = train_dataset.get_seq_len()
        _set_if_present(config, "data.dataset.params.seq_len", config.seq_len)
        _set_if_present(config, "model.params.seq_len", config.seq_len)

    encoder_config, decoder_config, discriminator_config = _model_configs(config)
    model = instantiate_from_config(config.model)
    if hasattr(model, "initialize_from_dataloader"):
        model.initialize_from_dataloader(train_dataloader)

    run_name = _run_name(config, decoder_config, args.ablation_name)
    run_dir = _run_dir(config, run_name, args.output_dir, args.base_scratch_dir)
    print("run_dir: ", run_dir)

    precision = _precision_from_config(config)
    print(precision)
    callbacks, enable_progress_bar = _callbacks(config, run_dir)

    batch_size = int(config.data.batch_size)
    actual_batch_size = batch_size * max(num_gpus, 1)
    use_wandb = bool(config.training.get("use_wandb", True)) if args.use_wandb is None else bool(args.use_wandb)
    logger = _setup_logger(
        config,
        run_name,
        run_dir,
        h5_path,
        config_orig,
        encoder_config,
        decoder_config,
        discriminator_config,
        batch_size,
        actual_batch_size,
        use_wandb,
    )

    devices = args.devices
    if devices is None:
        devices = num_gpus if accelerator in {"gpu", "cuda"} else "auto"
    if isinstance(devices, (list, tuple)):
        device_count = len(devices)
    elif isinstance(devices, int):
        device_count = devices if devices > 0 else num_gpus
    else:
        device_count = num_gpus
    strategy = "ddp" if accelerator in {"gpu", "cuda"} and device_count > 1 else "auto"

    trainer = pl.Trainer(
        max_epochs=int(args.max_epochs if args.max_epochs is not None else config.training.get("max_epochs", 400)),
        accelerator=accelerator,
        default_root_dir=run_dir,
        devices=devices,
        strategy=strategy,
        logger=logger,
        log_every_n_steps=int(config.training.get("log_every_n_steps", 1)),
        enable_checkpointing=True,
        enable_progress_bar=enable_progress_bar,
        callbacks=callbacks,
        precision=precision,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches is not None else 1.0,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches is not None else 1.0,
    )

    save_config(config_orig, run_dir)
    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
