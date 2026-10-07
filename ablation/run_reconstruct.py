"""Generate cVAE reconstruction datasets for paper-style downstream evaluation."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from ablation.run_generate import _preferred_checkpoint
from ablation.utils import resolve_checkpoint_dir_with_config
from snpgen.inference import get_output_filename, save_synthetic_dataset
from snpgen.utils import instantiate_from_config

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _num_workers() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK", 4))


def _load_model(config, checkpoint_path: str, device: str):
    model = instantiate_from_config(config.model)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state), strict=False)
    model.to(device)
    model.eval()
    return model


def _batch_xy(batch):
    if isinstance(batch, dict):
        return batch["x"], batch["y"]
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return batch[0], batch[1]
    raise TypeError(f"Expected dict batch or (x, y) tuple, got {type(batch).__name__}")


def _reconstruct(model, dataloader, device: str, sample_posterior: bool) -> dict:
    targets = []
    reconstructions = []
    for batch in tqdm(dataloader, desc="Reconstructing"):
        x, y = _batch_xy(batch)
        x = x.to(device)
        y = y.to(device).long()
        with torch.no_grad():
            mu, logvar = model.autoencoder.encode(x, y)
            if sample_posterior:
                z = model.autoencoder.reparameterize(mu, logvar)
            else:
                z = mu
            recons = model.autoencoder.decode(z, y, argmax=True)
        targets.append(y.detach().cpu().numpy())
        reconstructions.append(recons.detach().cpu().numpy())
    return {
        "targets": np.concatenate(targets, axis=0),
        "reconstructions": np.concatenate(reconstructions, axis=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--splits", nargs="+", default=["train_val", "test"], choices=["train", "val", "test", "train_val", "full"])
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample-posterior", action="store_true", default=True)
    parser.add_argument("--deterministic", dest="sample_posterior", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.checkpoint_dir = resolve_checkpoint_dir_with_config(args.checkpoint_dir)

    checkpoint_name = args.checkpoint
    selected_marker = os.path.join(args.checkpoint_dir, "selected_checkpoint.txt")
    if checkpoint_name is None and os.path.exists(selected_marker):
        with open(selected_marker) as handle:
            checkpoint_name = handle.read().strip()
        print(f"Using selected checkpoint from {selected_marker}: {checkpoint_name}")
    if checkpoint_name is None:
        checkpoint_name = _preferred_checkpoint(args.checkpoint_dir)
        if checkpoint_name is not None:
            print(f"Using best validation checkpoint: {checkpoint_name}")
    if checkpoint_name is None:
        print("No selected or monitored checkpoint found, defaulting to last.ckpt")
        checkpoint_name = "last.ckpt"

    checkpoint_path = checkpoint_name
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_path)

    config_path = os.path.join(args.checkpoint_dir, "config.yaml")
    assert os.path.exists(config_path), f"config.yaml not found at {config_path}"
    assert os.path.exists(checkpoint_path), f"checkpoint not found at {checkpoint_path}"

    seed_everything(args.seed, workers=True)
    config = OmegaConf.load(config_path)
    h5_path = config["dataset_path"]
    print(f"Loading Dataset from: {h5_path}")

    raw_dataset = instantiate_from_config(
        config.data.raw_dataset,
        file_path=h5_path,
        metadata=True,
        seed=config.get("seed", args.seed),
    )
    block_ids = raw_dataset.get_metadata("full", "block_id") if hasattr(raw_dataset, "get_metadata") else None

    if "seq_len" in config:
        config.seq_len = int(config.seq_len)

    model = _load_model(config, checkpoint_path, args.device)

    for split in args.splits:
        output_filename = get_output_filename(base_name="vae", mode="reconstruction", split=split)
        output_path = os.path.join(args.checkpoint_dir, output_filename)
        if os.path.exists(output_path) and not args.overwrite:
            print(f"Reconstruction dataset already exists: {output_path}")
            continue

        dataset = instantiate_from_config(
            config.data.dataset,
            raw_dataset.get_split(split),
            block_ids=block_ids,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=_num_workers(),
            pin_memory=True,
            drop_last=False,
            persistent_workers=_num_workers() > 0,
        )
        result = _reconstruct(model, dataloader, args.device, args.sample_posterior)
        save_synthetic_dataset(
            result=result,
            output_path=output_path,
            mode="reconstruction",
            extra_attrs={"ablation_generation_mode": "reconstruction", "split": split},
        )
        print(f"Saved reconstruction dataset to: {output_path}")


if __name__ == "__main__":
    main()
