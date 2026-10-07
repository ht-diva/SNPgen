"""Generate paper-style synthetic cohorts for one ablation checkpoint."""

from __future__ import annotations

import argparse
import glob
import h5py
import os
import re

import numpy as np
import torch
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf
from tqdm import tqdm

from ablation.utils import resolve_checkpoint_dir_with_config
from snpgen.inference import get_output_filename
from snpgen.utils import instantiate_from_config

OmegaConf.register_new_resolver("eval", eval, replace=True)

FLOAT_PATTERN = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"


def _monitored_vae_checkpoint(checkpoint_dir: str) -> str | None:
    candidates = glob.glob(os.path.join(checkpoint_dir, "*val_accuracy_recons=*.ckpt"))
    if not candidates:
        return None

    def score(path: str) -> tuple[float, str]:
        match = re.search(rf"val_accuracy_recons={FLOAT_PATTERN}\.ckpt$", os.path.basename(path))
        value = float(match.group(1)) if match else float("-inf")
        return value, path

    return os.path.basename(max(candidates, key=score))


def _best_checkpoint(checkpoint_dir: str) -> str | None:
    candidates = glob.glob(os.path.join(checkpoint_dir, "best-*-val_recon_acc=*.ckpt"))
    candidates.extend(glob.glob(os.path.join(checkpoint_dir, "best-*-val_loss=*.ckpt")))
    if not candidates:
        return None
    return os.path.basename(candidates[0])


def _preferred_checkpoint(checkpoint_dir: str) -> str | None:
    best_checkpoint = _best_checkpoint(checkpoint_dir)
    if best_checkpoint is not None:
        return best_checkpoint
    return _monitored_vae_checkpoint(checkpoint_dir)


def _load_checkpoint_model(config, checkpoint_path: str, device: str):
    model = instantiate_from_config(config.model)
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state.get("state_dict", state)
    model_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    skipped = sorted(set(state_dict) - set(compatible_state))
    if skipped:
        print(f"Skipping {len(skipped)} incompatible checkpoint tensors: {', '.join(skipped)}")
    model.load_state_dict(compatible_state, strict=False)
    model.to(device)
    model.eval()
    return model


def _sample_model(model, labels: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "sample"):
        return model.sample(labels, argmax=True)
    if hasattr(model, "autoencoder") and hasattr(model.autoencoder, "sample"):
        return model.autoencoder.sample(labels, argmax=True)
    raise TypeError(f"{type(model).__name__} does not expose sample(labels)")


def _generate_from_labels(model, labels: np.ndarray, batch_size: int, device: str) -> dict:
    samples = []
    targets = []
    labels = np.asarray(labels)
    for start in tqdm(range(0, labels.shape[0], batch_size), desc="Generating"):
        batch_labels = torch.as_tensor(labels[start:start + batch_size], device=device).long()
        with torch.no_grad():
            batch_samples = _sample_model(model, batch_labels)
        samples.append(batch_samples.detach().cpu().numpy().astype(np.int8, copy=False))
        targets.append(batch_labels.detach().cpu().numpy())
    return {
        "targets": np.concatenate(targets, axis=0),
        "samples": np.concatenate(samples, axis=0),
    }


def _write_generated_from_labels(
    model,
    labels: np.ndarray,
    batch_size: int,
    device: str,
    output_path: str,
    extra_attrs: dict,
    augmentation_strategy: str | None = None,
) -> int:
    labels = np.asarray(labels)
    tmp_path = f"{output_path}.tmp.{os.getpid()}"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    sample_ds = None
    written = 0
    with h5py.File(tmp_path, "w") as handle:
        handle.create_dataset("targets", data=labels, compression="gzip")
        if augmentation_strategy:
            handle.attrs["augmentation_strategy"] = augmentation_strategy
        for key, value in extra_attrs.items():
            handle.attrs[key] = value

        iterator = tqdm(range(0, labels.shape[0], batch_size), desc="Generating")
        for start in iterator:
            batch_labels_np = labels[start:start + batch_size]
            batch_labels = torch.as_tensor(batch_labels_np, device=device).long()
            with torch.no_grad():
                batch_samples = _sample_model(model, batch_labels)
            batch_samples_np = batch_samples.detach().cpu().numpy().astype(np.int8, copy=False)

            if sample_ds is None:
                sample_shape = (labels.shape[0],) + batch_samples_np.shape[1:]
                sample_ds = handle.create_dataset(
                    "syn_samples",
                    shape=sample_shape,
                    dtype=batch_samples_np.dtype,
                    compression="gzip",
                    chunks=(min(batch_size, labels.shape[0]),) + batch_samples_np.shape[1:],
                )

            end = start + batch_samples_np.shape[0]
            sample_ds[start:end] = batch_samples_np
            written = end

    os.replace(tmp_path, output_path)
    return written


def _balanced_binary_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    num_controls = int((labels == 0).sum())
    return np.concatenate([
        np.zeros(num_controls, dtype=np.int64),
        np.ones(num_controls, dtype=np.int64),
    ])


def _fixed_prevalence_labels(reference: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """Create labels using only a reference prevalence, never held-out identities."""
    reference = np.asarray(reference).reshape(-1)
    prevalence = float(np.mean(reference == 1))
    n_cases = int(round(prevalence * n_samples))
    labels = np.concatenate(
        (np.zeros(n_samples - n_cases, dtype=np.int64), np.ones(n_cases, dtype=np.int64))
    )
    np.random.default_rng(seed).shuffle(labels)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--modes", nargs="+", default=["complete", "augmented"], choices=["complete", "augmented"])
    parser.add_argument("--batch-size", type=int, default=6144)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--complete-label-design",
        choices=["train_prevalence", "full_exact"],
        default="train_prevalence",
        help="Use training prevalence (strict default) or the legacy full label vector.",
    )
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
        seed=config.get("seed", args.seed),
    )
    _train_data, train_labels = raw_dataset.get_split("train")
    train_labels = np.asarray(train_labels).reshape(-1).astype(np.int64)
    n_full = int(raw_dataset.data.shape[0])
    if args.complete_label_design == "full_exact":
        complete_labels = np.asarray(raw_dataset.targets).reshape(-1).astype(np.int64)
    else:
        complete_labels = _fixed_prevalence_labels(train_labels, n_full, args.seed)
    estimated_controls = int(round(float(np.mean(train_labels == 0)) * n_full))

    model = _load_checkpoint_model(config, checkpoint_path, args.device)

    for mode in args.modes:
        output_filename = get_output_filename(base_name="syn", mode=mode)
        output_path = os.path.join(args.checkpoint_dir, output_filename)
        if os.path.exists(output_path) and not args.overwrite:
            print(f"Dataset already exists: {output_path}")
            continue

        if mode == "complete":
            labels = complete_labels
            extra_attrs = {
                "ablation_generation_mode": "complete",
                "label_design": args.complete_label_design,
                "label_reference_split": "train" if args.complete_label_design == "train_prevalence" else "full",
            }
            augmentation_strategy = None
        elif mode == "augmented":
            labels = np.concatenate(
                (np.zeros(estimated_controls, dtype=np.int64), np.ones(estimated_controls, dtype=np.int64))
            )
            extra_attrs = {
                "ablation_generation_mode": "augmented",
                "label_design": "balanced_from_training_control_prevalence",
                "label_reference_split": "train",
            }
            augmentation_strategy = "binary_balanced"
        else:
            raise NotImplementedError(mode)

        n_written = _write_generated_from_labels(
            model=model,
            labels=labels,
            batch_size=args.batch_size,
            device=args.device,
            output_path=output_path,
            extra_attrs=extra_attrs,
            augmentation_strategy=augmentation_strategy,
        )
        print(f"Saved dataset to: {output_path}")
        print(f"Generated {n_written} samples")


if __name__ == "__main__":
    main()
