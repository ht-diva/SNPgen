"""Shared helpers for ablation command-line entrypoints."""

from __future__ import annotations

import json
import os
import re
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
from omegaconf import OmegaConf


OmegaConf.register_new_resolver("eval", eval, replace=True)


_SAVED_RUN_NAME_RE = re.compile(
    r"^(?P<run_prefix>.+?)(?P<disc>_disc)?(?:_(?P<encoder_type>[^_]+))?_emb"
    r"(?P<emb_size>\d+)(?:_actualEmb(?P<actual_emb>\d+))?(?:_(?P<extra_name>.+))?$"
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_paths: Iterable[str]):
    configs = [OmegaConf.load(path) for path in config_paths]
    if not configs:
        raise ValueError("At least one config path is required")
    return OmegaConf.merge(*configs)


def _base_scratch_dir_from_saved_config(path: Path) -> str | None:
    parts = path.resolve().parts
    if "checkpoints" not in parts:
        return None
    checkpoints_index = parts.index("checkpoints")
    if checkpoints_index == 0:
        return None
    return str(Path(*parts[:checkpoints_index]))


def load_saved_vae_metadata(saved_vae_config: str, trait_name: str | None = None) -> Dict[str, Any]:
    config_path = Path(saved_vae_config).expanduser().resolve()
    config = OmegaConf.load(config_path)
    run_dir = config_path.parent
    run_name = run_dir.name
    run_stem = run_name.rsplit("-", 1)[0]
    match = _SAVED_RUN_NAME_RE.match(run_stem)

    saved_model = config.get("model", {})
    saved_model_params = saved_model.get("params", {}) if saved_model else {}
    autoencoder_config = saved_model_params.get("autoencoder_config", {}) if saved_model_params else {}
    autoencoder_params = autoencoder_config.get("params", {}) if autoencoder_config else {}
    decoder_config = autoencoder_params.get("decoder_config", {}) if autoencoder_params else {}

    seq_len = int(config.get("seq_len", config.data.dataset.params.get("seq_len", 0)))
    z_dim = int(decoder_config.get("params", {}).get("z_dim", seq_len)) if decoder_config else seq_len
    model_size = str(autoencoder_config.get("model_size", "small")) if autoencoder_config else "small"

    metadata: Dict[str, Any] = {
        "saved_vae_config": str(config_path),
        "proj_name": trait_name or (run_dir.parent.name if run_dir.parent.name != "checkpoints" else run_dir.name),
        "base_scratch_dir": _base_scratch_dir_from_saved_config(config_path),
        "run_name": run_name,
        "dataset_path": str(config["dataset_path"]),
        "seq_len": seq_len,
        "decoder_z_dim": z_dim,
        "z_dim": z_dim,
        "model_size": model_size,
        "use_discriminator": "Discriminator" in str(saved_model.get("target", "")),
        "encoder_type": "encoder",
        "emb_size": z_dim,
        "actual_emb_size": z_dim,
        "extra_name": f"{model_size}_white",
    }

    if match:
        metadata["emb_size"] = int(match.group("emb_size"))
        metadata["actual_emb_size"] = int(match.group("actual_emb")) if match.group("actual_emb") else z_dim
        metadata["encoder_type"] = match.group("encoder_type") or "encoder"
        metadata["extra_name"] = match.group("extra_name") or f"{model_size}_white"

    if not metadata["extra_name"]:
        metadata["extra_name"] = f"{model_size}_white"

    return metadata


def save_config(config, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "config.yaml")
    OmegaConf.save(config, path)
    return path


def resolve_checkpoint_dir_with_config(checkpoint_dir: str) -> str:
    """Return an existing checkpoint dir containing config.yaml.

    Some early WGAN/CRBM ablation runs inherited the saved VAE `_disc` naming
    flag even though those baselines do not use the VAE auxiliary discriminator.
    New runs should omit `_disc`; this fallback keeps downstream scripts working
    when they are handed the corrected non-disc path for an older disc-named run.
    """

    checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir))
    if os.path.exists(os.path.join(checkpoint_dir, "config.yaml")):
        return checkpoint_dir

    parent = os.path.dirname(checkpoint_dir)
    name = os.path.basename(checkpoint_dir)
    candidates = []

    if "_disc_" in name:
        candidates.append(os.path.join(parent, name.replace("_disc_", "_", 1)))

    for prefix in ("ablation_wgan_gp", "ablation_crbm"):
        token = f"_{prefix}_"
        disc_token = f"_{prefix}_disc_"
        if token in name and disc_token not in name:
            candidates.append(os.path.join(parent, name.replace(token, disc_token, 1)))

    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "config.yaml")):
            print(f"Resolved checkpoint dir with config.yaml: {checkpoint_dir} -> {candidate}")
            return candidate

    return checkpoint_dir


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
