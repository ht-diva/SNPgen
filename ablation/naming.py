"""Shared naming helpers for ablation checkpoint directories."""

from __future__ import annotations

import os
from typing import Any


ABLATION_PREFIXES = {
    "conditional_vae": "ablation_cvae",
    "conditional_wgan_gp": "ablation_wgan_gp",
    "conditional_crbm": "ablation_crbm",
}


def ablation_prefix(ablation_name: str | None) -> str | None:
    if ablation_name is None:
        return None
    return ABLATION_PREFIXES.get(ablation_name, ablation_name)


def experiment_value(config, key: str, default: Any = None) -> Any:
    experiment = config.get("experiment", {})
    return experiment.get(key, default)


def build_run_name(config, decoder_config: dict | None = None, ablation_name: str | None = None, slurm_jobid: str | None = None) -> str:
    decoder_config = decoder_config or {}
    slurm_jobid = slurm_jobid or os.environ.get("SLURM_JOB_ID", "local")
    proj_name = experiment_value(config, "proj_name", "trait")
    model_size = experiment_value(config, "model_size", "small")
    encoder_type = experiment_value(config, "encoder_type", "encoder")
    emb_size = int(experiment_value(config, "emb_size", decoder_config.get("z_dim", config.get("z_dim", 128))))
    extra_name = experiment_value(config, "extra_name", f"{model_size}_white")
    use_discriminator = bool(experiment_value(config, "use_discriminator", "Discriminator" in str(config.model.target)))
    run_prefix = experiment_value(config, "run_prefix", "vae")
    actual_emb_size = int(decoder_config.get("z_dim", experiment_value(config, "z_dim", emb_size)))

    if ablation_name is None:
        ablation_name = experiment_value(config, "ablation_name", None)
    normalized_prefix = ablation_prefix(ablation_name)
    if normalized_prefix is not None:
        run_prefix = normalized_prefix

    return (
        f"{proj_name}_{run_prefix}{'_disc' if use_discriminator else ''}"
        f"{f'_{encoder_type}' if encoder_type != 'encoder' else ''}"
        f"_emb{emb_size}{f'_actualEmb{actual_emb_size}' if actual_emb_size != emb_size else ''}"
        f"{f'_{extra_name}' if extra_name != '' else ''}-{slurm_jobid}"
    )


def build_run_dir(config, run_name: str, output_dir: str | None = None, base_scratch_dir: str | None = None) -> str:
    if output_dir:
        return output_dir
    proj_name = experiment_value(config, "proj_name", "trait")
    base_scratch_dir = base_scratch_dir or experiment_value(config, "base_scratch_dir", "./runs")
    return os.path.join(base_scratch_dir, "checkpoints", proj_name, run_name)
