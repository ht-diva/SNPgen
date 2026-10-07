"""Conditional VAE baseline for SNPgen ablations.

This module keeps the original SNPgen encoder/decoder architecture intact and
adds explicit label-conditioning adapters around it.  The conditioning follows
the standard cVAE pattern: concatenate the class code with the encoder input and
with the decoder latent input.  The adapters project those concatenated tensors
back to the original channel counts, so the core VAE body can remain identical
to the VAE used by the VAE+DDPM pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from snpgen.models.modules.utils import get_proper_state_dict_vae
from snpgen.training.engine.vae import AutoencoderDiscriminatorTrainingWrapper, AutoencoderTrainingWrapper
from snpgen.utils import instantiate_from_config


class ConditionalAutoencoder(nn.Module):
    """VAE with phenotype-conditioned decoding."""

    def __init__(
        self,
        encoder_config: Dict[str, Any],
        decoder_config: Dict[str, Any],
        n_classes: int = 2,
        input_ch: int = 3,
        z_channels: int = 1,
        z_dim: int = 128,
        condition_encoder: bool = True,
        condition_decoder: bool = True,
        ckpt_path: str = None,
        load_ema_ckpt: bool = True,
    ):
        super().__init__()
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.n_classes = n_classes
        self.input_ch = input_ch
        self.z_channels = z_channels
        self.z_dim = z_dim
        self.condition_encoder = condition_encoder
        self.condition_decoder = condition_decoder

        self.encoder = instantiate_from_config(encoder_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.encoder_condition_projection = (
            nn.Conv1d(input_ch + n_classes, input_ch, kernel_size=1)
            if condition_encoder
            else nn.Identity()
        )
        self.decoder_condition_projection = (
            nn.Conv1d(z_channels + n_classes, z_channels, kernel_size=1)
            if condition_decoder
            else nn.Identity()
        )

        if ckpt_path is not None:
            state_dict = get_proper_state_dict_vae(ckpt_path, ema=load_ema_ckpt)
            self.load_state_dict(state_dict, strict=False)

    def get_cls_layer(self):
        return None

    def get_last_layer(self):
        return self.decoder.get_last_layer()

    def _label_channels(self, y: torch.Tensor, length: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        y = y.reshape(-1).long()
        y_onehot = F.one_hot(y, num_classes=self.n_classes).to(dtype=dtype, device=device)
        return y_onehot[:, :, None].expand(-1, -1, length)

    def _condition_input(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self.condition_encoder:
            return x
        cond = self._label_channels(y, x.shape[-1], x.dtype, x.device)
        return self.encoder_condition_projection(torch.cat([x, cond], dim=1))

    def _condition_latent(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self.condition_decoder:
            return z
        cond = self._label_channels(y, z.shape[-1], z.dtype, z.device)
        return self.decoder_condition_projection(torch.cat([z, cond], dim=1))

    def encode(self, x: torch.Tensor, y: torch.Tensor = None, sample: bool = False):
        if self.condition_encoder:
            if y is None:
                raise ValueError("Conditional encoder requires phenotype labels")
            x = self._condition_input(x, y)
        mu, logvar = self.encoder(x)
        if sample:
            return self.reparameterize(mu, logvar)
        return mu, logvar

    def decode(self, z: torch.Tensor, y: torch.Tensor, argmax: bool = True):
        return self.decoder(self._condition_latent(z, y), argmax=argmax)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, z: torch.Tensor, y: torch.Tensor, argmax: bool = True):
        return self.decode(z, y, argmax=argmax)

    def forward_recons(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        argmax: bool = True,
        sample_posterior: bool = True,
    ):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar) if sample_posterior else mu
        return self.decode(z, y, argmax=argmax)

    def sample(self, labels: torch.Tensor, argmax: bool = True):
        labels = labels.reshape(-1).long()
        z = torch.randn(
            labels.shape[0],
            self.z_channels,
            self.z_dim,
            device=next(self.parameters()).device,
        )
        return self.decode(z, labels, argmax=argmax)


class _ConditionalAutoencoderMixin:
    """Mixin that routes batch labels through the conditional autoencoder."""

    def get_batch_xy(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]
        if isinstance(batch, dict):
            return batch["x"], batch["y"]
        raise ValueError("Conditional VAE batches must be dicts or (x, y) tuples")

    def inner_training_step(self, batch, batch_idx, loss_kwargs=None):
        loss_kwargs = {} if loss_kwargs is None else dict(loss_kwargs)
        x, y = self.get_batch_xy(batch)
        x, target = self.check_shape(x)

        mu, logvar = self.autoencoder.encode(x, y)
        z = self.autoencoder.reparameterize(mu, logvar)
        reconstructions = self.autoencoder.decode(z, y, argmax=False)

        extra_info = {
            "mean": mu,
            "logvar": logvar,
            "split": "train",
            "channel_first": self.channel_first,
            "oh_inputs": self.get_onehot_input(x),
        }
        loss_kwargs.update(extra_info)
        loss, log_dict = self.loss(target, reconstructions, **loss_kwargs)
        log_dict.update(self.compute_recons_metrics(target, reconstructions, split="train", return_dict=True))
        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def inner_validation_step(self, batch, batch_idx, loss_kwargs=None):
        loss_kwargs = {} if loss_kwargs is None else dict(loss_kwargs)
        x, y = self.get_batch_xy(batch)
        x, target = self.check_shape(x)

        mu, logvar = self.autoencoder.encode(x, y)
        z = self.autoencoder.reparameterize(mu, logvar)
        reconstructions = self.autoencoder.decode(z, y, argmax=False)

        extra_info = {
            "mean": mu,
            "logvar": logvar,
            "split": "val",
            "channel_first": self.channel_first,
            "oh_inputs": self.get_onehot_input(x),
        }
        loss_kwargs.update(extra_info)
        loss, log_dict = self.loss(target, reconstructions, **loss_kwargs)
        recons_metrics_log = self.compute_recons_metrics(target, reconstructions, split="val", return_dict=True)
        log_dict.update(recons_metrics_log)
        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss


class ConditionalAutoencoderTrainingWrapper(_ConditionalAutoencoderMixin, AutoencoderTrainingWrapper):
    """Conditional VAE wrapper without adversarial discriminator."""


class ConditionalAutoencoderDiscriminatorTrainingWrapper(_ConditionalAutoencoderMixin, AutoencoderDiscriminatorTrainingWrapper):
    """Conditional VAE wrapper with the same discriminator flow as the paper VAE."""
