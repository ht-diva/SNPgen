"""Yelmen-style conditional WGAN-GP baseline for SNP panels.

The original implementation is an
unconditional multiscale 1D convolutional WGAN-GP for binary haplotypes.
This port preserves that block topology and adds label conditioning through
class-dependent multiscale offsets, so the public ablation API can sample
from binary disease labels without converting the model into a different
convolutional architecture.
"""

from __future__ import annotations

import math
from typing import Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn


def _batch_xy(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch["x"], batch["y"]
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return batch[0], batch[1]
    raise ValueError("Expected dict batch or (x, y) tuple")


def _as_dosage_channel(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    if x.ndim == 2:
        dosage = x.float()
    elif x.ndim == 3 and x.shape[1] == 3:
        weights = torch.arange(3, dtype=x.dtype, device=x.device).view(1, 3, 1)
        dosage = (x * weights).sum(dim=1)
    elif x.ndim == 3 and x.shape[-1] == 3:
        weights = torch.arange(3, dtype=x.dtype, device=x.device).view(1, 1, 3)
        dosage = (x * weights).sum(dim=-1)
    elif x.ndim == 3 and x.shape[1] == 1:
        dosage = x[:, 0].float()
    else:
        raise ValueError(f"Expected SNP tensor as (B,L), (B,3,L), (B,L,3), or (B,1,L), got {x.shape}")
    return dosage[:, :seq_len].unsqueeze(1).clamp(0.0, 2.0) / 2.0


def _target_model_len(seq_len: int, min_latent_size: int = 4, max_groups: int = 7) -> tuple[int, int, int]:
    if seq_len < 4:
        raise ValueError("Yelmen WGAN requires seq_len >= 4")
    if min_latent_size < 1:
        raise ValueError("min_latent_size must be >= 1")
    groups = max(2, min(max_groups, math.floor(math.log(max((seq_len + 1) / min_latent_size, 1.0), 4))))
    base = 4**groups
    latent_size = max(1, math.ceil((seq_len + 1) / base))
    model_len = latent_size * base - 1
    depth = groups * 2
    return model_len, latent_size, depth


def _pad_or_crop(x: torch.Tensor, length: int) -> torch.Tensor:
    current = x.shape[-1]
    if current == length:
        return x
    if current > length:
        return x[..., :length]
    return nn.functional.pad(x, (0, length - current))


def _raise_if_nonfinite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite value detected in WGAN-GP {name}")


class _YelmenBlock(nn.Module):
    def __init__(self, channels: int, mult: int, block_type: str, sampling: int, noise_dim: int = 0, alpha: float = 0.01):
        super().__init__()
        if block_type == "g" and sampling == 1:
            self.block = nn.Sequential(
                nn.Conv1d(channels * mult + noise_dim + 1, channels * (mult - 2), 3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(channels * (mult - 2)),
                nn.LeakyReLU(alpha),
                nn.ConvTranspose1d(channels * (mult - 2), channels * (mult - 4), 3, stride=2, padding=0, bias=False),
                nn.BatchNorm1d(channels * (mult - 4)),
                nn.LeakyReLU(alpha),
                nn.Conv1d(channels * (mult - 4), channels * (mult - 6), 3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(channels * (mult - 6)),
                nn.LeakyReLU(alpha),
                nn.ConvTranspose1d(channels * (mult - 6), channels * (mult - 8), 3, stride=2, padding=0, bias=False),
                nn.BatchNorm1d(channels * (mult - 8)),
                nn.LeakyReLU(alpha),
            )
        elif block_type == "d" and sampling == -1:
            self.block = nn.Sequential(
                nn.Conv1d(channels * mult + 1, channels * (mult + 2), 3, stride=1, padding=1),
                nn.InstanceNorm1d(channels * (mult + 2), affine=True),
                nn.LeakyReLU(alpha),
                nn.Conv1d(channels * (mult + 2), channels * (mult + 4), 3, stride=2, padding=0),
                nn.InstanceNorm1d(channels * (mult + 4), affine=True),
                nn.LeakyReLU(alpha),
                nn.Conv1d(channels * (mult + 4), channels * (mult + 6), 3, stride=1, padding=1),
                nn.InstanceNorm1d(channels * (mult + 6), affine=True),
                nn.LeakyReLU(alpha),
                nn.Conv1d(channels * (mult + 6), channels * (mult + 8), 3, stride=2, padding=0),
                nn.InstanceNorm1d(channels * (mult + 8), affine=True),
                nn.LeakyReLU(alpha),
            )
        elif block_type == "d" and sampling == 0:
            self.block = nn.Sequential(
                nn.Conv1d(channels * mult, channels * mult, 3, stride=1, padding=1),
                nn.InstanceNorm1d(channels * mult, affine=True),
                nn.LeakyReLU(alpha),
                nn.Conv1d(channels * mult, channels * mult, 3, stride=1, padding=1),
                nn.InstanceNorm1d(channels * mult, affine=True),
                nn.LeakyReLU(alpha),
            )
        elif block_type == "g" and sampling == 0:
            self.block = nn.Sequential(
                nn.Conv1d(channels * mult, channels * mult, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(channels * mult),
                nn.LeakyReLU(alpha),
                nn.Conv1d(channels * mult, channels * mult, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(channels * mult),
                nn.LeakyReLU(alpha),
            )
        else:
            raise ValueError(f"Invalid block_type={block_type!r}, sampling={sampling!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _ConditionalOffset(nn.Module):
    def __init__(self, n_classes: int, *shape: int):
        super().__init__()
        self.base = nn.Parameter(torch.normal(mean=0.0, std=1.0, size=(1, *shape)))
        self.by_label = nn.Embedding(n_classes, math.prod(shape))
        nn.init.zeros_(self.by_label.weight)
        self.shape = shape

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        label = self.by_label(y.reshape(-1).long()).reshape(y.shape[0], *self.shape)
        return self.base.expand(y.shape[0], *self.shape) + label


class _UnconditionalOffset(nn.Module):
    """The trainable location vector used by the original Yelmen WGAN."""

    def __init__(self, *shape: int):
        super().__init__()
        self.base = nn.Parameter(torch.normal(mean=0.0, std=1.0, size=(1, *shape)))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.base.expand(y.shape[0], *self.base.shape[1:])


class _BinaryScalarOffset(_UnconditionalOffset):
    """Minimal phenotype extension: one learned broadcast scalar per scale."""

    def __init__(self, *shape: int):
        super().__init__(*shape)
        self.label_delta = nn.Parameter(torch.zeros(()))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        base = super().forward(y)
        label = y.reshape(-1, *([1] * (base.ndim - 1))).to(base.dtype)
        return base + label * self.label_delta


def _offset(mode: str, n_classes: int, *shape: int) -> nn.Module:
    if mode == "multiscale_offsets":
        return _ConditionalOffset(n_classes, *shape)
    if mode == "binary_scalar":
        return _BinaryScalarOffset(*shape)
    if mode == "none":
        return _UnconditionalOffset(*shape)
    raise ValueError(f"Unknown conditioning_mode={mode!r}")


class YelmenConvGenerator(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_classes: int = 2,
        channels: int = 10,
        noise_dim: int = 2,
        alpha: float = 0.01,
        min_latent_size: int = 4,
        conditioning_mode: str = "multiscale_offsets",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.model_len, self.latent_size, self.depth = _target_model_len(seq_len, min_latent_size=min_latent_size)
        self.groups = self.depth // 2
        self.channels = channels
        self.noise_dim = noise_dim

        top_mult = 4 + 8 * (self.groups - 1)
        self.input_offsets = nn.ModuleList()
        self.input_offsets.append(_offset(conditioning_mode, n_classes, 1, self.latent_size))

        for i in range(2, self.depth + 1, 2):
            self.input_offsets.append(_offset(conditioning_mode, n_classes, 1, self.latent_size * (2**i) - 1))

        self.noise_offsets = nn.ModuleList(
            _offset("multiscale_offsets" if conditioning_mode == "multiscale_offsets" else "none", n_classes, noise_dim, self.latent_size * (2**i) - 1)
            for i in range(2, self.depth, 2)
        )
        self.initial_noise_offset = _offset(
            "multiscale_offsets" if conditioning_mode == "multiscale_offsets" else "none",
            n_classes,
            noise_dim,
            self.latent_size,
        )

        self.block1 = nn.Sequential(
            nn.Conv1d(noise_dim + 1, channels * top_mult, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(channels * top_mult),
            nn.LeakyReLU(alpha),
            nn.ConvTranspose1d(channels * top_mult, channels * (top_mult - 2), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(channels * (top_mult - 2)),
            nn.LeakyReLU(alpha),
            nn.Conv1d(channels * (top_mult - 2), channels * (top_mult - 4), 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(channels * (top_mult - 4)),
            nn.LeakyReLU(alpha),
            nn.ConvTranspose1d(channels * (top_mult - 4), channels * (top_mult - 6), 3, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(channels * (top_mult - 6)),
            nn.LeakyReLU(alpha),
        )

        self.sample_blocks = nn.ModuleList()
        mult = top_mult - 6
        while mult > 6:
            self.sample_blocks.append(_YelmenBlock(channels, mult, block_type="g", sampling=1, noise_dim=noise_dim, alpha=alpha))
            mult -= 8

        self.res_blocks = nn.ModuleDict({
            str(i): _YelmenBlock(channels, block_out_mult, block_type="g", sampling=0, alpha=alpha)
            for i, block_out_mult in enumerate(range(top_mult - 14, 5, -8), start=1)
            if i % 2 == 1
        })
        self.final_block = nn.Sequential(
            nn.Conv1d(channels * 6 + noise_dim + 1, channels * 4, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(channels * 4),
            nn.LeakyReLU(alpha),
            nn.ConvTranspose1d(channels * 4, channels * 2, 3, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(channels * 2),
            nn.LeakyReLU(alpha),
            nn.Conv1d(channels * 2, channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(alpha),
            nn.ConvTranspose1d(channels, channels // 2, 3, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(channels // 2),
            nn.LeakyReLU(alpha),
        )
        self.output_block = nn.Sequential(
            nn.Conv1d(channels // 2 + 1, 1, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def sample_noise(self, batch_size: int, y: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor]]:
        z = torch.randn(batch_size, self.noise_dim, self.latent_size, device=device)
        z = z + self.initial_noise_offset(y).to(device=device, dtype=z.dtype)
        noise_list = []
        for i, offset in enumerate(self.noise_offsets, start=1):
            length = self.latent_size * (2 ** (2 * i)) - 1
            noise = torch.randn(batch_size, self.noise_dim, length, device=device)
            noise_list.append(noise + offset(y).to(device=device, dtype=noise.dtype))
        return z, noise_list

    def forward(self, z: torch.Tensor, y: torch.Tensor, noise_list: list[torch.Tensor]) -> torch.Tensor:
        x = torch.cat((self.input_offsets[0](y).to(device=z.device, dtype=z.dtype), z), dim=1)
        x = self.block1(x)

        for i, block in enumerate(self.sample_blocks, start=1):
            x = torch.cat((self.input_offsets[i](y).to(device=x.device, dtype=x.dtype), noise_list[i - 1], x), dim=1)
            x = block(x)
            key = str(i)
            if key in self.res_blocks:
                x = x + self.res_blocks[key](x)

        final_offset_idx = len(self.sample_blocks) + 1
        x = torch.cat((self.input_offsets[final_offset_idx](y).to(device=x.device, dtype=x.dtype), noise_list[-1], x), dim=1)
        x = self.final_block(x)
        x = torch.cat((self.input_offsets[final_offset_idx + 1](y).to(device=x.device, dtype=x.dtype), x), dim=1)
        return self.output_block(x)


class YelmenConvCritic(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_classes: int = 2,
        pack_m: int = 1,
        channels: int = 10,
        alpha: float = 0.01,
        min_latent_size: int = 4,
        conditioning_mode: str = "multiscale_offsets",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.model_len, self.latent_size, self.depth = _target_model_len(seq_len, min_latent_size=min_latent_size)
        self.groups = self.depth // 2
        self.channels = channels
        self.pack_m = pack_m

        top_mult = 4 + 8 * (self.groups - 1)
        self.input_offsets = nn.ModuleList(
            _offset(conditioning_mode, n_classes, 1, self.latent_size * (2**i) - 1)
            for i in range(self.depth, 1, -2)
        )

        self.block1 = nn.Sequential(
            nn.Conv1d(pack_m + 1, channels, 3, stride=1, padding=1),
            nn.InstanceNorm1d(channels, affine=True),
            nn.LeakyReLU(alpha),
            nn.Conv1d(channels, channels * 2, 3, stride=2, padding=0),
            nn.InstanceNorm1d(channels * 2, affine=True),
            nn.LeakyReLU(alpha),
            nn.Conv1d(channels * 2, channels * 4, 3, stride=1, padding=1),
            nn.InstanceNorm1d(channels * 4, affine=True),
            nn.LeakyReLU(alpha),
            nn.Conv1d(channels * 4, channels * 6, 3, stride=2, padding=0),
            nn.InstanceNorm1d(channels * 6, affine=True),
            nn.LeakyReLU(alpha),
        )

        self.sample_blocks = nn.ModuleList()
        mult = 6
        while mult < top_mult - 6:
            self.sample_blocks.append(_YelmenBlock(channels, mult, block_type="d", sampling=-1, alpha=alpha))
            mult += 8
        self.res_blocks = nn.ModuleDict({
            str(i): _YelmenBlock(channels, block_in_mult, block_type="d", sampling=0, alpha=alpha)
            for i, block_in_mult in enumerate(range(6, top_mult - 6, 8), start=1)
            if i % 2 == 1
        })
        final_norm = nn.Identity if self.latent_size == 1 else lambda ch: nn.InstanceNorm1d(ch, affine=True)

        self.final_block = nn.Sequential(
            nn.Conv1d(channels * (top_mult - 6) + 1, channels * (top_mult - 4), 3, stride=1, padding=1),
            final_norm(channels * (top_mult - 4)),
            nn.LeakyReLU(alpha),
            nn.Conv1d(channels * (top_mult - 4), channels * (top_mult - 2), 3, stride=2, padding=0),
            final_norm(channels * (top_mult - 2)),
            nn.LeakyReLU(alpha),
            nn.Conv1d(channels * (top_mult - 2), channels * top_mult, 3, stride=1, padding=1),
            final_norm(channels * top_mult),
            nn.LeakyReLU(alpha),
            nn.Conv1d(channels * top_mult, 1, 3, stride=2, padding=1),
            final_norm(1),
            nn.LeakyReLU(alpha),
            nn.Flatten(),
            nn.Linear(self.latent_size, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = _pad_or_crop(x, self.model_len)
        x = torch.cat((self.input_offsets[0](y).to(device=x.device, dtype=x.dtype), x), dim=1)
        x = self.block1(x)

        for i, block in enumerate(self.sample_blocks, start=1):
            key = str(i)
            if key in self.res_blocks:
                x = x + self.res_blocks[key](x)
            x = torch.cat((self.input_offsets[i](y).to(device=x.device, dtype=x.dtype), x), dim=1)
            x = block(x)

        x = torch.cat((self.input_offsets[len(self.sample_blocks) + 1](y).to(device=x.device, dtype=x.dtype), x), dim=1)

        return self.final_block(x).reshape(-1)


class ConditionalWGAN_GPModule(pl.LightningModule):
    """Lightning wrapper around the Yelmen multiscale WGAN-GP architecture."""

    def __init__(
        self,
        seq_len: int,
        latent_dim: int | None = None,
        n_classes: int = 2,
        generator_channels: int | None = None,
        critic_channels: int | None = None,
        channels: int = 10,
        noise_dim: int = 2,
        alpha: float = 0.01,
        min_latent_size: int = 4,
        pack_m: int = 1,
        generator_lr: float = 5.0e-4,
        critic_lr: float = 5.0e-4,
        betas: Tuple[float, float] = (0.5, 0.9),
        gradient_penalty_weight: float = 10.0,
        critic_steps: int = 10,
        condition_on_label: bool = True,
        conditioning_mode: str = "multiscale_offsets",
        mask_padded_tail: bool = False,
    ):
        super().__init__()
        if latent_dim is not None and latent_dim != noise_dim:
            noise_dim = latent_dim
        if generator_channels is not None:
            channels = generator_channels
        if channels < 2:
            raise ValueError("Yelmen WGAN requires channels >= 2 because the final block uses channels // 2.")
        if not condition_on_label:
            conditioning_mode = "none"
        elif conditioning_mode == "none":
            raise ValueError("conditioning_mode='none' requires condition_on_label=False")
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generator = YelmenConvGenerator(
            seq_len=seq_len,
            n_classes=n_classes,
            channels=channels,
            noise_dim=noise_dim,
            alpha=alpha,
            min_latent_size=min_latent_size,
            conditioning_mode=conditioning_mode,
        )
        self.critic = YelmenConvCritic(
            seq_len=seq_len,
            n_classes=n_classes,
            pack_m=pack_m,
            channels=channels,
            alpha=alpha,
            min_latent_size=min_latent_size,
            conditioning_mode=conditioning_mode,
        )
        self.register_buffer("_critic_update_count", torch.zeros((), dtype=torch.long), persistent=True)

    def _sample_generator(self, y: torch.Tensor) -> torch.Tensor:
        z, noise_list = self.generator.sample_noise(y.shape[0], y, self.device)
        fake = self.generator(z, y, noise_list)
        if bool(self.hparams.mask_padded_tail) and self.generator.model_len > int(self.hparams.seq_len):
            mask = torch.arange(fake.shape[-1], device=fake.device) < int(self.hparams.seq_len)
            fake = fake * mask.reshape(1, 1, -1).to(fake.dtype)
        return fake

    def _pack_real_by_label(
        self,
        real: torch.Tensor,
        y: torch.Tensor,
        allow_replacement: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pack_m = int(self.hparams.pack_m)
        if pack_m == 1:
            return real, y

        if not bool(self.hparams.condition_on_label):
            usable = (real.shape[0] // pack_m) * pack_m
            if usable == 0:
                raise ValueError(f"pack_m={pack_m} requires at least {pack_m} samples per batch")
            packed = real[:usable, 0, :].reshape(-1, pack_m, real.shape[-1])
            return packed, torch.zeros(packed.shape[0], dtype=torch.long, device=real.device)

        packed = []
        labels = []
        for label in torch.unique(y, sorted=True):
            idx = torch.nonzero(y == label, as_tuple=False).flatten()
            usable = (idx.numel() // pack_m) * pack_m
            if usable == 0 and allow_replacement and idx.numel() > 0:
                choice = idx[torch.randint(idx.numel(), (pack_m,), device=idx.device)]
                grouped = real[choice, 0, :].reshape(1, pack_m, real.shape[-1])
                packed.append(grouped)
                labels.append(label.expand(grouped.shape[0]))
                continue
            if usable == 0:
                continue
            grouped = real[idx[:usable], 0, :].reshape(-1, pack_m, real.shape[-1])
            packed.append(grouped)
            labels.append(label.expand(grouped.shape[0]))

        if not packed:
            raise ValueError(
                f"pack_m={pack_m} requires at least {pack_m} samples from one label in each training batch. "
                "Increase batch_size or reduce pack_m."
            )
        return torch.cat(packed, dim=0), torch.cat(labels, dim=0)

    def _sample_packed_generator(self, y: torch.Tensor) -> torch.Tensor:
        pack_m = int(self.hparams.pack_m)
        if pack_m == 1:
            return self._sample_generator(y)
        repeated_y = y.repeat_interleave(pack_m)
        fake = self._sample_generator(repeated_y)
        return fake[:, 0, :].reshape(y.shape[0], pack_m, fake.shape[-1])

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        alpha = torch.rand(real.shape[0], 1, 1, device=real.device)
        interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
        score = self.critic(interpolated, y)
        gradients = torch.autograd.grad(
            outputs=score,
            inputs=interpolated,
            grad_outputs=torch.ones_like(score),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0].flatten(1)
        return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()

    def training_step(self, batch, batch_idx):
        x, y = _batch_xy(batch)
        y = y.reshape(-1).long().to(self.device)
        real = _pad_or_crop(_as_dosage_channel(x, self.hparams.seq_len).to(self.device), self.generator.model_len)
        real, packed_y = self._pack_real_by_label(real, y)

        opt_g, opt_c = self.optimizers()

        fake = self._sample_packed_generator(packed_y)
        _raise_if_nonfinite("fake samples", fake)
        critic_real = self.critic(real, packed_y).mean()
        critic_fake = self.critic(fake.detach(), packed_y).mean()
        gp = self._gradient_penalty(real, fake.detach(), packed_y)
        critic_loss = critic_fake - critic_real + self.hparams.gradient_penalty_weight * gp
        _raise_if_nonfinite("critic loss", critic_loss)

        opt_c.zero_grad(set_to_none=True)
        self.manual_backward(critic_loss)
        opt_c.step()
        self._critic_update_count.add_(1)

        generator_loss = torch.zeros((), device=self.device)
        if int(self._critic_update_count.item()) % int(self.hparams.critic_steps) == 0:
            fake = self._sample_packed_generator(packed_y)
            _raise_if_nonfinite("generator samples", fake)
            generator_loss = -self.critic(fake, packed_y).mean()
            _raise_if_nonfinite("generator loss", generator_loss)
            opt_g.zero_grad(set_to_none=True)
            self.manual_backward(generator_loss)
            opt_g.step()

        self.log_dict(
            {
                "train/loss/critic": critic_loss.detach(),
                "train/loss/generator": generator_loss.detach(),
                "train/scalars/gp": gp.detach(),
                "train/scalars/critic_real": critic_real.detach(),
                "train/scalars/critic_fake": critic_fake.detach(),
            },
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )

    def validation_step(self, batch, batch_idx):
        x, y = _batch_xy(batch)
        y = y.reshape(-1).long().to(self.device)
        real = _pad_or_crop(_as_dosage_channel(x, self.hparams.seq_len).to(self.device), self.generator.model_len)
        real, packed_y = self._pack_real_by_label(real, y, allow_replacement=True)
        fake = self._sample_packed_generator(packed_y)
        _raise_if_nonfinite("validation fake samples", fake)

        critic_real = self.critic(real, packed_y).mean()
        critic_fake = self.critic(fake, packed_y).mean()
        generator_loss = -critic_fake
        distribution_gap = torch.abs(critic_real - critic_fake)
        self.log_dict(
            {
                "val/loss": generator_loss.detach(),
                "val/loss/generator": generator_loss.detach(),
                "val/scalars/critic_real": critic_real.detach(),
                "val/scalars/critic_fake": critic_fake.detach(),
                "val/scalars/distribution_gap": distribution_gap.detach(),
            },
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return generator_loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.generator_lr, betas=self.hparams.betas)
        opt_c = torch.optim.Adam(self.critic.parameters(), lr=self.hparams.critic_lr, betas=self.hparams.betas)
        return [opt_g, opt_c]

    @torch.no_grad()
    def sample(
        self,
        labels: torch.Tensor | None = None,
        argmax: bool = True,
        num_samples: int | None = None,
    ) -> torch.Tensor:
        if labels is None:
            if self.hparams.condition_on_label:
                raise ValueError("labels are required when condition_on_label=True")
            if num_samples is None:
                raise ValueError("num_samples is required when labels=None")
            labels = torch.zeros(int(num_samples), dtype=torch.long, device=self.device)
        else:
            labels = labels.reshape(-1).long().to(self.device)
        values = self._sample_generator(labels)[..., : self.hparams.seq_len]
        _raise_if_nonfinite("sample output", values)
        if argmax:
            return torch.round(values[:, 0] * 2.0).clamp(0, 2).long()
        probs = values[:, 0].clamp(0.0, 1.0)
        return torch.stack(
            (
                (1.0 - 2.0 * probs).clamp_min(0.0),
                (1.0 - (2.0 * probs - 1.0).abs()).clamp_min(0.0),
                (2.0 * probs - 1.0).clamp_min(0.0),
            ),
            dim=1,
        )
