"""Yelmen-style two-stage RBM/CRBM cascade for SNP ablations.

Yelmen et al. generate long sequences by training a standard RBM on the first
block and a CRBM on an overlapping next block whose prefix is pinned.  This
port keeps that cascade and adapts it to short disease panels: stage 0 samples
the first half conditioned on the binary disease label, then stage 1 pins the
label plus the generated first half and samples the second half.
"""

from __future__ import annotations

import torch
from torch import nn
import lightning.pytorch as pl


def _batch_xy(batch):
    if isinstance(batch, dict):
        return batch["x"], batch["y"]
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return batch[0], batch[1]
    raise TypeError(f"Expected dict batch or (x, y) tuple, got {type(batch).__name__}")


def _as_dosage(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    if x.ndim == 3 and x.shape[1] == 3:
        values = torch.arange(3, device=x.device, dtype=x.dtype).view(1, 3, 1)
        dosage = (x * values).sum(dim=1)
    elif x.ndim == 3 and x.shape[-1] == 3:
        values = torch.arange(3, device=x.device, dtype=x.dtype).view(1, 1, 3)
        dosage = (x * values).sum(dim=-1)
    elif x.ndim == 3 and x.shape[1] == 1:
        dosage = x[:, 0]
    elif x.ndim == 2:
        dosage = x
    else:
        raise ValueError(f"Unsupported SNP tensor shape for CRBM: {tuple(x.shape)}")
    return dosage[:, :seq_len].round().clamp(0, 2).long()


def _dosage_to_allele_bits(dosage: torch.Tensor) -> torch.Tensor:
    allele_1 = (dosage >= 1).float()
    allele_2 = (dosage >= 2).float()
    return torch.stack((allele_1, allele_2), dim=-1).reshape(dosage.shape[0], -1)


def _dosage_to_random_phase_haplotypes(dosage: torch.Tensor) -> torch.Tensor:
    """Pseudo-phase B diploid dosages into 2B exchangeable binary haplotypes."""
    dosage = dosage.long()
    heterozygous = dosage == 1
    first = (dosage == 2).float()
    phase = torch.randint(0, 2, dosage.shape, device=dosage.device).float()
    first = torch.where(heterozygous, phase, first)
    second = dosage.float() - first
    return torch.stack((first, second), dim=1).reshape(-1, dosage.shape[1])


def _allele_bits_to_dosage(bits: torch.Tensor, seq_len: int) -> torch.Tensor:
    n_samples = bits.shape[1]
    return bits.t().reshape(n_samples, seq_len, 2).sum(dim=-1).clamp(0, 2).long()


class _YelmenBinaryRBM(nn.Module):
    """Binary RBM block with the original centered OOE update rule."""

    def __init__(
        self,
        num_visible: int,
        num_hidden: int,
        fixed_nodes: int,
        lr: float,
        gibbs_steps: int,
        reg_l2: float,
        var_init: float,
        reset_perm_chain_batch: bool,
        upd_centered: bool,
        num_pcd: int | None = None,
    ):
        super().__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden)
        self.fixed_nodes = int(fixed_nodes)
        self.lr = float(lr)
        self.gibbs_steps = int(gibbs_steps)
        self.reg_l2 = float(reg_l2)
        self.reset_perm_chain_batch = bool(reset_perm_chain_batch)
        self.upd_centered = bool(upd_centered)
        self.num_pcd = None if num_pcd is None else int(num_pcd)

        self.register_buffer("W", torch.randn(num_hidden, num_visible) * var_init)
        self.register_buffer("W2", torch.zeros(num_hidden, num_visible))
        self.register_buffer("vbias", torch.zeros(num_visible))
        self.register_buffer("hbias", torch.zeros(num_hidden))
        self.register_buffer("X_pc", torch.empty(num_visible, 0))
        self.register_buffer("vis_data_av", torch.zeros(num_visible))
        self.register_buffer("hid_data_av", torch.zeros(num_hidden))

    @torch.no_grad()
    def set_visible_bias_from_mean(self, mean: torch.Tensor) -> None:
        prob1 = mean.clamp(1e-5, 1 - 1e-5).to(self.vbias.device)
        self.vbias.copy_(-torch.log(1.0 / prob1 - 1.0))
        self.vis_data_av.copy_(prob1)

    def sample_hiddens(self, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mh = torch.sigmoid((self.W.mm(V).t() + self.hbias).t())
        return torch.bernoulli(mh), mh

    def sample_visibles(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mv = torch.sigmoid((self.W.t().mm(H).t() + self.vbias).t())
        return torch.bernoulli(mv), mv

    def gibbs(self, V: torch.Tensor, steps: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        steps = int(steps or self.gibbs_steps)
        h, mh = self.sample_hiddens(V)
        v_tmp, mv = self.sample_visibles(h)
        v_tmp[: self.fixed_nodes, :] = V[: self.fixed_nodes, :]
        mv[: self.fixed_nodes, :] = V[: self.fixed_nodes, :]

        for _ in range(1, steps):
            h, mh = self.sample_hiddens(v_tmp)
            v_tmp, mv = self.sample_visibles(h)
            v_tmp[: self.fixed_nodes, :] = V[: self.fixed_nodes, :]
            mv[: self.fixed_nodes, :] = V[: self.fixed_nodes, :]
        return v_tmp, mv, h, mh

    @torch.no_grad()
    def fit_batch(self, v_pos: torch.Tensor) -> torch.Tensor:
        batch_size = v_pos.shape[1]
        if self.reset_perm_chain_batch:
            neg_chains = batch_size if self.num_pcd is None else self.num_pcd
            self.X_pc = torch.bernoulli(torch.rand(self.num_visible, neg_chains, device=v_pos.device))
        elif self.X_pc.shape[1] == 0:
            neg_chains = batch_size if self.num_pcd is None else self.num_pcd
            self.X_pc = torch.bernoulli(torch.rand(self.num_visible, neg_chains, device=v_pos.device))

        self._set_fixed_nodes_from_batch(self.X_pc, v_pos)

        _h_pos, h_pos_m = self.sample_hiddens(v_pos)
        v_neg, _mv_neg, _h_neg, h_neg_m = self.gibbs(self.X_pc, self.gibbs_steps)
        self.X_pc = v_neg.detach()

        if self.upd_centered:
            self._update_centered(v_pos, h_pos_m, v_neg, h_neg_m)
        else:
            self._update_uncentered(v_pos, h_pos_m, v_neg, h_neg_m)

        return torch.mean((v_pos[self.fixed_nodes :] - v_neg[self.fixed_nodes :]) ** 2)

    @torch.no_grad()
    def _set_fixed_nodes_from_batch(self, target: torch.Tensor, source: torch.Tensor) -> None:
        if self.fixed_nodes == 0:
            return
        if target.shape[1] == source.shape[1]:
            target[: self.fixed_nodes, :] = source[: self.fixed_nodes, :]
            return
        index = torch.randint(source.shape[1], (target.shape[1],), device=source.device)
        target[: self.fixed_nodes, :] = source[: self.fixed_nodes, index]

    @torch.no_grad()
    def _update_centered(
        self,
        v_pos: torch.Tensor,
        h_pos_m: torch.Tensor,
        v_neg: torch.Tensor,
        h_neg_m: torch.Tensor,
    ) -> None:
        pos_norm = 1.0 / float(v_pos.shape[1])
        neg_norm = 1.0 / float(v_neg.shape[1])
        self.vis_data_av.copy_(torch.mean(v_pos, dim=1))
        self.hid_data_av.copy_(torch.mean(h_pos_m, dim=1))

        x_pos = (v_pos.t() - self.vis_data_av).t()
        h_pos = (h_pos_m.t() - self.hid_data_av).t()
        x_neg = (v_neg.t() - self.vis_data_av).t()
        h_neg = (h_neg_m.t() - self.hid_data_av).t()

        delta_w = (
            h_pos.mm(x_pos.t()) * pos_norm
            - h_neg.mm(x_neg.t()) * neg_norm
            - 2.0 * self.reg_l2 * self.W
        )
        self.W.add_(self.lr * delta_w)

        delta_vbias = (
            torch.sum(v_pos, dim=1) * pos_norm
            - torch.sum(v_neg, dim=1) * neg_norm
            - torch.mv(delta_w.t(), self.hid_data_av)
            - 2.0 * self.reg_l2 * self.vbias
        )
        self.vbias.add_(self.lr * delta_vbias)

        delta_hbias = (
            torch.sum(h_pos_m, dim=1) * pos_norm
            - torch.sum(h_neg_m, dim=1) * neg_norm
            - torch.mv(delta_w, self.vis_data_av)
            - 2.0 * self.reg_l2 * self.hbias
        )
        self.hbias.add_(self.lr * delta_hbias)
        self.W2.copy_(torch.square(self.W))

    @torch.no_grad()
    def _update_uncentered(
        self,
        v_pos: torch.Tensor,
        h_pos_m: torch.Tensor,
        v_neg: torch.Tensor,
        h_neg_m: torch.Tensor,
    ) -> None:
        pos_norm = 1.0 / float(v_pos.shape[1])
        neg_norm = 1.0 / float(v_neg.shape[1])
        self.W.add_(
            self.lr * (h_pos_m.mm(v_pos.t()) * pos_norm - h_neg_m.mm(v_neg.t()) * neg_norm)
            - 2.0 * self.lr * self.reg_l2 * self.W
        )
        self.vbias.add_(self.lr * (torch.sum(v_pos, dim=1) * pos_norm - torch.sum(v_neg, dim=1) * neg_norm))
        self.hbias.add_(self.lr * (torch.sum(h_pos_m, dim=1) * pos_norm - torch.sum(h_neg_m, dim=1) * neg_norm))
        self.W2.copy_(torch.square(self.W))


class ConditionalCRBMModule(pl.LightningModule):
    """Two-stage label-conditioned RBM/CRBM cascade."""

    def __init__(
        self,
        seq_len: int,
        n_classes: int = 2,
        num_hidden: int = 1000,
        lr: float = 0.001,
        gibbs_steps: int = 50,
        generation_gibbs_steps: int = 50,
        reg_l2: float = 0.0,
        var_init: float = 1e-4,
        reset_perm_chain_batch: bool = True,
        upd_centered: bool = True,
        num_pcd: int | None = None,
        first_segment_len: int | None = None,
        condition_on_label: bool = True,
        genotype_encoding: str = "legacy_dosage_bits",
    ):
        super().__init__()
        if n_classes != 2:
            raise ValueError("ConditionalCRBMModule currently supports binary labels only.")
        if seq_len < 2:
            raise ValueError("ConditionalCRBMModule requires seq_len >= 2.")
        if genotype_encoding not in {"legacy_dosage_bits", "random_phase_haplotypes"}:
            raise ValueError(
                "genotype_encoding must be 'legacy_dosage_bits' or "
                f"'random_phase_haplotypes', got {genotype_encoding!r}."
            )
        first_segment_len = int(first_segment_len or (seq_len // 2))
        if first_segment_len <= 0 or first_segment_len >= seq_len:
            raise ValueError("first_segment_len must be in [1, seq_len - 1].")

        self.save_hyperparameters()
        self.automatic_optimization = False
        self.label_bits = int(condition_on_label)
        self.units_per_snp = 2 if genotype_encoding == "legacy_dosage_bits" else 1
        self.first_segment_len = first_segment_len
        self.second_segment_len = int(seq_len - first_segment_len)

        common = dict(
            num_hidden=num_hidden,
            lr=lr,
            gibbs_steps=gibbs_steps,
            reg_l2=reg_l2,
            var_init=var_init,
            reset_perm_chain_batch=reset_perm_chain_batch,
            upd_centered=upd_centered,
            num_pcd=num_pcd,
        )
        self.stage0 = _YelmenBinaryRBM(
            num_visible=self.label_bits + self.units_per_snp * self.first_segment_len,
            fixed_nodes=self.label_bits,
            **common,
        )
        self.stage1 = _YelmenBinaryRBM(
            num_visible=self.label_bits + self.units_per_snp * self.first_segment_len + self.units_per_snp * self.second_segment_len,
            fixed_nodes=self.label_bits + self.units_per_snp * self.first_segment_len,
            **common,
        )
        self.register_buffer("_initialized_visible_bias", torch.tensor(False), persistent=True)

    def _stage_visible_from_xy(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dosage = _as_dosage(x, int(self.hparams.seq_len))
        labels = y.reshape(-1).to(dosage.device).long().clamp(0, 1)
        if self.hparams.genotype_encoding == "random_phase_haplotypes":
            haplotypes = _dosage_to_random_phase_haplotypes(dosage)
            labels = labels.repeat_interleave(2)
            first_bits = haplotypes[:, : self.first_segment_len]
            second_bits = haplotypes[:, self.first_segment_len :]
        else:
            first_bits = _dosage_to_allele_bits(dosage[:, : self.first_segment_len])
            second_bits = _dosage_to_allele_bits(dosage[:, self.first_segment_len :])
        prefix = []
        if self.label_bits:
            prefix.append(labels.float().unsqueeze(1))
        stage0_visible = torch.cat((*prefix, first_bits), dim=1)
        stage1_visible = torch.cat((*prefix, first_bits, second_bits), dim=1)
        return stage0_visible, stage1_visible

    @torch.no_grad()
    def initialize_from_dataloader(self, dataloader) -> None:
        total0 = torch.zeros(self.stage0.num_visible)
        total1 = torch.zeros(self.stage1.num_visible)
        n_samples = 0
        for batch in dataloader:
            x, y = _batch_xy(batch)
            visible0, visible1 = self._stage_visible_from_xy(x, y)
            total0 += visible0.cpu().sum(dim=0)
            total1 += visible1.cpu().sum(dim=0)
            n_samples += visible0.shape[0]
        if n_samples == 0:
            raise ValueError("Cannot initialize CRBM visible bias from an empty dataloader.")
        self.stage0.set_visible_bias_from_mean(total0 / n_samples)
        self.stage1.set_visible_bias_from_mean(total1 / n_samples)
        self._initialized_visible_bias.fill_(True)

    def _require_initialized(self) -> None:
        if not bool(self._initialized_visible_bias.item()):
            raise RuntimeError("Call initialize_from_dataloader() before fitting ConditionalCRBMModule.")

    def on_train_start(self) -> None:
        self._require_initialized()

    def training_step(self, batch, batch_idx):
        self._require_initialized()
        x, y = _batch_xy(batch)
        visible0, visible1 = self._stage_visible_from_xy(x, y)
        v0 = visible0.to(self.device).t().contiguous()
        v1 = visible1.to(self.device).t().contiguous()

        with torch.no_grad():
            loss0 = self.stage0.fit_batch(v0)
            loss1 = self.stage1.fit_batch(v1)
            weight_norm0 = torch.linalg.matrix_norm(self.stage0.W)
            weight_norm1 = torch.linalg.matrix_norm(self.stage1.W)

        self.log_dict(
            {
                "train/loss/stage0_recon_mse": loss0,
                "train/loss/stage1_recon_mse": loss1,
                "train/loss/recon_mse": 0.5 * (loss0 + loss1),
                "train/scalars/stage0_weight_norm": weight_norm0,
                "train/scalars/stage1_weight_norm": weight_norm1,
            },
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )

    def validation_step(self, batch, batch_idx):
        self._require_initialized()
        x, y = _batch_xy(batch)
        visible0, visible1 = self._stage_visible_from_xy(x, y)
        v0 = visible0.to(self.device).t().contiguous()
        v1 = visible1.to(self.device).t().contiguous()

        v0_neg, _mv0, _h0, _mh0 = self.stage0.gibbs(v0, int(self.hparams.gibbs_steps))
        v1_neg, _mv1, _h1, _mh1 = self.stage1.gibbs(v1, int(self.hparams.gibbs_steps))
        loss0 = torch.mean((v0[self.stage0.fixed_nodes :] - v0_neg[self.stage0.fixed_nodes :]) ** 2)
        loss1 = torch.mean((v1[self.stage1.fixed_nodes :] - v1_neg[self.stage1.fixed_nodes :]) ** 2)
        loss = 0.5 * (loss0 + loss1)
        self.log_dict(
            {
                "val/loss": loss,
                "val/loss/stage0_recon_mse": loss0,
                "val/loss/stage1_recon_mse": loss1,
                "val/loss/recon_mse": loss,
            },
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        return []

    @torch.no_grad()
    def sample(
        self,
        labels: torch.Tensor | None = None,
        argmax: bool = True,
        num_samples: int | None = None,
    ) -> torch.Tensor:
        del argmax
        if labels is None:
            if self.hparams.condition_on_label:
                raise ValueError("labels are required when condition_on_label=True")
            if num_samples is None:
                raise ValueError("num_samples is required when labels=None")
            labels = torch.zeros(int(num_samples), dtype=torch.long, device=self.device)
        else:
            labels = labels.reshape(-1).long().to(self.device).clamp(0, 1)
        n_samples = labels.shape[0]
        if self.hparams.genotype_encoding == "random_phase_haplotypes":
            labels = labels.repeat_interleave(2)
        n_chains = labels.shape[0]

        v0 = torch.bernoulli(torch.rand(self.stage0.num_visible, n_chains, device=self.device))
        if self.label_bits:
            v0[0, :] = labels.float()
        v0_sample, _mv0, _h0, _mh0 = self.stage0.gibbs(v0, int(self.hparams.generation_gibbs_steps))
        first_bits = v0_sample[self.label_bits :, :]

        v1 = torch.bernoulli(torch.rand(self.stage1.num_visible, n_chains, device=self.device))
        if self.label_bits:
            v1[0, :] = labels.float()
        v1[self.label_bits : self.stage1.fixed_nodes, :] = first_bits
        v1_sample, _mv1, _h1, _mh1 = self.stage1.gibbs(v1, int(self.hparams.generation_gibbs_steps))
        second_bits = v1_sample[self.stage1.fixed_nodes :, :]

        if self.hparams.genotype_encoding == "random_phase_haplotypes":
            haplotypes = torch.cat((first_bits, second_bits), dim=0).t()
            return haplotypes.reshape(n_samples, 2, int(self.hparams.seq_len)).sum(dim=1).long()
        first = _allele_bits_to_dosage(first_bits, self.first_segment_len)
        second = _allele_bits_to_dosage(second_bits, self.second_segment_len)
        return torch.cat((first, second), dim=1)[:, : int(self.hparams.seq_len)]
