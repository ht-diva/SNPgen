import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Iterator, List, Optional, Tuple, Union, Set

from snpgen.utils import instantiate_from_config

class GeneralLoss(nn.Module):
    def __init__(
        self,
        logvar_init: float = 0.0,
        learn_logvar: bool = False,
        kl_weight: float = 1.0
    ):
        super().__init__()
        self.kl_weight = kl_weight

        # output log variance
        self.logvar = nn.Parameter(
            torch.full((), logvar_init), requires_grad=learn_logvar
        )
        self.learn_logvar = learn_logvar

        self.recon_loss = nn.CrossEntropyLoss(reduction='none')

    def get_trainable_autoencoder_parameters(self, named=False) -> Iterator[nn.Parameter]:
        if self.learn_logvar:
            if named:
                yield ('logvar', self.logvar)
            else:
                yield self.logvar
        yield from ()
        
    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {'logvar'}

    def calculate_adaptive_weight(
        self, loss: torch.Tensor, nll_loss: torch.Tensor, base_weight: torch.Tensor, last_layer: torch.Tensor
    ) -> torch.Tensor:
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        loss_grads = torch.autograd.grad(loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(loss_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * base_weight
        return d_weight

    def get_nll_loss(self, rec_loss: torch.Tensor) -> torch.Tensor:
        ''' It modifies rec_loss only when self.logvar != 0.0 (i.e. if we learn logvar or if logvar_init != 0.0) '''
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        return nll_loss

    def compute_kl(self, mean, logvar):
        sum_dims = tuple(range(1, len(mean.shape))) # sum over all dimensions except batch dimension
        return torch.mean(-0.5 * torch.sum(1.0 + logvar - mean.pow(2) - logvar.exp(), dim=sum_dims))

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        *, 
        mean: torch.Tensor = None,
        logvar: torch.Tensor = None,
        split: str = "train",
        additional_losses: Dict = {},
        return_nll_loss: bool = False,
        channel_first: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, dict]:

        if not channel_first:
            # If the reconstruction are in channel last format, we need to permute them to channel first
            # to compute the CrossEntropyLoss
            reconstructions = reconstructions.permute(0, 2, 1) # (B, L, C) -> (B, C, L)
            
        # Compute reconstruction loss
        recons_loss = self.recon_loss(reconstructions, inputs)
        nll_loss = self.get_nll_loss(recons_loss)
        
        # Compute KL loss
        kl_loss = self.compute_kl(mean, logvar)

        # Compute total loss
        loss = nll_loss + self.kl_weight * kl_loss

        # Add and log additional losses (if any)
        additional_log = dict()
        for k, v in additional_losses.items():
            additional_loss = v['loss']
            base_weight = v.get('base_weight', 1.0)
            factor = v.get('factor', 1.0)
            if self.training:
                if 'adaptive_weight' in v and v['adaptive_weight']:
                    adaptive_weight = self.calculate_adaptive_weight(additional_loss, nll_loss,
                                                                     base_weight=base_weight,
                                                                     last_layer=v['last_layer'])
                    additional_log[f"{split}/scalars/{k}/adaptive_weight"] = adaptive_weight.detach() # log adaptive weight
                else:
                    adaptive_weight = base_weight
            else:
                adaptive_weight = torch.tensor(1.0)
            loss += additional_loss * adaptive_weight * factor
            additional_log[f"{split}/loss/{k}"] = self.get_scalar_log_loss(additional_loss)

        # Build log
        log = dict()
        log.update(
            {
                f"{split}/loss/total": self.get_scalar_log_loss(loss, clone=True),
                f"{split}/loss/nll": self.get_scalar_log_loss(nll_loss),
                f"{split}/loss/rec": self.get_scalar_log_loss(recons_loss),
                f"{split}/loss/kl": self.get_scalar_log_loss(kl_loss),
                f"{split}/scalars/logvar": self.logvar.detach(),
            }
        )
        log.update(additional_log)

        if return_nll_loss:
            return loss, nll_loss, log
        else:
            return loss, log

    def get_scalar_log_loss(self, loss, clone=False):
        '''
        The idea is to get a scalar loss for logging purposes.
         - If the loss is already a scalar, we just return it.
         - If the loss is a one-dimensional tensor, we compute the mean over the batch dimension.
         - If the loss is a multi-dimensional tensor, we sum over all dimensions
           except the first (batch dimension) and then compute the mean over the batch dimension.
        '''
        if clone:
            loss = loss.clone()

        loss = loss.detach()
        loss_dim = loss.dim()

        if loss_dim == 0:
            return loss
        elif loss_dim == 1:
            return loss.mean()
        else:
            # Easy way to sum over all dimensions except the first (bacth size) 
            # and then compute the mean over the batch dim.
            return loss.flatten(1).sum(-1).mean()
    

class GeneralLossWithDiscriminator(GeneralLoss):
    def __init__(
        self,
        disc_start: int,
        discriminator_config: Dict,
        disc_factor: float = 1.0,
        disc_weight: float = 1.0,
        disc_loss: str = "hinge",
        **kwargs
    ):
        super().__init__(**kwargs)
        assert disc_loss in ["hinge", "vanilla"]
        self.discriminator_iter_start = disc_start
        self.discriminator = instantiate_from_config(discriminator_config)
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss

    def get_trainable_parameters(self, named=False) -> Iterator[nn.Parameter]:
        if named:
            return self.discriminator.named_parameters()
        else:
            return self.discriminator.parameters()

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        *,
        optimizer_idx: int,
        global_step: int,
        last_layer: torch.Tensor,
        oh_inputs: torch.Tensor,
        split: str = "train",
        **kwargs
    ) -> Tuple[torch.Tensor, dict]:
        if optimizer_idx == 0:
            # generator update
            log_d_weight = True
            loss, nll_loss, log = super().forward(inputs, reconstructions, split=split, return_nll_loss=True, **kwargs)
            if global_step >= self.discriminator_iter_start or not self.training:
                logits_fake = self.discriminator(reconstructions.contiguous())
                g_loss = -torch.mean(logits_fake)
                if self.training:
                    d_weight = self.calculate_adaptive_weight(g_loss, nll_loss,
                                                              base_weight=self.discriminator_weight,
                                                              last_layer=last_layer)
                else:
                    d_weight = torch.tensor(1.0)
                    log_d_weight = False
            else:
                d_weight = torch.tensor(0.0)
                g_loss = torch.tensor(0.0, requires_grad=True)

            loss += d_weight * self.disc_factor * g_loss
            log.update({
                f"{split}/loss/g": self.get_scalar_log_loss(g_loss),
            })
            if log_d_weight:
                log[f"{split}/scalars/d_weight"] = d_weight.detach()

        elif optimizer_idx == 1:
            # second pass for discriminator update
            logits_real = self.discriminator(oh_inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            if global_step >= self.discriminator_iter_start or not self.training:
                loss = self.disc_factor * self.disc_loss(logits_real, logits_fake)
            else:
                loss = torch.tensor(0.0, requires_grad=True)

            log = {
                f"{split}/loss/disc": self.get_scalar_log_loss(loss, clone=True),
                f"{split}/logits/real": self.get_scalar_log_loss(logits_real),
                f"{split}/logits/fake": self.get_scalar_log_loss(logits_fake),
            }

        else:
            raise ValueError(f"Invalid optimizer_idx: {optimizer_idx}")
        
        return loss, log


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(F.softplus(-logits_real))
        + torch.mean(F.softplus(logits_fake))
    )
    return d_loss
