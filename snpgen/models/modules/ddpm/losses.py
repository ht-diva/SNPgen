from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from snpgen.models.modules.embedding.embedders import GeneralConditioner
from snpgen.utils import append_dims, instantiate_from_config
from snpgen.models.modules.ddpm.denoiser import Denoiser


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config: Dict,
        loss_weighting_config: Dict,
        loss_type: str = "l2",
        offset_noise_level: float = 0.0,
        batch2model_keys: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__()

        assert loss_type in ["l2", "l1"]
        
        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)

        self.loss_type = loss_type
        self.offset_noise_level = offset_noise_level

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

        self.n_frames = None

    def get_noised_input(
        self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        noised_input = input + noise * sigmas_bc
        return noised_input

    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        input: torch.Tensor,
        batch: Dict,
    ) -> torch.Tensor:
        cond = conditioner(batch)
        return self._forward(network, denoiser, cond, input, batch)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        input: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        sigmas = self.sigma_sampler(input.shape[0]).to(input)
        sigmas_bc = append_dims(sigmas, input.ndim)

        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (input.shape[0], 1, input.shape[2])
                if self.n_frames is not None
                else (input.shape[0], input.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        noised_input = self.get_noised_input(sigmas_bc, noise, input)

        model_output = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )
        w = append_dims(self.loss_weighting(sigmas), input.ndim)
        
        if 'loss_mask' in batch:
            mask = batch['loss_mask']
        else:
            mask = None
        
        return self.get_loss(model_output, input, w, mask=mask)

    def get_loss(self, model_output, target, w, mask=None):
        if self.loss_type == "l2":
            return mse_loss(model_output, target, w=w, mask=mask)
        elif self.loss_type == "l1":
            return mae_loss(model_output, target, w=w, mask=mask)
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")
        
        
def _compute_base_loss(x, y, type='l2', w=1.,):
    """Helper function to compute either MSE or MAE base loss."""
    diff = x - y
    if type == 'l2':
        loss = diff ** 2
    elif type == 'l1':
        loss = diff.abs()
    else:
        raise ValueError(f"Invalid loss type: {type}")
    
    return w * loss

def _compute_loss(x, y, type, w=1., mask=None):
    """Computes either masked or unmasked loss between input and target.
    Args:
        x (torch.Tensor): Input tensor to compute loss for
        y (torch.Tensor): Target tensor to compute loss against
        type (str): Type of base loss to compute
        w (float, optional): Weight factor for loss computation. Defaults to 1.0
        mask (torch.Tensor, optional): Optional boolean mask tensor for masked loss computation. 
            If provided, loss will be computed only on True values in the mask. Defaults to None.
    Returns:
        torch.Tensor: Computed loss values per batch sample. Returns mean loss if unmasked,
            or masked average loss if mask is provided.
    """
    base_loss = _compute_base_loss(x, y, type=type, w=w)
    
    if mask is None:
        # Unmasked version
        return torch.mean(base_loss.reshape(y.shape[0], -1), 1)
    
    # assert that mask is a tensor and a boolean tensor
    assert torch.is_tensor(mask), "Mask should be a tensor"
    assert mask.dtype == torch.bool, "Mask should be a boolean tensor"
    
    # Masked version
    masked_loss = base_loss * mask
    masked_loss = masked_loss.reshape(x.shape[0], -1)
    
    # Compute denominator based on mask dimensions
    # If mask has the same dimensions as x, then we suppose it's a batch of masks,
    # otherwise it's a single mask equal for all samples.
    denom = mask.reshape(x.shape[0], -1).sum(dim=1) if mask.ndim == x.ndim else mask.sum()
    return masked_loss.sum(dim=1) / (denom + 1e-8)

# Public interface functions
def mse_loss(x, y, w=1., mask=None):
    return _compute_loss(x, y, 'l2', w, mask=mask)

def mae_loss(x, y, w=1., mask=None):
    return _compute_loss(x, y, 'l1', w, mask=mask)

