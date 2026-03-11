import torch
import torch.nn as nn
import os
import os.path
import warnings
from copy import deepcopy

from contextlib import contextmanager

import lightning.pytorch as pl

from lightning.pytorch import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_warn, rank_zero_info, rank_zero_only
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import STEP_OUTPUT

from typing import Any, Dict, List, Optional, Union

try:
    import amp_C
    APEX_IS_AVAILABLE = True
except Exception:
    APEX_IS_AVAILABLE = False

from snpgen.utils import get_pylogger, print_rank_zero

log = get_pylogger(__name__)

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Adapted from: https://github.com/huggingface/pytorch-image-models/blob/8a713b09e5ee917a4b2379738d4f2afefc64e276/timm/utils/model_ema.py#L133-L260
# -------------------------------------------------------------------------------------------------------------------------------------------------
class ModelEmaV3(nn.Module):
    """ Model Exponential Moving Average V3

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This module chooses the best EMA implementation for performance.

    Decay warmup based on code by @crowsonkb, her comments:
      If inv_gamma=1 and power=1, implements a simple average. inv_gamma=1, power=2/3 are
      good values for models you plan to train for a million or more steps (reaches decay
      factor 0.999 at 31.6K steps, 0.9999 at 1M steps), inv_gamma=1, power=3/4 for models
      you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
      215.4k steps).

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory

    Args:
        model (nn.Module): model to which apply EMA
        decay (float): decay rate for exponential moving average
        min_decay (float): minimum decay rate
        update_after_step (int): start updating EMA after this step
        use_warmup (bool): use warmup for EMA decay
        warmup_gamma (float): warmup gamma. Only required if `use_warmup` is True
        warmup_power (float): warmup power. Only required if `use_warmup` is True
        use_karras (bool): use Karras et al. EMA decay
        karras_gamma (float): Karras gamma. Only required if `use_karras` is True
        device (Optional[torch.device]): device on which store the EMA model
        exclude_buffers (bool): don't apply EMA to model buffers
        verbose (bool): print information about EMA
    """

    EMA_IMPLEMENTATIONS = ['foreach_lerp', 'apex' 'foreach_mul_add', 'lerp', 'base'] # in order of priority for performance (see ./benchmark.py)

    def __init__(
            self,
            model,
            decay: float = 0.9999,
            min_decay: float = 0.0,
            update_after_step: int = 0,
            use_warmup: bool = False,
            warmup_gamma: float = 1.0,
            warmup_power: float = 3/4,
            use_karras: bool = False,
            karras_gamma: Optional[float] = None,
            device: Optional[torch.device] = None,
            exclude_buffers: bool = False,
            verbose: bool = False
    ):
        super().__init__()

        assert not (use_warmup and use_karras), "Cannot use both warmup and karras_decay"

        if use_karras:
            assert karras_gamma is not None, "karras_gamma must be set if use_karras is True"
            if verbose: rank_zero_info(f"Using Karras et al. EMA decay with gamma={karras_gamma}. The provided decay value will be ignored.")

        if use_warmup:
            assert warmup_gamma is not None and warmup_power is not None, "warmup_gamma and warmup_power must be set if use_warmup is True"
            if verbose: rank_zero_info(f"Using warmup for EMA decay with gamma={warmup_gamma} and power={warmup_power}")

        # Make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.requires_grad_(False)
        self.module.eval()

        # Hack: store the original model in a list so it won't saved in the state_dict
        self._tracked_module = [model]

        self._decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_warmup = use_warmup
        self.warmup_gamma = warmup_gamma
        self.warmup_power = warmup_power
        self.use_karras = use_karras
        self.karras_gamma = karras_gamma
        self.device = device  # perform EMA on different device from model if set
        self.exclude_buffers = exclude_buffers
        self.verbose = verbose

        if device is not None:
            if verbose: rank_zero_info(f"EMA model will be stored on '{device}' device")
            self.module.to(device=device)

        self.ema_implementation = None
        self.register_buffer('step', torch.tensor(0))

    @property
    def tracked_module(self):
        return self._tracked_module[0]

    @property
    def decay(self):
        if self.use_karras:
            step = self.step.item()
            return (1 - 1 / (step + 1)) ** (1 + self.karras_gamma)
        
        return self._decay

    def get_decay(self, step: Optional[int] = None) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        if step is None:
            return self.decay

        step = max(0, step - self.update_after_step - 1)

        if step <= 0:
            return 0.0

        if self.use_warmup:
            decay = 1 - (1 + step / self.warmup_gamma) ** -self.warmup_power
            decay = max(min(decay, self.decay), self.min_decay)
        else:
            decay = self.decay # (1 + step) / (10 + step)

        return decay

    @torch.no_grad()
    def update(self, step: Optional[int] = None):
        if step is not None:
            self.step = torch.tensor(step)
        else:
            step = self.step.item()
            self.step += 1

        decay = self.get_decay(step)
        self.apply_update_(self.tracked_module, decay, skip_buffers=self.exclude_buffers)
        
    def apply_update_(self, model: nn.Module, decay: float, skip_buffers: bool = False):
        # Interpolate parameters and optionally buffers
        ema_values = []
        model_values = []
        
        # Handle parameters and buffers separately instead of using model.state_dict().values()
        # to be able to check for requires_grad

        # Handle parameters
        for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
            if model_p.is_floating_point() and model_p.requires_grad:
                ema_values.append(ema_p)
                model_values.append(model_p.to(ema_p.device, non_blocking=True))
            else:
                ema_p.copy_(model_p.to(ema_p.device, non_blocking=True))

        # Handle buffers
        for ema_b, model_b in zip(self.module.buffers(), model.buffers()):
            if skip_buffers:
                # Simply copy buffers without EMA
                ema_b.copy_(model_b.to(ema_b.device, non_blocking=True))
            elif model_b.is_floating_point():
                # Apply EMA for floating-point buffers
                ema_values.append(ema_b)
                model_values.append(model_b.to(ema_b.device, non_blocking=True))
            else:
                # Copy non-floating point buffers
                ema_b.copy_(model_b.to(ema_b.device, non_blocking=True))

        self.perform_ema(ema_values, model_values, decay)

    def perform_ema(self, ema_weights: List[torch.Tensor], model_weights: List[torch.Tensor], decay: float):
        ''' Perform the EMA update using the best available implementation. '''
            
        # Determine the best EMA implementation (only once)
        if self.ema_implementation is None:
            supported_implementations = []
            if hasattr(torch, '_foreach_lerp_'):
                supported_implementations.append('foreach_lerp')
            if APEX_IS_AVAILABLE and ema_weights[0].is_cuda:
                supported_implementations.append('apex')
            if hasattr(torch, '_foreach_mul_') and hasattr(torch, '_foreach_add_'):
                supported_implementations.append('foreach_mul_add')
            if hasattr(torch.Tensor, 'lerp_'):
                supported_implementations.append('lerp')
            supported_implementations.append('base')

            # get best available implementation
            self.ema_implementation = next((impl for impl in self.EMA_IMPLEMENTATIONS if impl in supported_implementations), None)

            if self.ema_implementation == 'apex':
                # setup overflow buffer for apex
                self._overflow_buf = torch.IntTensor([0]).to(ema_weights[0].device)

            if self.verbose:
                rank_zero_info(f"\nUsing the '{self.ema_implementation}' implementation for EMA.")

        self._perform_ema(self.ema_implementation, ema_weights, model_weights, decay)
        
    def _perform_ema(self, implementation: str, ema_weights: List[torch.Tensor], model_weights: List[torch.Tensor], decay: float):
        if implementation == 'foreach_lerp':
            torch._foreach_lerp_(ema_weights, model_weights, weight=1. - decay)
        elif implementation == 'apex':
            # Perform EMA update using NVIDIA's APEX library: OUT = a * X + b * Y
            amp_C.multi_tensor_axpby(
                65536, # maximum number of tensors that can be processed in a single call
                self._overflow_buf, # a tensor used to detect if any numerical overflows occur during the operation.
                [ema_weights, model_weights, ema_weights], # [X, Y, OUT]
                decay, # a
                1. - decay, # b
                -1,
            )
        elif implementation == 'foreach_mul_add':
            torch._foreach_mul_(ema_weights, scalar=decay)
            torch._foreach_add_(ema_weights, model_weights, alpha=1. - decay)
        elif implementation == 'lerp':
            for ema_v, model_v in zip(ema_weights, model_weights):
                ema_v.lerp_(model_v, weight=1. - decay)
        elif implementation == 'base':
            for ema_v, model_v in zip(ema_weights, model_weights):
                ema_v.copy_(decay * ema_v + (1. - decay) * model_v)
        else:
            raise ValueError(f"Unknown EMA implementation '{implementation}'")
        
    @torch.no_grad()
    def set(self):
        for ema_v, model_v in zip(self.module.state_dict().values(), self.tracked_module.state_dict().values()):
            ema_v.copy_(model_v.to(ema_v.device, non_blocking=True))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
class EMAScope:
    """
    Class-container to store and restore model parameters and buffers.

    Args:
        store_device (Union[torch.device, str, None]): device on which store the backup of the parameters and buffers.
    """
    def __init__(self, store_device = None):
        self.store_device = store_device

    def store(self, model):
        ''' Save the current parameters and buffers for restoring later. '''
        if self.store_device is not None:
            self.collected_params = [param.clone().to(self.store_device) for param in model.parameters()]
            self.collected_buffers = [buffer.clone().to(self.store_device) for buffer in model.buffers()]
        else:
            self.collected_params = [param.clone() for param in model.parameters()]
            self.collected_buffers = [buffer.clone() for buffer in model.buffers()]

    def restore(self, model):
        """
        Restore the parameters and buffers stored with the `store` method.
        Useful to validate the model with EMA parameters/buffers without affecting the
        original optimization process.
        """
        for c_param, param in zip(self.collected_params, model.parameters()):
            param.data.copy_(c_param.data)

        for c_buffer, buffer in zip(self.collected_buffers, model.buffers()):
            buffer.data.copy_(c_buffer.data)

        #del self.collected_params
        #del self.collected_buffers

    def copy_to(self, shadow_model, model):
        ''' Copy shadow model parameters and buffers into given model. '''
        for s_param, param in zip(shadow_model.parameters(), model.parameters()):
            param.data.copy_(s_param.data)

        for s_buffer, buffer in zip(shadow_model.buffers(), model.buffers()):
            buffer.data.copy_(s_buffer.data)

    @contextmanager
    def ema_scope(self, pl_module, context=None):
        if pl_module.use_ema:
            self.store(pl_module.model_ema.tracked_module)
            self.copy_to(pl_module.model_ema.module, pl_module.model_ema.tracked_module)
            if context is not None:
                print_rank_zero(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if pl_module.use_ema:
                self.restore(pl_module.model_ema.tracked_module)
                if context is not None:
                    print_rank_zero(f"{context}: Restored training weights")
    

class EMACallbackV3(Callback):
    def __init__(self, ema_scope: Optional[EMAScope] = None, validate_with_ema_weights: bool = False, use_ema_weights: bool = False):
        assert ema_scope is not None or not (validate_with_ema_weights or use_ema_weights), "ema_scope must be provided if validate_with_ema_weights or use_ema_weights are true."
        self.ema_scope = ema_scope
        self.validate_with_ema_weights = validate_with_ema_weights
        self.use_ema_weights = use_ema_weights

    # Since the pl.LightningModule is moved to the desired accelerator (e.g. GPU) only when the training starts
    # and 'model_ema' is a sub-module of the pl.LightningModule, it will moved to the device too.
    # However, the 'model_ema' could have been initialized with a specific desired device (e.g. CPU),
    # so at this point we need to move it back to the desired device.
    # Note: this is not optimal because, even if only for a moment, the 'model_ema' will be on the
    #       the "wrong" device occupying its memory, which could result in OOM.
    def on_fit_start(self, trainer, pl_module):
        if pl_module.global_rank == 0:
            assert getattr(pl_module, "use_ema", False), "You are using the EMA callback but the 'use_ema' attribute of the LightningModule is not set to True."
            if pl_module.use_ema:
                assert hasattr(pl_module, "model_ema"), "You are using the EMA callback but the LightningModule does not have the 'model_ema' attribute."
                assert hasattr(pl_module.model_ema, "module"), "The 'model_ema' attribute must have a 'module' attribute."
                if getattr(pl_module.model_ema, "device", None) is not None:
                    rank_zero_info(f"Moving back the EMA model to the '{pl_module.model_ema.device}' device.")
                    pl_module.model_ema.module.to(pl_module.model_ema.device)

                # Just to be sure
                pl_module.model_ema.module.requires_grad_(False)
                pl_module.model_ema.module.eval()

    @rank_zero_only
    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        ''' Update the stored parameters using a moving average. It's done only on rank zero.'''
        # Update currently maintained parameters.
        pl_module.model_ema.update(step=trainer.global_step)

    def on_validation_start(self, trainer, pl_module):
        if self.validate_with_ema_weights:
            # do validation using the stored parameters"
            # save original parameters before replacing with EMA version
            self.ema_scope.store(pl_module.model_ema.tracked_module)

            # update the LightningModule with the EMA weights
            # ~ Copy EMA parameters to LightningModule
            self.ema_scope.copy_to(pl_module.model_ema.module, pl_module.model_ema.tracked_module)

    def on_validation_end(self, trainer, pl_module):
        if self.validate_with_ema_weights:
            # Restore original parameters to resume training later
            self.ema_scope.restore(pl_module.model_ema.tracked_module)

    def on_train_end(self, trainer, pl_module):
        if self.use_ema_weights:
            # Update the LightningModule with the EMA weights
            self.ema_scope.copy_to(pl_module.model_ema.module, pl_module.model_ema.tracked_module)
            rank_zero_info("Model weights replaced with the EMA version.")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        pass

    def on_load_checkpoint(self, callback_state):
        pass







# ------------------------------------------------------------------------------------------------------------------------------------------
# Adapted from: https://github.com/BioinfoMachineLearning/bio-diffusion/blob/e4bad15139815e562a27fb94dab0c31907522bc5/src/utils/__init__.py
# ------------------------------------------------------------------------------------------------------------------------------------------
class EMA(Callback):
    """
    Implements Exponential Moving Averaging (EMA).
    When training a model, this callback will maintain moving averages of the trained parameters.
    When evaluating, we use the moving averages copy of the trained parameters.
    When saving, we save an additional set of parameters with the prefix `ema`.
    Args:
        decay (float): The exponential decay used when calculating the moving average. Has to be between 0-1.
        apply_ema_every_n_steps (int): Apply EMA every n global steps.
        module_name: An optional string or list of strings to specify the module name(s) to which apply EMA.
            If None, EMA will be applied to all the parameters returned by ``pl_module.state_dict().values()``.
            Specifying the module names is useful for example to exclude parameters of custom losses from EMA updates.
        start_step (int): Start applying EMA from ``start_step`` global step onwards.
        min_decay (float): Minimum decay value to use.
        warmup (bool): Whether to use EMA warmup.
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1. Only used if `warmup` is True.
        power (float): Exponential factor of EMA warmup. Default: 2/3. Only used if `warmup` is True.
        karras_beta (bool): Whether to the Karras et al. beta version of EMA.
        save_ema_weights_in_callback_state (bool): Enable saving EMA weights in callback state.
        evaluate_ema_weights_instead (bool): Validate the EMA weights instead of the original weights.
            Note this means that when saving the model, the validation metrics are calculated with the EMA weights.

    Note:
        If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
        good values for models you plan to train for a million or more steps (reaches decay
        factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
        you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
        215.4k steps).

    Adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/callbacks/ema.py
    """

    def __init__(
        self,
        decay: float = 0.9999,
        apply_ema_every_n_steps: int = 1,
        module_name: Optional[Union[str, List[str]]] = None,
        start_step: int = 0,
        min_decay: float = 0.0,
        warmup: bool = False,
        power: float = 2 / 3,
        inv_gamma: float = 1.0,
        karras_beta: bool = False,
        save_ema_weights_in_callback_state: bool = False,
        evaluate_ema_weights_instead: bool = False,
    ):
        if not APEX_IS_AVAILABLE:
            rank_zero_warn(
                "EMA has better performance when Apex is installed: https://github.com/NVIDIA/apex#installation."
            )
        if not (0 <= decay <= 1) and not karras_beta:
            raise MisconfigurationException("EMA decay value must be between 0 and 1")
        
        self._ema_model_weights: Optional[List[torch.Tensor]] = None
        self._overflow_buf: Optional[torch.Tensor] = None
        self._cur_step: Optional[int] = None
        self._weights_buffer: Optional[List[torch.Tensor]] = None
        self.apply_ema_every_n_steps = apply_ema_every_n_steps
        self.start_step = start_step
        self.save_ema_weights_in_callback_state = save_ema_weights_in_callback_state
        self.evaluate_ema_weights_instead = evaluate_ema_weights_instead
        self.decay = decay
        self.min_decay = min_decay
        self.warmup = warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.karras_beta = karras_beta

        self.module_name = module_name
        if isinstance(self.module_name, str):
            self.module_name = [self.module_name]

    def get_module_weights(self, pl_module: "pl.LightningModule", clone: bool = False, cpu: bool = False, verbose: bool = False) -> List[torch.Tensor]:
        if self.module_name is None:
            if verbose: log.info(f"EMA will be applied to all the parameters of the module.")
            weights = list(pl_module.state_dict().values())
        else:
            if verbose: log.info(f"EMA will be applied to the following module(s): {self.module_name}")
            weights = []
            for name in self.module_name:
                module = getattr(pl_module, name)
                weights += list(module.state_dict().values())

        if clone:
            weights = [p.detach().clone() for p in weights]

        if cpu:
            weights = [p.to('cpu') for p in weights]

        return weights
    
    def get_module_keys(self, pl_module: "pl.LightningModule") -> List[str]:
        keys = list(pl_module.state_dict().keys())
        if self.module_name is not None:
            keys = [k for k in keys if k.split('.')[0] in self.module_name]
        return keys
    
    @property
    def beta(self):
        if self.karras_beta:
            return (1 - 1 / (self._cur_step + 1)) ** (1 + self.power)
        return self.decay
    
    def get_curent_decay(self) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, self._cur_step - self.start_step - 1)

        if step <= 0:
            return 0.0

        if self.warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            cur_decay_value = self.beta #(1 + step) / (10 + step)

        cur_decay_value = min(cur_decay_value, self.beta)
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        log.info("Creating EMA weights copy.")
        if hasattr(pl_module, 'get_tracked_ema_modules'):
            # if the model has a method to get the EMA modules, use it
            self.module_name = pl_module.get_tracked_ema_modules()
            rank_zero_info(f"The pl.LightningModule has a method to get the EMA modules. Using it to get the module(s) to apply EMA.")

        if self.module_name is None:
            rank_zero_info("EMA will be applied to all the parameters of the module.")  
        else:
            rank_zero_info(f"EMA will be applied to the following module(s): {self.module_name}")

        if self._ema_model_weights is None:
            self._ema_model_weights = self.get_module_weights(pl_module, clone=True, verbose=True) # init EMA weights with a copy of the model weights
        # ensure that all the weights are on the correct device
        self._ema_model_weights = [p.to(pl_module.device) for p in self._ema_model_weights]
        self._overflow_buf = torch.IntTensor([0]).to(pl_module.device)

    def ema(self, pl_module: "pl.LightningModule") -> None:
        if APEX_IS_AVAILABLE and pl_module.device.type == "cuda":
            return self.apply_multi_tensor_ema(pl_module)
        return self.apply_ema(pl_module)

    def apply_multi_tensor_ema(self, pl_module: "pl.LightningModule") -> None:
        model_weights = self.get_module_weights(pl_module)

        # Perform EMA update using NVIDIA's APEX library: OUT = a * X + b * Y
        amp_C.multi_tensor_axpby(
            65536, # maximum number of tensors that can be processed in a single call
            self._overflow_buf, # a tensor used to detect if any numerical overflows occur during the operation.
            [self._ema_model_weights, model_weights, self._ema_model_weights], # [X, Y, OUT]
            self.get_curent_decay(), # a
            1. - self.get_curent_decay(), # b
            -1,
        )
  
    def apply_ema(self, pl_module: "pl.LightningModule") -> None:
        for orig_weight, ema_weight in zip(self.get_module_weights(pl_module), self._ema_model_weights):
            if ema_weight.data.dtype != torch.long and orig_weight.data.dtype != torch.long:
                # ensure that non-trainable parameters (e.g., feature distributions) are not included in EMA weight averaging
                diff = ema_weight.data - orig_weight.data
                diff.mul_(1. - self.get_curent_decay())
                ema_weight.sub_(diff)

    def should_apply_ema(self, step: int) -> bool:
        return step != self._cur_step and step >= self.start_step and step % self.apply_ema_every_n_steps == 0

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if self.should_apply_ema(trainer.global_step):
            self._cur_step = trainer.global_step
            self.ema(pl_module)

    def state_dict(self) -> Dict[str, Any]:
        if self.save_ema_weights_in_callback_state:
            return dict(cur_step=self._cur_step, ema_weights=self._ema_model_weights)
        return dict(cur_step=self._cur_step)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._cur_step = state_dict["cur_step"]
        if self._ema_model_weights is None:
            self._ema_model_weights = state_dict.get("ema_weights")

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        checkpoint_callback = trainer.checkpoint_callback

        if trainer.ckpt_path and checkpoint_callback is not None:
            ext = checkpoint_callback.FILE_EXTENSION
            if trainer.ckpt_path.endswith(f"-EMA{ext}"):
                log.info(
                    "loading EMA based weights. "
                    "The callback will treat the loaded EMA weights as the main weights"
                    " and create a new EMA copy when training."
                )
                return
            ema_path = trainer.ckpt_path.replace(ext, f"-EMA{ext}")
            if os.path.exists(ema_path):
                ema_state_dict = torch.load(ema_path, map_location=torch.device("cpu"), weights_only=False)
                self._ema_model_beights = ema_state_dict["state_dict"].values()
                del ema_state_dict
                log.info("EMA weights have been loaded successfully. Continuing training with saved EMA weights.")
            else:
                warnings.warn(
                    "we were unable to find the associated EMA weights when re-loading, "
                    "training will start with new EMA weights.",
                    UserWarning,
                )

    def replace_model_weights(self, pl_module: "pl.LightningModule") -> None:
        self._weights_buffer = self.get_module_weights(pl_module, clone=True)
        new_state_dict = {k: v for k, v in zip(self.get_module_keys(pl_module), self._ema_model_weights)}
        pl_module.load_state_dict(new_state_dict, strict=False)

    def restore_original_weights(self, pl_module: "pl.LightningModule") -> None:
        new_state_dict = {k: v for k, v in zip(self.get_module_keys(pl_module), self._weights_buffer)}
        pl_module.load_state_dict(new_state_dict, strict=False)
        del self._weights_buffer

    @property
    def ema_initialized(self) -> bool:
        return self._ema_model_weights is not None

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.replace_model_weights(pl_module)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.restore_original_weights(pl_module)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.replace_model_weights(pl_module)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.restore_original_weights(pl_module)


class EMAModelCheckpoint(ModelCheckpoint):
    """
    Light wrapper around Lightning's `ModelCheckpoint` to, upon request, save an EMA copy of the model as well.

    Adapted from: https://github.com/NVIDIA/NeMo/blob/be0804f61e82dd0f63da7f9fe8a4d8388e330b18/nemo/utils/exp_manager.py#L744
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_ema_callback(self, trainer: "pl.Trainer") -> Optional[EMA]:
        ema_callback = None
        for callback in trainer.callbacks:
            # TODO: the second condition is a temporary fix needed when we are using Juptyer notebooks with autoreload
            if isinstance(callback, EMA) or (".".join([callback.__module__, callback.__class__.__name__]) == 'snpgen.training.callbacks.ema.EMA'):
                ema_callback = callback
        return ema_callback

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        ema_callback = self._get_ema_callback(trainer)
        if ema_callback is not None:
            if self.verbose:
                rank_zero_info(f"Saving EMA weights to separate checkpoint {filepath}") 
            # Set EMA weights into the model
            ema_callback.replace_model_weights(trainer.lightning_module)           
            # Save model (i.e. EMA model)
            super()._save_checkpoint(trainer, filepath)
            # Restore original weights
            ema_callback.restore_original_weights(trainer.lightning_module)

    def _ema_format_filepath(self, filepath: str) -> str:
        return filepath.replace(self.FILE_EXTENSION, f"{self.CHECKPOINT_JOIN_CHAR}EMA{self.FILE_EXTENSION}")
    
    def format_checkpoint_name(
        self, metrics: Dict[str, torch.Tensor], filename: Optional[str] = None, ver: Optional[int] = None
    ) -> str:
        filepath = super().format_checkpoint_name(metrics, filename, ver)
        filepath = self._ema_format_filepath(filepath)
        return filepath