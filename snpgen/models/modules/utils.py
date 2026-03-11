import math
from operator import __add__
from functools import reduce, partial
from typing import Callable, Any, Iterator, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from snpgen.utils import get_proper_state_dict

NamedParams = Iterator[Dict[str, torch.nn.parameter.Parameter]]

def get_proper_state_dict_vae(load_path, ema=True):
    if ema:
        print(f"Loading VAE EMA weights from {load_path}")
        keyword = 'model_ema.module.'
    else:
        print(f"Loading VAE non-EMA weights {load_path}")
        keyword = 'autoencoder.'
    return get_proper_state_dict(load_path, keyword)


def get_proper_state_dict_ddpm(load_path, ema=True):
    if ema:
        print(f"Loading DDPM EMA weights from {load_path}")
        keyword = 'model_ema.module.'
    else:
        print(f"Loading DDPM non-EMA weights {load_path}")
        keyword = 'model.'
    return get_proper_state_dict(load_path, keyword)


def get_num_groups(num_channels: int, num_groups=32) -> int:
    """
    Calculate the number of groups for a given number of channels.

    Args:
        num_channels (int): The number of channels.
        num_groups (int): The desired number of groups. Defaults to 32.

    Returns:
        int: The number of groups nearest to the desired number of groups which exactly divides num_channels.
    """
    def divisors(n):
        result = set()
        for i in range(1, int(n**0.5)+1):
            if n % i == 0:
                result.add(i)
                result.add(n//i)
        return list(result)
    
    return min(divisors(num_channels), key=lambda x:abs(x-num_groups))


# Adapted from https://github.com/huggingface/pytorch-image-models/blob/aeb1ed7a15594505c1585697c1cd90cb49e7a115/timm/optim/_param_groups.py#L13
# to support named parameters as well
def param_groups_weight_decay(
        model_or_named_params: Union[nn.Module, NamedParams],
        weight_decay=1e-5,
        no_weight_decay_list=()
):
    
    assert weight_decay is not None and weight_decay >= 0.0
    
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    
    if hasattr(model_or_named_params, "named_parameters"):
        # nn.Module
        params_iterator = model_or_named_params.named_parameters()
    else:
        # NamedParams
        params_iterator = model_or_named_params
    
    for name, param in params_iterator:
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]
    
    
def get_params_names_with_no_weight_decay(model: nn.Module):
    """
    Recursively retrieves the names of all parameters in the given model and its submodules
    that should not have weight decay applied during optimization. This is determined by
    checking if the submodule has a `no_weight_decay` method, which should return the names
    of parameters that should not have weight decay.
    
    Args:
        model (nn.Module): The model to inspect.
        
    Returns:
        List[str]: A list of parameter names that should not have weight decay.
    """
    
    def _get_full_param_names(module: nn.Module, prefix: str):
        param_names = []
        
        for name, submodule in module.named_children():
            submodule_prefix = f"{prefix}.{name}" if prefix else name
            # Check if the submodule has a `no_weight_decay` method
            if hasattr(submodule, "no_weight_decay") and callable(getattr(submodule, "no_weight_decay")):
                no_wd_params = submodule.no_weight_decay()
                if isinstance(no_wd_params, str):
                    no_wd_params = [no_wd_params]
                param_names.extend(f"{submodule_prefix}.{param}" for param in no_wd_params)
            # Recursively process submodules
            param_names.extend(_get_full_param_names(submodule, submodule_prefix))
        return param_names
    
    # Start with the main model
    return _get_full_param_names(model, '')
    
    
def get_params_with_decay(
    module: nn.Module,
    named_params: NamedParams = None,
    weight_decay=None,
    check_skip_list=True
):

    """
    Get parameters with weight decay.
    This function can work with either a module directly or named parameters, and optionally applies
    weight decay selectively based on parameter names.
    Args:
        module (nn.Module): The PyTorch module to get parameters from
        named_params (NamedParams, optional): Named parameters if already extracted.
            Takes precedence over module if provided. Defaults to None.
        weight_decay (float, optional): Weight decay value to apply. If None,
            returns all parameters without grouping. Defaults to None.
        check_skip_list (bool, optional): Whether to check for parameters that
            should skip weight decay. Defaults to True.
    Returns:
        list: If weight_decay is None, returns a simple list of parameters.
              If weight_decay is provided, returns parameter groups with
              appropriate weight decay settings.
    """
    
    # If named_params is provided, give priority to it
    if named_params is not None:
        module_or_named_params = named_params
    else:
        module_or_named_params = module
    
    if weight_decay is None:
        if hasattr(module_or_named_params, "parameters"):
            # nn.Module
            return list(module_or_named_params.parameters())
        else:
            # NamedParams
            return [p for _, p in module_or_named_params]
      
    no_wd_names = []      
    if check_skip_list:
        no_wd_names = get_params_names_with_no_weight_decay(module)
    return param_groups_weight_decay(module_or_named_params, weight_decay, no_wd_names)


def merge_param_groups(*param_groups_list):
    """
    Merges multiple parameter groups into a single list of parameter groups.
    Args:
        *param_groups_list: A variable number of lists, where each list contains parameter groups.
                            Each parameter group can either be a list of parameters or a list of dictionaries
                            with 'params' and optionally 'weight_decay' keys.
    Returns:
        A list of merged parameter groups. If the input contains lists of parameters, they are concatenated.
        If the input contains lists of dictionaries, the dictionaries are merged based on their 'weight_decay' values if present.
        Each resulting dictionary will contain a 'params' key and, if applicable, a 'weight_decay' key.
    """
    if not param_groups_list:
        return []
        
    # If param_groups_list is a list of lists of parameters
    if all(not isinstance(groups, list) or not isinstance(groups[0], dict) for groups in param_groups_list):
        return [p for groups in param_groups_list for p in groups]
    
    # Else, param_groups_list is a list of lists of dictionaries, each containing for sure a 'params' key and possibly a 'weight_decay' key
    decay_to_params = {}
    for groups in param_groups_list:
        if not groups:
            continue
              
        for group in groups:
            wd = group.get('weight_decay', None)
            if wd not in decay_to_params:
                decay_to_params[wd] = []
            decay_to_params[wd].extend(group['params'])

    return [{'params': params, 'weight_decay': wd} if wd is not None else {'params': params} for wd, params in decay_to_params.items()]
    
    
def collect_params_from_modules(modules, weight_decay=None, check_skip_list=True):
    """
    Collects and merges parameters from multiple modules for optimization.

    This function processes a list of module configurations, extracts their parameters
    based on specified criteria, and merges them into a single parameter group list
    suitable for optimizers.

    Args:
        modules (list): List of dictionaries containing module configurations.
            Each dictionary should have:
            - 'module' (nn.Module): The actual module instance
            - 'named_params' (NamedParams, optional): Specific named parameters to include
            - 'log' (bool, optional): Whether to log the number of parameters added from the module

        weight_decay (float, optional): Weight decay value to apply to parameters.
            If None, uses default weight decay settings.
        
        check_skip_list (bool, optional): Whether to check for parameters that should
            be skipped from weight decay. Defaults to True.

    Returns:
        list: Merged parameter groups ready for optimizer consumption.

    Example:
        modules = [
            {'module': layer1,},
            {'module': layer2, 'named_params': ['weight'], 'log': True}
        ]
        params = collect_params_from_modules(modules, weight_decay=0.01)
    """
    all_params = []
    for item in modules:
        if item is None:
            continue
        module = item['module']
        named_params = item.get('named_params', None)
        log = item.get('log', False)
        params = get_params_with_decay(module, named_params, weight_decay, check_skip_list=check_skip_list)
        if params:
            all_params.append(params)
            if log:
                name = module._get_name() if hasattr(module, "_get_name") else module.__class__.__name__
                param_count = sum(len(g['params']) for g in params) if isinstance(params[0], dict) else len(params)
                print(f"Adding {param_count} trainable parameters from {name}.")
        
    return merge_param_groups(*all_params)
 
 
def find_weight_decay(optimizer, optimizer_config=None, key=None):
    """
    This function attempts to extract the weight decay parameter from either a provided optimizer
    (possibly partially instantiated) or a configuration dictionary. It follows several
    fallback strategies to locate the weight decay value.

    Args:
        optimizer (torch.optim.Optimizer or functools.partial): The optimizer instance or a 
            partial function that returns an optimizer when called with parameters.
        optimizer_config (dict, optional): Configuration dictionary that may contain weight decay
            settings.
        key (str, optional): Key to access a nested optimizer configuration. Defaults to None.
    Returns:
        float or None: The weight decay value if found, None otherwise.
    Example:
        >>> from functools import partial
        >>> from torch.optim import Adam
        >>> opt = partial(Adam, weight_decay=0.01)
        >>> config = {'weight_decay': 0.02}
        >>> find_weight_decay(opt, config)
        0.02
        
        >>> opt = partial(Adam, weight_decay=0.01)
        >>> config = {'model1': {'weight_decay': 0.02}}
        >>> find_weight_decay(opt, config, key='model1')
        0.02
        
        >>> opt = partial(Adam, weight_decay=0.01)
        >>> config = {}
        >>> find_weight_decay(opt, config)
        0.01
    """
    if optimizer_config is not None:
        if key is not None and key in optimizer_config:
            optimizer_config = optimizer_config[key]
            
        if 'weight_decay' in optimizer_config['params']:
            return optimizer_config['params']['weight_decay']
    
    # Check if optimzier is a partial function
    if isinstance(optimizer, partial):
        # Instantiate a dummy optimizer
        optimizer = optimizer(nn.Linear(1,1).parameters())
    
    if 'weight_decay' in optimizer.defaults:
        return optimizer.defaults['weight_decay']
    
    if hasattr(optimizer, 'weight_decay'):
        return optimizer.weight_decay
    
    return None

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)
    
    
class Conv1dSamePadding(nn.Conv1d):
    def __init__(self,*args, padding=0, **kwargs):
        self.same_padding = False
        if padding == 'same':
            self.same_padding = True
            padding = 0
        super().__init__(*args, padding=padding, **kwargs)
        if self.same_padding:
            self.zero_pad_1d = nn.ZeroPad1d(reduce(__add__,
                [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        if self.same_padding:
            input = self.zero_pad_1d(input)
        return super().forward(input)


# https://github.com/davrot/pytorch_sequence_tools/blob/25c8c9df227f1632da0c55fe72718eb6e11c2119/Functional2Layer.py#L5    
class Functional2Layer(torch.nn.Module):
    def __init__(
        self, func: Callable[..., torch.Tensor], *args: Any, **kwargs: Any
    ) -> None:
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        
        # I don't know if this is a good idea, but it's the only way
        # to show the correct name in the summary produced by torchinfo
        # since (as of 21 November 2024) torchinfo uses the class name
        # instead of _get_name()
        self.__class__.__name__ = self._get_name()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.func(input, *self.args, **self.kwargs)

    def extra_repr(self) -> str:
        func_name = (
            self.func.__name__ if hasattr(self.func, "__name__") else str(self.func)
        )
        args_repr = ", ".join(map(repr, self.args))
        kwargs_repr = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"func={func_name}, args=({args_repr}), kwargs={{{kwargs_repr}}}"
    
    def _get_name(self) -> str:
        func_name = (
            self.func.__name__ if hasattr(self.func, "__name__") else str(self.func)
        )
        return f"{func_name.capitalize()}2Layer"


#### Latent Diffusion stuff ####

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        if 'padding' in kwargs and kwargs['padding'] == 'same':
            return Conv1dSamePadding(*args, **kwargs)
        else:
            return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )