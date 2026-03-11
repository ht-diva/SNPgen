import os
import functools
from functools import wraps
from inspect import isfunction
import logging
import importlib
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from collections import abc

import torch
from lightning.pytorch.utilities import rank_zero_only, rank_zero_info, rank_zero_warn

# logger

def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger

logpy = get_pylogger(__name__)

# helpers

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)
print_rank_zero = rank_zero_only(print)
print_rank_zero_once = once(print_rank_zero)
rank_zero_info_once = once(rank_zero_info)
rank_zero_warn_one = once(rank_zero_warn)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def expand_dims_like(x, y):
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def save_config(config, dir):
    os.makedirs(dir, exist_ok=True) 
    OmegaConf.save(config, f"{dir}/config.yaml", resolve=False)
    OmegaConf.save(config, f"{dir}/config_resolved.yaml", resolve=True)


def instantiate_from_config(config, *args, **kwargs):
    """Create object from a config
    
    Args:
        config: A dictionary-like object containing the keys `target` and `params`.
        *args: Additional arguments to be passed to the object constructor.
        partial: If True, return a partial object.
        custom_partial_fn: A function which will be called when the partial function is actually called. 
                           It must accept an arbitrary number of arguments, which will be the arguments passed
                           to the partial function plus the keyword arguments passed to the custom_partial_fn_kwargs.
        custom_partial_fn_kwargs: A dictionary of keyword arguments to be passed to the custom_partial_fn.
        **kwargs: Additional keyword arguments to be passed to the object constructor.
    """
    
    partial = kwargs.pop("partial", False)
    custom_partial_fn = kwargs.pop("custom_partial_fn", None)
    custom_partial_fn_kwargs = kwargs.pop("custom_partial_fn_kwargs", {})

    try:
        'key' in config # check if config is a dict-like object
    except TypeError:
        return config
    
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    
    kwargs_params = config.get("params", dict())
    # Convert OmegaConf to dict to be able to update it with not supported primitive types (e.g. NumPy arrays)
    if isinstance(kwargs_params, DictConfig):
        kwargs_params = OmegaConf.to_object(kwargs_params)
    kwargs_params.update(kwargs)
    
    if partial:
        if custom_partial_fn is None:
            return functools.partial(get_obj_from_str(config["target"]), *args, **kwargs_params)
        else:
            def _inner_partial_fn(*args, **kwargs):
                custom_partial_fn(**kwargs, **custom_partial_fn_kwargs)
                return get_obj_from_str(config["target"])(*args, **kwargs)
            return functools.partial(_inner_partial_fn, **kwargs_params)
    else:
        return get_obj_from_str(config["target"])(*args, **kwargs_params)
    

def instantiate_multiple_objects_from_config(config, return_dict=True, **kwargs):
    result = {}
    if not 'target' in config:
        # config is a dict of multiple configs
        partial_optimizers = {}
        for k, v in config.items():
            partial_optimizers[k] = instantiate_from_config(
                v,
                **kwargs,
                custom_partial_fn_kwargs={'target': v['target'], 'log_key': k}
                )
        result = partial_optimizers
    else:
        # config is a single config
        result =  {'0': instantiate_from_config(
            config,
            **kwargs,
            custom_partial_fn_kwargs={'target': config['target'], 'log_key': None}
            )
        }

    if return_dict:
        return result
    else:
        return list(result.values())


def instantiate_partial_optimizers_from_config(config):

    def _log_fn(**kwargs):
        target = kwargs.pop('target', None)
        log_key = kwargs.pop('log_key', None)
        print(f"""Instantiating >>> {target} <<< optimizer from config{f" for model '{log_key}'" if log_key else ""} with params: {kwargs}""")
        if not 'lr' in kwargs:
            logpy.warning(f"""Optimizer {target}{f" (model: {log_key})" if log_key else ""} does not have the 'lr' parameter specified in the config.""")

    return instantiate_multiple_objects_from_config(
        config,
        return_dict=True,
        partial=True,
        custom_partial_fn=_log_fn
    )


def instantiate_partial_schedulers_from_config(config):

    def _log_fn(**kwargs):
        target = kwargs.pop('target', None)
        log_key = kwargs.pop('log_key', None)
        print(f"""Instantiating >>> {target} <<< scheduler from config{f" for model '{log_key}'" if log_key else ""} with params: {kwargs}""")

    return instantiate_multiple_objects_from_config(
        config,
        return_dict=True,
        partial=True,
        custom_partial_fn=_log_fn
    )
 

def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def scale_lr_optimizer_config(config, num_gpus=1):
    try:
        for key, value in config.items():
            if key == 'lr' or key == 'learning_rate':
                orig_val = value
                config[key] *= num_gpus
                logpy.info(f"Scaling learning rate from {orig_val} to {config[key]}")
            else:
                scale_lr_optimizer_config(value, num_gpus)
    except:
        pass


def get_proper_state_dict(load_path, keyword):
        
    def filter_fn(item):
        k, v = item
        return k.startswith(keyword)

    def convert_fn(item):
        k, v = item
        if k.startswith(keyword):
            k = k.replace(keyword, '', 1)
        return k, v
    
    _state_dict = torch.load(load_path, weights_only=False)['state_dict']  
    state_dict = dict(map(convert_fn, filter(filter_fn, _state_dict.items())))
    return state_dict


def supports_flash_attention(device_id=0):
    """Check if a GPU supports FlashAttention."""
    major, minor = torch.cuda.get_device_capability(device_id)
    
    # Check if the GPU architecture is Ampere (SM 8.x) or newer (SM 9.0)
    is_sm8x = major == 8 and minor >= 0
    is_sm90 = major == 9 and minor == 0

    return is_sm8x or is_sm90


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__ # type: ignore (it's ok that the function appears as not defined)
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    

def is_list_like(*objs, allow_sets=True, func=all):
    ''' Check if inputs are list-like
    Parameters
    ----------
    *objs : object
        Objects to check.
    allow_sets : bool, optional.
        If this parameter is `False`, sets will not be considered list-like.
        Default: `True`
    func : funtional object, optional.
        The function to be applied to each element. Useful ones are `all` and
        `any`.
        Default: `all`
    Notes
    -----
    Direct copy from pandas, with slight modification to accept *args and
    all/any, etc, functionality by `func`.
    https://github.com/pandas-dev/pandas/blob/bdb00f2d5a12f813e93bc55cdcd56dcb1aae776e/pandas/_libs/lib.pyx#L1026
    Note that pd.DataFrame also returns True.
    '''
       
    return func(
        isinstance(obj, abc.Iterable)
        # we do not count strings/unicode/bytes as list-like
        and not isinstance(obj, (str, bytes))
        # exclude zero-dimensional numpy arrays, effectively scalars
        and not (hasattr(obj, "ndim") and obj.ndim == 0)
        # exclude sets if allow_sets is False
        and not (allow_sets is False and isinstance(obj, abc.Set))
        for obj in objs
    )