import inspect
from inspect import Parameter
from functools import reduce

def default_args(cls):
    argspec = dict(inspect.signature(cls.__init__).parameters)
    argspec.pop("self")
    default_args = {
        param: argspec[param].default
        for param in argspec
        if (argspec[param] != Parameter.empty
            and argspec[param].default != Parameter.empty)
    }
    return default_args


def args_names(cls):
    argspec = dict(inspect.signature(cls.__init__).parameters)
    argspec.pop("self")
    return list(argspec.keys())


def get_config_value(key, cfg):
    return reduce(lambda c, k: c[k], key.split('.'), cfg)