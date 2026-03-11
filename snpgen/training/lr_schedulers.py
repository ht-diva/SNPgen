import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import LambdaLR

from snpgen.utils import instantiate_from_config

# From https://github.com/huggingface/transformers/blob/345b9b1a6a308a1fa6559251eb33ead2211240ac/src/transformers/optimization.py#L134
def get_cosine_schedule_with_warmup_lr_lambda(
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5
):
    def _get_cosine_schedule_with_warmup_lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return _get_cosine_schedule_with_warmup_lr_lambda


#######################################################################################################
#                                          Lambda Schedulers                                          #
#######################################################################################################
# Schedulers which are subclasses of LambdaLRScheduler, having a 'schedule' method that returns
# a multiplication factor for the learning rate. They should be used with torch.optim.lr_scheduler.LambdaLR

def LambdaLRWrapper(optimizer, scheduler_config, no_overwrite=False, **kwargs):
    _scheduler_config = scheduler_config.copy()
    
    # If no_overwrite is True, the scheduler_config will have higher priority than the kwargs
    if no_overwrite:
        cfg = {**kwargs, **_scheduler_config.get('params', dict())} # priority from right to left
    else:
        cfg = {**_scheduler_config.get('params', dict()), **kwargs} # priority from right to left
        
    _scheduler_config['params'] = cfg
    scheduler: LambdaLRScheduler = instantiate_from_config(_scheduler_config)
    return LambdaLR(optimizer, lr_lambda=scheduler.schedule)

class LambdaLRScheduler:
    def __init__(self, steps_per_epoch=-1, total_training_steps=-1):
        self.steps_per_epoch = steps_per_epoch
        self.total_training_steps = total_training_steps

    def schedule(self, n, **kwargs):
        raise NotImplementedError

# Adapted from https://github.com/Stability-AI/generative-models/blob/9d759324e914de6c96dbd1468b3a4a50243c6528/sgm/lr_scheduler.py#L4
class LambdaWarmUpCosineSchedulerMultiIterations(LambdaLRScheduler):
    """
    supports repeated iterations, configurable via lists
    """

    def __init__(
        self,
        f_min, f_max, f_start, # remember that these are multipliers of the base lr
        warm_up_epochs=[-1], warm_up_steps=[-1], cycle_length=[-1],
        verbosity_interval=0,
        **kwargs
    ):
        assert (
            len(warm_up_steps)
            == len(warm_up_epochs)
            == len(f_min)
            == len(f_max)
            == len(f_start)
            == len(cycle_length)
        )
        super().__init__(**kwargs)

        for i, c_l in enumerate(cycle_length):
            if c_l < 0:
                assert self.total_training_steps > 0, "Since 'cycle_length' is not set, 'total_training_steps' must be provided"
                cycle_length[i] = self.total_training_steps

        for i, (w_e, w_s) in enumerate(zip(warm_up_epochs, warm_up_steps)):
            if w_s >= 0:
                assert w_e == -1, "Both 'warmup_epochs' and 'warm_up_steps' cannot be set"
            else:
                assert w_e >= 0, "Either 'warmup_epochs' or 'warm_up_steps' must be set"
                assert self.steps_per_epoch > 0, "Since 'warm_up_epochs' is set, 'steps_per_epoch' must be provided"
                warm_up_steps[i] = w_e * self.steps_per_epoch

        print(
            f"Setting up {self.__class__.__name__}. warm_up_steps is {warm_up_steps}, "
            f"f_min is {f_min}, f_max is {f_max}, f_start is {f_start} and "
            f"cycle_length is {cycle_length}."
        )
        self.warm_up_steps = warm_up_steps
        self.f_min = f_min
        self.f_max = f_max
        self.f_start = f_start
        self.cycle_length = cycle_length
        self.cum_cycles = np.cumsum([0] + list(self.cycle_length))
        self.last_f = 0.0
        self.verbosity_interval = verbosity_interval

    def find_in_interval(self, n):
        interval = 0
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(
                    f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                    f"current cycle {cycle}"
                )
        if n < self.warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            t = (n - self.warm_up_steps[cycle]) / (
                self.cycle_length[cycle] - self.warm_up_steps[cycle]
            )
            t = min(t, 1.0)
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (
                1 + np.cos(t * np.pi)
            )
            self.last_f = f
            return f

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)
    
    def _get_values(self):
        x = np.arange(self.cum_cycles[-1])
        y = [self.schedule(i) for i in x]
        return x, y
    
    def plot(self):
        x, y = self._get_values()
        plt.plot(x, y)
        plt.show()

class LambdaWarmUpCosineScheduler(LambdaWarmUpCosineSchedulerMultiIterations):
    def __init__(
        self,
        f_min,
        f_max,
        f_start,
        warm_up_epochs=-1,
        warm_up_steps=-1,
        cycle_length=-1,
        verbosity_interval=0,
        **kwargs
    ):
        super().__init__(
            f_min=[f_min],
            f_max=[f_max],
            f_start=[f_start],
            warm_up_epochs=[warm_up_epochs],
            warm_up_steps=[warm_up_steps],
            cycle_length=[cycle_length],
            verbosity_interval=verbosity_interval,
            **kwargs
        )

class LambdaLinearScheduler(LambdaWarmUpCosineScheduler):
    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(
                    f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                    f"current cycle {cycle}"
                )

        if n < self.warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.warm_up_steps[
                cycle
            ] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (
                self.cycle_length[cycle] - n
            ) / (self.cycle_length[cycle])
            self.last_f = f
            return f
        

######################################################################################################
#                                          Torch Schedulers                                          #
######################################################################################################
# Schedulers which are subclasses of torch.optim.lr_scheduler._LRScheduler and return the scaled learning rate

class InverseLR(torch.optim.lr_scheduler._LRScheduler):
    """Implements an inverse decay learning rate schedule with an optional exponential
    warmup. When last_epoch=-1, sets initial lr as lr.
    inv_gamma is the number of steps/epochs required for the learning rate to decay to
    (1 / 2)**power of its original value.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        inv_gamma (float): Inverse multiplicative factor of learning rate decay. Default: 1.
        power (float): Exponential factor of learning rate decay. Default: 1.
        warmup (float): Exponential warmup factor (0 <= warmup < 1, 0 to disable)
            Default: 0.
        final_lr (float): The final learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, inv_gamma=1., power=1., warmup=0., final_lr=0., last_epoch=-1, **kwargs):

        print(
            f"Setting up {self.__class__.__name__}. inv_gamma is {inv_gamma}, "
            f"power is {power}, warmup is {warmup}, final_lr is {final_lr} and "
            f"last_epoch is {last_epoch}."
        )

        self.inv_gamma = inv_gamma
        self.power = power
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        self.warmup = warmup
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (1 + self.last_epoch / self.inv_gamma) ** -self.power
        return [warmup * max(self.final_lr, base_lr * lr_mult)
                for base_lr in self.base_lrs]
    
    def _get_values(self, n_steps):
        orig_last_epoch = self.last_epoch # backup

        self.last_epoch = 0
        y = []
        for i in range(n_steps):
            y.append(self._get_closed_form_lr()[0])
            self.last_epoch += 1
        x = np.arange(n_steps)

        self.last_epoch = orig_last_epoch # restore
        return x, y
    
    def plot(self, n_steps):
        x, y = self._get_values(n_steps)
        plt.plot(x, y)
        plt.show()