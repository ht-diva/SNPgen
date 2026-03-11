import torch
import torch.nn as nn
import lightning.pytorch as pl

from contextlib import contextmanager, nullcontext

from typing import Any, Dict, List, Optional, Sequence

from lightning.pytorch.callbacks import Callback

from snpgen.training.callbacks.ema import ModelEmaV3, EMACallbackV3, EMAScope
from snpgen.utils import rank_zero_info_once
from snpgen.models.modules.utils import get_params_names_with_no_weight_decay

class BaseEngine(pl.LightningModule):

    IGNORED_HPARAMS = []

    def __new__(cls, *args, **kwargs):
        if cls is BaseEngine:
            raise TypeError(f"You are trying to instantiate an abstract class {cls.__name__}. Please use a concrete subclass.")
        return super().__new__(cls)
    
    def __init__(self, use_ema=False):
        super().__init__()
        
        # Disable profiling executor. This reduces memory and increases speed.
        try:
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
        except AttributeError:
            pass

        self.use_ema = use_ema

    def setup_ema_model(
            self,
            #model: torch.nn.Module,
            decay: float = 0.9999,
            min_decay: float = 0.0,
            update_after_step: int = 0,
            use_warmup: bool = False,
            warmup_gamma: float = 1.0,
            warmup_power: float = 2/3,
            use_karras: bool = False,
            karras_gamma: Optional[float] = None,
            device: Optional[torch.device] = None,
            exclude_buffers: bool = False,
            verbose: bool = False,
            **kwargs
        ):
        ''' Setup the EMA model if the needed.'''
        if self.use_ema:
            model = self.get_tracked_ema_module()
            self.model_ema = ModelEmaV3(
                model,
                decay=decay,
                min_decay=min_decay,
                update_after_step=update_after_step,
                use_warmup=use_warmup,
                warmup_gamma=warmup_gamma,
                warmup_power=warmup_power,
                use_karras=use_karras,
                karras_gamma=karras_gamma,
                device=device,
                exclude_buffers=exclude_buffers,
                verbose=verbose,
                **kwargs
            )
            self._ema_scope = EMAScope(store_device=self.model_ema.device)
    
    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        callbacks = super().configure_callbacks()
        if getattr(self, "use_ema", False):
            assert hasattr(self, "model_ema"), "You need to instantiate the 'model_ema' in your engine."
            # Add the callback responsible for updating the EMA weights at the end of each training batch.
            self._ema_callback = EMACallbackV3(
                    ema_scope=self._ema_scope,
                    validate_with_ema_weights=False,
                    use_ema_weights=False
                )
            callbacks.append(self._ema_callback)
        return callbacks

    def get_tracked_ema_module(self) -> nn.Module | None:
        ''' Returns the module that should be tracked by EMA.'''
        if self.use_ema:
            ema_modules = self.get_tracked_ema_modules()
            if len(ema_modules) > 1:
                rank_zero_info_once(
                    "You specified more than one module to be tracked with EMA,"
                    f" but currently the {self.model_ema.__class__.__name__} class only supports tracking one module."
                    " The first specified module will be used."
                )
            return getattr(self, ema_modules[0])

    def get_tracked_ema_modules(self) -> List[str]:
        out = []
        if self.use_ema:
            out = self.EMA_TRACKED_MODULES
        return out
    
    def compute_metric(self, metric, y, y_pred, return_dict=True):
        '''
            return_dict: whether to return a dictionary of metrics or update the metrics object.
                If False, use cls_metrics.compute() to get the actual metrics when needed.
                Also, remember to reset the metrics at the end of the epoch using cls_metrics.reset()
        '''        
        if return_dict:
            output = metric(y_pred, y)
            return output
        else:
            metric.update(y_pred, y)
            
    @property
    def params_names_with_no_weight_decay(self):
        return get_params_names_with_no_weight_decay(self)
    
    @property
    def ignored_hparams(self):
        ignores = self.IGNORED_HPARAMS
        classes = self.__class__.__bases__
        while True:
            new_classes = []
            for cls in classes:
                if hasattr(cls, 'IGNORED_HPARAMS'):
                    ignores.extend(cls.IGNORED_HPARAMS)
                    new_classes.extend([c for c in cls.__bases__])
            classes = new_classes
            if len(new_classes) == 0:
                break
                
        return list(set(ignores))
        
    def ema_scope(self, context=None):
        if not self.use_ema:
            return nullcontext()
        else:
            return self._ema_scope.ema_scope(pl_module=self, context=context)

    def on_train_start(self):
        super().on_train_start()
        if getattr(self, 'samples_seen', None) is None:
            self.samples_seen = 0    

    # Track number of samples seen to use it as x-axis in W&B to compare runs
    # with different batch sizes (e.g. when training on multiple GPUs)
    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)

        if isinstance(batch, (list, tuple)):
            batch_size = batch[0].shape[0]
        elif isinstance(batch, dict):
            batch_size = batch[list(batch.keys())[0]].shape[0]
        else:
            batch_size = batch.shape[0]
        self.samples_seen += batch_size * self.trainer.num_devices * self.trainer.num_nodes # actual batch size
        self.logger.log_metrics({'trainer/samples_seen': self.samples_seen})

    @property
    def num_steps_per_epoch(self) -> int:
        """
        Get number of steps per epoch.
        Useful because when configure_optimizers() is called, self.trainer.num_training_batches is not yet set.
        """
        if self.trainer.train_dataloader is None:
            self.trainer.fit_loop.setup_data()
        return len(self.trainer.train_dataloader) // self.trainer.accumulate_grad_batches

    @property
    def num_training_steps(self) -> int:
        """
        Get number of training steps
        Useful because when configure_optimizers() is called, self.trainer.num_training_batches is not yet set.
        """
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps

        if self.trainer.train_dataloader is None:
            self.trainer.fit_loop.setup_data()
        dataset_size = len(self.trainer.train_dataloader) # len(self.trainer.train_dataloader) = len(dataset)/(batchsize * num_devices)
        num_steps = dataset_size * self.trainer.max_epochs // self.trainer.accumulate_grad_batches

        return num_steps