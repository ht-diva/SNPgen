import math
from typing import Any, Dict, Optional, Union
from tqdm import tqdm

import lightning.pytorch as pl

from snpgen.utils import is_notebook

BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]"

class SimpleProgressBar(pl.callbacks.ProgressBar):
    def __init__(self):
        super().__init__()
        self.bar = None
        self.val_bar = None
        self.enabled = True
        self.is_notebook = is_notebook()

    def remove_metrics(self, metrics, prefix, postfix):
        return {k: v for k, v in metrics.items() if not k.startswith(prefix) and not k.endswith(postfix)}

    def on_train_epoch_start(self, trainer, pl_module):
        if self.enabled:
            self.bar = tqdm(
                total=convert_inf(self.total_train_batches),
                desc=f"Epoch {trainer.current_epoch+1}",
                position=0,
                leave=True,
                dynamic_ncols=True,
                #file=sys.stdout,
                #smoothing=0,
                bar_format=BAR_FORMAT,
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.bar:
            self.bar.update(1)
            metrics = self.get_metrics(trainer, pl_module)
            # remove validation metrics and metrics ending with '_epoch' because they are from previous epoch
            metrics = self.remove_metrics(metrics, prefix='val/', postfix='_epoch') 
            self.bar.set_postfix(metrics)

    def on_validation_batch_start(self, trainer, pl_module,batch, batch_idx, dataloader_idx=0):
        if not self.is_notebook:
            self._current_eval_dataloader_idx = dataloader_idx
            if self.enabled and self.val_bar is None:
                self.val_bar = tqdm(
                    total=convert_inf(self.total_val_batches_current_dataloader),
                    desc=f"Validation",
                    position=1,
                    leave=False,
                    dynamic_ncols=True,
                    #file=sys.stdout,
                    #smoothing=0,
                    bar_format=BAR_FORMAT,
                )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.is_notebook and self.val_bar:
            self.val_bar.update(1)

    def on_validation_end(self, trainer, pl_module):
        if not self.is_notebook and self.val_bar:
            self.val_bar.close()
            self.val_bar = None
            
    def on_train_epoch_end(self, trainer, pl_module):
        if self.bar:
            self.bar.set_postfix(self.get_metrics(trainer, pl_module))
            self.bar.close()
            self.bar = None
            print('')
            
    def on_train_end(self, trainer, pl_module):
        if self.bar:
            self.bar.close()
            self.bar = None
        if self.val_bar:
            self.val_bar.close()
            self.val_bar = None

    def disable(self):
        if self.bar:
            self.bar.close()
        if self.val_bar:
            self.val_bar.close()
        self.bar = None
        self.val_bar = None
        self.enabled = False


def convert_inf(x: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
    """The tqdm doesn't support inf/nan values.

    We have to convert it to None.

    """
    if x is None or math.isinf(x) or math.isnan(x):
        return None
    return x