import math
import sys
from typing import Any, Dict, Optional, Union
from tqdm import tqdm

import lightning.pytorch as pl

from snpgen.utils import is_notebook

BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]"


class SimpleProgressBar(pl.callbacks.ProgressBar):
    """Lightweight tqdm progress bar for training and validation.

    In interactive mode (TTY), shows a live tqdm bar updated every step.
    In non-interactive mode (e.g. SLURM log files), disables tqdm entirely
    and prints a one-line summary every *print_every_n_steps* steps instead,
    avoiding the flood of incremental lines that tqdm emits when stderr is
    not a TTY.
    """

    def __init__(self, print_every_n_steps: int = 300) -> None:
        super().__init__()
        self.bar: Optional[tqdm] = None
        self.val_bar: Optional[tqdm] = None
        self.enabled = True
        self.is_notebook = is_notebook()
        # True when running in a real terminal (interactive), False in SLURM/batch
        self._interactive: bool = sys.stderr.isatty()
        self.print_every_n_steps = print_every_n_steps

    def remove_metrics(self, metrics: dict, prefix: str, postfix: str) -> dict:
        return {k: v for k, v in metrics.items()
                if not k.startswith(prefix) and not k.endswith(postfix)}

    @staticmethod
    def _format_metrics(metrics: dict) -> str:
        parts = []
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4g}")
            else:
                parts.append(f"{k}={v}")
        return ", ".join(parts)

    # -- Training ----------------------------------------------------------

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if self.enabled and self._interactive and trainer.is_global_zero:
            self.bar = tqdm(
                total=convert_inf(self.total_train_batches),
                desc=f"Epoch {trainer.current_epoch + 1}",
                position=0,
                leave=True,
                dynamic_ncols=True,
                bar_format=BAR_FORMAT,
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        metrics = self.get_metrics(trainer, pl_module)
        metrics = self.remove_metrics(metrics, prefix='val/', postfix='_epoch')

        if self.bar:
            # Interactive: update live bar
            self.bar.update(1)
            self.bar.set_postfix(metrics)
        elif self.enabled and not self._interactive and trainer.is_global_zero:
            # Non-interactive: throttled plain-text logging
            step = batch_idx + 1
            total = convert_inf(self.total_train_batches)
            if step % self.print_every_n_steps == 0 or step == total:
                epoch = trainer.current_epoch + 1
                total_str = str(total) if total is not None else "?"
                print(
                    f"Epoch {epoch} [{step}/{total_str}] "
                    f"{self._format_metrics(metrics)}",
                    flush=True,
                )

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.bar:
            self.bar.set_postfix(self.get_metrics(trainer, pl_module))
            self.bar.close()
            self.bar = None
            print('')

    # -- Validation --------------------------------------------------------

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx: int = 0
    ) -> None:
        if self.is_notebook:
            return
        self._current_eval_dataloader_idx = dataloader_idx
        if self.enabled and self._interactive and self.val_bar is None and trainer.is_global_zero:
            self.val_bar = tqdm(
                total=convert_inf(self.total_val_batches_current_dataloader),
                desc="Validation",
                position=1,
                leave=False,
                dynamic_ncols=True,
                bar_format=BAR_FORMAT,
            )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not self.is_notebook and self.val_bar:
            self.val_bar.update(1)

    def on_validation_end(self, trainer, pl_module) -> None:
        if self.val_bar:
            self.val_bar.close()
            self.val_bar = None
        elif self.enabled and not self._interactive and trainer.is_global_zero:
            # Print a one-line validation summary in non-interactive mode
            metrics = self.get_metrics(trainer, pl_module)
            val_metrics = {k: v for k, v in metrics.items() if k.startswith("val/")}
            if val_metrics:
                epoch = trainer.current_epoch + 1
                print(
                    f"Epoch {epoch} [val] {self._format_metrics(val_metrics)}",
                    flush=True,
                )

    # -- Cleanup -----------------------------------------------------------

    def on_train_end(self, trainer, pl_module) -> None:
        if self.bar:
            self.bar.close()
            self.bar = None
        if self.val_bar:
            self.val_bar.close()
            self.val_bar = None

    def disable(self) -> None:
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