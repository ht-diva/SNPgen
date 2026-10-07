"""Reusable checkpoint callback for periodic and best-validation snapshots."""

from __future__ import annotations

import glob
import os
from typing import Any

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException


class ValidationAndPeriodicCheckpoint(Callback):
    """Save periodic checkpoints and keep one best validation checkpoint.

    This callback calls ``trainer.save_checkpoint`` directly instead of relying
    on Lightning's ``ModelCheckpoint`` internals. That keeps checkpointing
    usable for manual-update modules whose ``global_step`` may not advance.
    """

    def __init__(
        self,
        dirpath: str,
        monitor: str = "val/loss",
        mode: str = "min",
        every_n_epochs: int | None = None,
        periodic_filename: str = "epoch={epoch}-step={step}",
        best_filename: str = "best-epoch={epoch}-step={step}-val_loss={val_loss:.4f}",
        metric_filename_key: str | None = None,
        save_last: bool = False,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if every_n_epochs is not None and int(every_n_epochs) <= 0:
            raise ValueError("every_n_epochs must be a positive integer or None")
        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode
        self.every_n_epochs = None if every_n_epochs is None else int(every_n_epochs)
        self.periodic_filename = periodic_filename
        self.best_filename = best_filename
        self.metric_filename_key = metric_filename_key
        self.save_last = bool(save_last)
        self.best_score: float | None = None
        self.best_path: str | None = None

    @property
    def state_key(self) -> str:
        return f"{self.__class__.__qualname__}[{self.monitor},{self.mode},{self.every_n_epochs}]"

    def state_dict(self) -> dict[str, Any]:
        return {"best_score": self.best_score, "best_path": self.best_path}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.best_score = state_dict.get("best_score")
        self.best_path = state_dict.get("best_path")

    def _metric_aliases(self, value: float | torch.Tensor) -> dict[str, float | torch.Tensor]:
        aliases = {
            self.monitor: value,
            self.monitor.replace("/", "_"): value,
        }
        if self.monitor == "val/loss":
            aliases["val_loss"] = value
        if self.metric_filename_key:
            aliases[self.metric_filename_key] = value
        return aliases

    def _monitor_candidates(self, trainer) -> dict[str, float | torch.Tensor]:
        metrics = dict(trainer.callback_metrics)
        epoch = metrics.get("epoch")
        metrics["epoch"] = int(epoch.item()) if isinstance(epoch, torch.Tensor) and epoch.numel() == 1 else int(trainer.current_epoch)
        step = metrics.get("step")
        metrics["step"] = int(step.item()) if isinstance(step, torch.Tensor) and step.numel() == 1 else int(trainer.global_step)
        return metrics

    def _format_path(
        self,
        template: str,
        trainer,
        metric_value: float | torch.Tensor | None = None,
    ) -> str:
        epoch = int(trainer.current_epoch)
        metrics = {"epoch": epoch, "step": int(trainer.global_step)}
        if metric_value is not None:
            metrics.update(self._metric_aliases(metric_value))
        basename = template.format(**metrics)
        if not basename.endswith(".ckpt"):
            basename = f"{basename}.ckpt"
        return os.path.join(self.dirpath, basename)

    def _save(self, trainer, path: str) -> None:
        os.makedirs(self.dirpath, exist_ok=True)
        trainer.save_checkpoint(path)
        if getattr(trainer, "is_global_zero", True):
            for logger in getattr(trainer, "loggers", []):
                logger.after_save_checkpoint(self)

    def _monitor_value(self, trainer) -> float | None:
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return None
        value = metrics[self.monitor]
        if isinstance(value, torch.Tensor):
            value = value.detach()
            if value.numel() != 1:
                return None
            value = value.item()
        value = float(value)
        if not torch.isfinite(torch.tensor(value)):
            return None
        return value

    def _is_better(self, value: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return value < self.best_score
        return value > self.best_score

    def _remove_checkpoint(self, trainer, path: str) -> None:
        strategy = getattr(trainer, "strategy", None)
        if strategy is not None and hasattr(strategy, "remove_checkpoint"):
            strategy.remove_checkpoint(path)
        elif os.path.exists(path):
            os.remove(path)

    def _remove_previous_best(self, trainer, current_path: str) -> None:
        candidates = []
        if self.best_path:
            candidates.append(self.best_path)
        candidates.extend(glob.glob(os.path.join(self.dirpath, "best-*.ckpt")))
        for path in set(candidates):
            # best_path is restored from checkpoint state and may refer to a
            # different run directory. Only remove best checkpoints owned by
            # this callback's configured directory.
            owned_path = (
                os.path.realpath(os.path.dirname(path)) == os.path.realpath(self.dirpath)
                and os.path.basename(path).startswith("best-")
                and path.endswith(".ckpt")
            ) if path else False
            if owned_path and path != current_path and os.path.exists(path):
                self._remove_checkpoint(trainer, path)

    def on_validation_end(self, trainer, pl_module) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        value = self._monitor_value(trainer)
        if value is None:
            raise MisconfigurationException(
                f"`ValidationAndPeriodicCheckpoint(monitor={self.monitor!r})` could not find "
                f"the monitored key in callback metrics: {list(trainer.callback_metrics)}."
            )
        if not self._is_better(value):
            return
        new_path = self._format_path(self.best_filename, trainer, metric_value=value)
        self._remove_previous_best(trainer, new_path)
        self._save(trainer, new_path)
        self.best_score = value
        self.best_path = new_path

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if not self.every_n_epochs:
            return
        epoch = int(trainer.current_epoch) + 1
        if epoch % self.every_n_epochs == 0:
            self._save(trainer, self._format_path(self.periodic_filename, trainer))

    def on_train_end(self, trainer, pl_module) -> None:
        if self.save_last:
            self._save(trainer, os.path.join(self.dirpath, "last.ckpt"))
