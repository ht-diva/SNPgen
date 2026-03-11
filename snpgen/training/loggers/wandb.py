from lightning.pytorch.loggers import WandbLogger
from typing import Dict, Optional, Union, List

def setup_wandb_logger(
        extra_config: Optional[Dict[str, str]] = None,
        extra_sync_metric: Optional[Union[str, List[str]]] = None,
        **kwargs
    ):

    wandb_logger = WandbLogger(**kwargs)

    if extra_config is not None:
        wandb_logger.experiment.config.update(extra_config)

    if extra_sync_metric is not None:
        if isinstance(extra_sync_metric, str):
            extra_sync_metric = [extra_sync_metric]
        for sync_metric in extra_sync_metric:
            wandb_logger.experiment.define_metric(sync_metric)
            wandb_logger.experiment.define_metric("*", step_metric=sync_metric, step_sync=True)

    return wandb_logger