import torch
import lightning.pytorch as pl
import math

from typing import Any, Dict, List, Optional, Tuple, Union

from snpgen.utils import default, disabled_train, instantiate_from_config

from snpgen.models.modules.ddpm.wrappers import OpenAIWrapper
from snpgen.models.modules.embedding import UNCONDITIONAL_CONFIG
from snpgen.training.lr_schedulers import LambdaWarmUpCosineScheduler
from .base import BaseEngine

from torch.optim.lr_scheduler import LambdaLR


class DiffusionEngine(BaseEngine):

    EMA_TRACKED_MODULES = ['model']
    IGNORED_HPARAMS = ['network', 'denoiser', 'first_stage', 'conditioner', 'sampler', 'loss_fn']

    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        learning_rate,
        conditioner_config = None,
        sampler_config = None,
        use_lr_scheduler = False,
        loss_fn_config = None,
        use_ema = False,
        ema_config = {},
        scale_factor: float = 1.0,
        lr_scheduler_warmup_epochs: int = 5,
        disable_first_stage_autocast: bool = False,
        input_key: str = "x",
        compile_model: bool = False,
        en_and_decode_n_samples_a_time: Optional[int] = None,
    ):
        super().__init__(use_ema=use_ema)
        self.save_hyperparameters(ignore=self.ignored_hparams)

        self.input_key = input_key
        self.learning_rate = learning_rate
        
        model = instantiate_from_config(network_config)
        self.model = OpenAIWrapper(model, compile_model=compile_model)

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_scheduler_warmup_epochs = lr_scheduler_warmup_epochs
        
        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast

        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

        self._init_first_stage(first_stage_config)
        self.setup_ema_model(**ema_config)

    def _init_first_stage(self, config):
        model = instantiate_from_config(config)
        model.eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        return batch[self.input_key]

    @torch.no_grad()
    def decode_first_stage(self, z, **kwargs):
        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])

        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.decode(
                    z[n * n_samples : (n + 1) * n_samples], **kwargs
                )
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x, **kwargs):
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(
                    x[n * n_samples : (n + 1) * n_samples], **kwargs
                )

                if 'sample' in kwargs and kwargs['sample'] == False:
                    out = out[0] # only return the mean

                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        return z

    def forward(self, x, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        x = self.encode_first_stage(x, sample=True)
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        return loss
  
    def on_train_start(self, *args, **kwargs):
        super().on_train_start()
        if self.loss_fn is None:
            raise ValueError("Loss function need to be set for training.") 

    def configure_optimizers(self):
        lr = self.learning_rate

        params = list(self.model.parameters())
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())

        opt = torch.optim.AdamW(params, lr=lr)

        if self.use_lr_scheduler:
            print("Setting up LambdaLR scheduler...")
            scheduler = LambdaWarmUpCosineScheduler(
                warm_up_steps=self.lr_scheduler_warmup_epochs*self.num_steps_per_epoch,
                f_min=0., f_max=1., f_start=1e-6,
                cycle_length=self.num_training_steps
            )
            
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)

        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **kwargs
        )
        samples = self.sampler(denoiser, randn, cond, uc=uc)
        return samples
