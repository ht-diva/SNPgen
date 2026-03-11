import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy

from snpgen.utils import instantiate_from_config, instantiate_partial_optimizers_from_config, instantiate_partial_schedulers_from_config, default
from snpgen.models.modules.utils import collect_params_from_modules, find_weight_decay
from .base import BaseEngine


class AutoencoderTrainingWrapper(BaseEngine):

    EMA_TRACKED_MODULES = ['autoencoder']
    IGNORED_HPARAMS = ['autoencoder']

    def __init__(
        self,
        autoencoder_config,
        loss_config,
        optimizer_config,
        scheduler_config=None,
        use_ema=False,
        ema_config={},
        convert_to_onehot=True,
        channel_first=True,
        use_lr_scheduler=False,
    ):
        super().__init__(use_ema=use_ema)
        self.save_hyperparameters(ignore=self.ignored_hparams)
        self.autoencoder = instantiate_from_config(autoencoder_config)

        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.Adam"}
        )

        self.scheduler_config = scheduler_config
        self.convert_to_onehot = convert_to_onehot
        self.channel_first = channel_first
        self.use_lr_scheduler = use_lr_scheduler

        self.loss = instantiate_from_config(loss_config)

        self.setup_ema_model(**ema_config)
        self.build_metrics()

    def build_metrics(self):
        # Setup reconstruction metric
        # num_classes is the number possible values in the input sequence
        input_ch = getattr(self.autoencoder.encoder, 'input_ch', None) or getattr(self.autoencoder.encoder, 'in_channels', None)
        assert input_ch is not None, "Could not find the number of input channels for the encoder."
        self.recons_metrics = MetricCollection({
            "metrics/recons/accuracy": MulticlassAccuracy(num_classes=input_ch, multidim_average='global', average='micro') 
        })
        self.train_recons_metrics = self.recons_metrics.clone(prefix='train/')
        self.val_recons_metrics = self.recons_metrics.clone(prefix='val/')

    def compute_recons_metrics(self, x, recons, split='train', return_dict=True):
        if split == 'train':
            recons_metrics = self.train_recons_metrics
        elif split == 'val':
            recons_metrics = self.val_recons_metrics
        else:
            raise ValueError(f"Invalid split: {split}")
        
        if not self.channel_first:
            # If the reconstruction are in channel last format, we need to permute them to channel first
            # to compute the metrics
            recons = recons.permute(0, 2, 1)
        
        return self.compute_metric(recons_metrics, x, recons, return_dict=return_dict) 

    def get_last_layer(self):
        return self.autoencoder.get_last_layer()
    
    def check_shape(self, x):
        if self.convert_to_onehot:
            if len(x.shape) == 2:
                # Convert input to one-hot
                target = x.to(torch.long)  # long required for nn.CrossEntropyLoss
                x = torch.nn.functional.one_hot(x.to(torch.long), num_classes=-1).to(torch.float)
                # Apply channel first if needed
                if self.channel_first:
                    x = x.permute(0, 2, 1)
            elif len(x.shape) == 3:
                # Input already one-hot
                target = x.argmax(dim=1 if self.channel_first else 2).to(torch.long)
                x = x.to(torch.float)
                # Ensure channels are in the correct dimension if needed
                if self.channel_first and x.shape[2] < x.shape[1]:  # If channels are last but should be first
                    x = x.permute(0, 2, 1)
                elif not self.channel_first and x.shape[1] < x.shape[2]:  # If channels are first but should be last
                    x = x.permute(0, 1, 2)
            else:
                raise ValueError(f"Invalid x shape: {x.shape}")
        else:
            x = x.to(torch.int32)
            target = x.to(torch.long)
        return x, target

    def get_autoencoder_params(self, weight_decay=None) -> list:
        modules = [
            {'module': self.autoencoder}
        ]
        
        if hasattr(self.loss, "get_trainable_autoencoder_parameters"):
            modules.append(
                {'module': self.loss, 'named_params': self.loss.get_trainable_autoencoder_parameters(named=True), 'log': True}
            )
            
        params = collect_params_from_modules(modules, weight_decay, check_skip_list=True)
        return params
    
    def get_batch_data(self, batch):
        ''' Extracts the data from the batch. If the batch is a list or tuple, it assumes that the first element is the data
            and (probably) the second element is the target. '''
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0]
        elif isinstance(batch, dict):
            return batch['x']
        else:
            return batch
        
    def get_onehot_input(self, x):
        if len(x.shape) == 2:
            x = torch.nn.functional.one_hot(x.to(torch.long), num_classes=-1).permute(0,2,1).to(torch.float)
        return x

    def inner_training_step(self, batch, batch_idx, loss_kwargs={}):
        x = self.get_batch_data(batch)
        x, target = self.check_shape(x)

        # Encoder
        mu, logvar = self.autoencoder.encode(x)

        # Reparameterization Trick
        z = self.autoencoder.reparameterize(mu, logvar)

        # Decoder
        reconstructions = self.autoencoder.decode(z, argmax=False)

        # Compute loss
        extra_info = {
            "mean": mu,
            "logvar": logvar,
            "split": "train",
            "channel_first": self.channel_first,
            "oh_inputs": self.get_onehot_input(x),
        }
        loss_kwargs.update(extra_info)
        loss, log_dict = self.loss(target, reconstructions, **loss_kwargs)

        # Compute reconstruction metrics
        recons_metrics_log = self.compute_recons_metrics(target, reconstructions, split='train', return_dict=True)
        log_dict.update(recons_metrics_log)

        # Log metrics
        self.log_dict(
            log_dict,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        return loss
 
    def training_step(self, batch, batch_idx):
        loss = self.inner_training_step(batch, batch_idx)
        return loss
    
    def inner_validation_step(self, batch, batch_idx, loss_kwargs={}):
        x = self.get_batch_data(batch)
        x, target = self.check_shape(x)

        # Encoder
        mu, logvar = self.autoencoder.encode(x)

        # Reparameterization Trick
        z = self.autoencoder.reparameterize(mu, logvar)

        # Decoder
        reconstructions = self.autoencoder.decode(z, argmax=False)

        # Compute loss
        extra_info = {
            "mean": mu,
            "logvar": logvar,
            "split": "val",
            "channel_first": self.channel_first,
            "oh_inputs": self.get_onehot_input(x),
        }
        loss_kwargs.update(extra_info)
        loss, log_dict = self.loss(target, reconstructions, **loss_kwargs)

        # Compute reconstruction metrics (only updating, while actual computation is done in on_validation_epoch_end)
        self.compute_recons_metrics(target, reconstructions, split='val', return_dict=False)

        # Log metrics
        self.log_dict(
            log_dict,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.inner_validation_step(batch, batch_idx)
        return loss

    def configure_optimizers(self):
        partial_opts = instantiate_partial_optimizers_from_config(self.optimizer_config)

        key = 'autoencoder'
        opt = partial_opts.get(key, list(partial_opts.values())[0]) # create partial optimizer for the autoencoder
        weight_decay = find_weight_decay(opt, self.optimizer_config, key=key) # find weight decay for the autoencoder if set
        opt = opt(self.get_autoencoder_params(weight_decay=weight_decay if weight_decay != 0 else None)) # instantiate the optimizer with the autoencoder parameters, adjusted for weight decay
        
        if self.use_lr_scheduler and self.scheduler_config is not None:
            print("Setting up LR scheduler...")

            partial_schedulers = instantiate_partial_schedulers_from_config(self.scheduler_config)
            scheduler = partial_schedulers.get(key, list(partial_schedulers.values())[0])(
                opt, steps_per_epoch=self.num_steps_per_epoch, cycle_length=self.num_training_steps
            )

            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }

            return [opt], [scheduler_dict]
        return opt

    def get_validation_epoch_end_metrics(self):
        return [self.val_recons_metrics]

    # Log desired validation metrics only at the end of the epoch
    # (but they are accumulated during each validation step)
    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        log_metrics = {}
        for metric in self.get_validation_epoch_end_metrics():
            output_metric = metric.compute()
            log_metrics.update(output_metric)
            # remember to reset metrics at the end of the epoch
            metric.reset()

        self.log_dict(
            log_metrics,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

class AutoencoderDiscriminatorTrainingWrapper(AutoencoderTrainingWrapper):

    IGNORED_HPARAMS = ['discriminator']

    def __init__(
        self,
        *args,
        disc_start_iter=0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False # Important: This property activates manual optimization.
        self.disc_start_iter = disc_start_iter
   
    def get_discriminator_params(self, weight_decay=None) -> list:
        modules = []
        
        if hasattr(self.loss, "get_trainable_parameters"):
            modules.append(
                {'module': self.loss, 'named_params': self.loss.get_trainable_parameters(named=True)}
            )
            
        params = collect_params_from_modules(modules, weight_decay, check_skip_list=True)
        return params
    
    def training_step(self, batch, batch_idx):
        opts = self.optimizers()
        if not isinstance(opts, list):
            # Non-adversarial case
            opts = [opts]
        optimizer_idx = batch_idx % len(opts)
        if self.global_step < self.disc_start_iter:
            optimizer_idx = 0
        opt = opts[optimizer_idx]
        opt.zero_grad()
        with opt.toggle_model():
            loss = self.inner_training_step(
                batch, batch_idx,
                loss_kwargs=dict(
                    optimizer_idx=optimizer_idx, global_step=self.global_step,
                    last_layer=self.get_last_layer(),
                )
            )
            self.manual_backward(loss)
        opt.step()

        # Update LR with schedulers (done manually because of manual optimization)
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            if not isinstance(schedulers, list):
                schedulers = [schedulers]

            for sch in schedulers:
                sch.step() # TODO is it correct to call step() for every scheduler on every step? Or the scheduler for the AE should be called only when optimizer_idx == 0?

    def validation_step(self, batch, batch_idx):
        loss = self.inner_validation_step(
            batch, batch_idx,
            loss_kwargs=dict(
                optimizer_idx=0, global_step=self.global_step,
                last_layer=self.get_last_layer(),
            )
        )
        return loss
   
    def configure_optimizers(self):

        partial_opts = instantiate_partial_optimizers_from_config(self.optimizer_config)

        # Autoencoder optimizer       
        ae_key = 'autoencoder'
        opt_ae = partial_opts.get(ae_key, list(partial_opts.values())[0]) # create partial optimizer for the autoencoder
        weight_decay = find_weight_decay(opt_ae, self.optimizer_config, key=ae_key) # find weight decay for the autoencoder if set
        opt_ae = opt_ae(self.get_autoencoder_params(weight_decay=weight_decay if weight_decay != 0 else None)) # instantiate the optimizer with the autoencoder parameters, adjusted for weight decay
        
        opts = [opt_ae]

        # Discriminator optimizer
        disc_params = self.get_discriminator_params()
        if len(disc_params) > 0:
            disc_key = 'discriminator'
            opt_disc = partial_opts.get(disc_key, list(partial_opts.values())[1])
            weight_decay = find_weight_decay(opt_disc, self.optimizer_config, key=disc_key)
            opt_disc = opt_disc(self.get_discriminator_params(weight_decay=weight_decay if weight_decay != 0 else None))
            opts.append(opt_disc)

        # LR schedulers
        if self.use_lr_scheduler and self.scheduler_config is not None:
            print("Setting up LR schedulers...")

            partial_schedulers = instantiate_partial_schedulers_from_config(self.scheduler_config)

            #### Autoencoder setup ####
            ae_dict = {
                "optimizer": opt_ae,
            }

            ae_scheduler_fn = partial_schedulers.get(ae_key, None)
            
            if ae_scheduler_fn is not None:
                ae_scheduler_fn = ae_scheduler_fn(
                    opt_ae, steps_per_epoch=self.num_steps_per_epoch, total_training_steps=self.num_training_steps
                )
            
                ae_scheduler_dict = {
                    "scheduler": ae_scheduler_fn,
                    "interval": "step",
                    "frequency": 1,
                }

                ae_dict["lr_scheduler"] = ae_scheduler_dict

            out = [ae_dict]

            #### Discriminator setup ####
            if len(disc_params) > 0:

                disc_dict = {
                    "optimizer": opt_disc,
                }

                disc_scheduler_fn = partial_schedulers.get(disc_key, None)

                if disc_scheduler_fn is not None:
                    disc_scheduler_fn = disc_scheduler_fn(
                        opt_disc, steps_per_epoch=self.num_steps_per_epoch, total_training_steps=self.num_training_steps
                    )
                
                    disc_scheduler_dict = {
                        "scheduler": disc_scheduler_fn,
                        "interval": "step",
                        "frequency": 1,
                    }

                    disc_dict["lr_scheduler"] = disc_scheduler_dict

                out.append(disc_dict)

            return out

        return opts
