import torch
import torch.nn as nn

from snpgen.utils import instantiate_from_config
from snpgen.models.modules.utils import get_proper_state_dict_vae

class Autoencoder(nn.Module):
    def __init__(
        self,
        encoder_config,
        decoder_config,
        ckpt_path=None,
        load_ema_ckpt=True
    ):
        super().__init__()
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

        # Encoder architecture
        self.encoder = instantiate_from_config(encoder_config)

        # Decoder Architecture
        self.decoder = instantiate_from_config(decoder_config)

        if ckpt_path is not None:
            state_dict = get_proper_state_dict_vae(ckpt_path, ema=load_ema_ckpt)
            self.load_state_dict(state_dict, strict=True)

    def get_cls_layer(self):
        return self.decoder.get_cls_layer()
    
    def get_last_layer(self):
        return self.decoder.get_last_layer()

    def encode(self, x, sample=False):
        mu, logvar = self.encoder(x)
        if sample:
            z = self.reparameterize(mu, logvar)
            return z
        else:
            return mu, logvar

    def decode(self, z, argmax=True):
        return self.decoder(z, argmax=argmax)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # randn_like automatically uses the same device as the input tensor
        return mu + eps * std
    
    def forward(self, z, argmax=True):
        # Only sample during inference
        decoder_out = self.decode(z, argmax=argmax)
        return decoder_out
    
    def forward_recons(self, x, argmax=True, sample_posterior=True):
        # For generating reconstructions during inference
        mu, logvar = self.encode(x)
        if sample_posterior:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        decoder_out = self.decode(z, argmax=argmax)
        return decoder_out

    def sample(self, num_samples, argmax=True):
        z_shape = [self.decoder_config.params['z_dim']]
        
        # Check if the decoder has a z_channels attribute,
        # meaning that the noise vector is multi-dimensional
        if hasattr(self.decoder_config.params, 'z_channels'):
            z_shape = [self.decoder_config.params['z_channels']] + z_shape
            
        z = torch.randn(num_samples, *z_shape,
                        device=next(self.parameters()).device)
        
        return self.decode(z, argmax=argmax)
    
    
class IdentityAutoencoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()

    def encode(self, x, sample=False):
        return self.encoder(x)

    def decode(self, z, argmax=True):
        return self.decoder(z)

    def forward(self, z, argmax=True):
        return self.decode(z, argmax=argmax)

    def forward_recons(self, x, argmax=True, sample_posterior=True):
        return self.decode(self.encode(x), argmax=argmax)

    def sample(self, num_samples, argmax=True):
        raise NotImplementedError("Cannot sample from identity autoencoder")